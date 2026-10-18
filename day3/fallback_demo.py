"""
浅层回退机制：Worker 失败 → Supervisor 重分派
"""
import os
import random
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))


# ==================================================
# 1. 定义状态
# ==================================================
class FallbackState(TypedDict):
    task:str
    current_worker:str
    worker_result:str
    worker_status:str
    failure_reason:str
    retry_count:int
    final_answer:str


# ==================================================
# 2. Supervisor 节点
# ==================================================
def supervisor_node(state:FallbackState) -> Command[Literal["worker_a","worker_b","__end__"]]:
    status=state.get("worker_status","pending")
    retry_count=state.get("retry_count",0)
    current=state.get("current_worker","")

    print(f"\n🎯 [Supervisor] 状态: {status}, 重试次数: {retry_count}, 当前 Worker: {current}")

    if status=="success":
        print(f"   ✅ 任务完成，结果: {state['worker_result'][:50]}...")
        return Command(
            goto="__end__",
            update={"final_answer":state["worker_result"]}
        )

    if status=="failed" and retry_count>=3:
        print(f"   💀 重试 {retry_count} 次仍失败，放弃")
        return Command(
            goto="__end__",
            update={"final_answer":f"任务失败：已尝试 {retry_count} 次"}
        )

    if status=="failed":
        print(f"   🔄 失败原因: {state.get('failure_reason','未知')}")
        if current=="worker_a":
            print(f"   🔀 切换到 worker_b")
            return Command(
                goto="worker_b",
                update={"current_worker":"worker_b","worker_status":"pending"}
            )
        else:
            print(f"   🔀 切换到 worker_a")
            return Command(
                goto="worker_a",
                update={"current_worker":"worker_a","worker_status":"pending"}
            )

    print(f"   🚀 首次执行，使用 worker_a")
    return Command(
        goto="worker_a",
        update={"current_worker":"worker_a","worker_status":"pending"}
    )


# ==================================================
# 3. Worker A 节点（60% 失败率）
# ==================================================
def worker_a_node(state:FallbackState) -> Command[Literal["supervisor"]]:
    print(f"   🔧 [Worker A] 处理: {state['task'][:40]}...")

    if random.random()<0.6:
        print(f"   ❌ [Worker A] 失败")
        return Command(
            goto="supervisor",
            update={
                "worker_status":"failed",
                "failure_reason":"Worker A 服务暂时不可用",
                "retry_count":state.get("retry_count",0)+1
            }
        )

    result=f"Worker A 成功处理：{state['task']}"
    print(f"   ✅ [Worker A] 成功")
    return Command(
        goto="supervisor",
        update={
            "worker_status":"success",
            "worker_result":result
        }
    )


# ==================================================
# 4. Worker B 节点（30% 失败率）
# ==================================================
def worker_b_node(state:FallbackState) -> Command[Literal["supervisor"]]:
    print(f"   🔧 [Worker B] 处理: {state['task'][:40]}...")

    if random.random()<0.3:
        print(f"   ❌ [Worker B] 失败")
        return Command(
            goto="supervisor",
            update={
                "worker_status":"failed",
                "failure_reason":"Worker B 服务暂时不可用",
                "retry_count":state.get("retry_count",0)+1
            }
        )

    result=f"Worker B 成功处理：{state['task']}"
    print(f"   ✅ [Worker B] 成功")
    return Command(
        goto="supervisor",
        update={
            "worker_status":"success",
            "worker_result":result
        }
    )


# ==================================================
# 5. 构建图
# ==================================================
builder=StateGraph(FallbackState)
builder.add_node("supervisor",supervisor_node)
builder.add_node("worker_a",worker_a_node)
builder.add_node("worker_b",worker_b_node)

builder.add_edge(START,"supervisor")

graph=builder.compile()

# ==================================================
# 6. 测试
# ==================================================
def main():
    print("=" * 60)
    print("🧪 浅层回退：Worker 失败 → Supervisor 重分派")
    print("=" * 60)    

    random.seed(42)

    result=graph.invoke({
        "task":"研究 2026 年 AI 芯片市场",
        "current_worker":"",
        "worker_result":"",
        "worker_status":"pending",
        "failure_reason":"",
        "retry_count":0,
        "final_answer":"",
    })

    print(f"\n{'=' * 60}")
    print("📄 最终结果：")
    print("=" * 60)
    print(result["final_answer"])
    print(f"\n总重试次数: {result['retry_count']}")

if __name__ == "__main__":
    main()
