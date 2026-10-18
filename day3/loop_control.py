"""
避免死循环：最大迭代次数 + 循环检测
"""
import os
import random
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))


# ==================================================
# 1. 定义状态（含迭代计数和决策历史）
# ==================================================
class LoopState(TypedDict):
    task:str
    worker_status:str
    worker_result:str
    iteration_count:int
    decision_history:list
    final_answer:str


# ==================================================
# 2. Supervisor 节点（带循环控制）
# ==================================================
MAX_ITERATIONS=5
REPEAT_THRESHOLD=3

def supervisor_node(state:LoopState) -> Command[Literal["worker_a","worker_b","__end__"]]:
    iteration_count=state.get("iteration_count",0)+1
    history=state.get("decision_history",[])
    status=state.get("worker_status","pending")

    print(f"\n🎯 [Supervisor] 迭代 {iteration_count}/{MAX_ITERATIONS} | 状态: {status}")

    if iteration_count>MAX_ITERATIONS:
        print(f"   ⛔ 超过最大迭代次数 ({MAX_ITERATIONS})，强制退出")
        return Command(
            goto="__end__",
            update={
                "final_answer":f"任务未完成（超过 {MAX_ITERATIONS} 次迭代）",
                "iteration_count":iteration_count,
            }
        )

    if status=="success":
        print(f"   ✅ 任务完成")
        return Command(
            goto="__end__",
            update={
                "final_answer":state["worker_result"],
                "iteration_count":iteration_count,
            }
        )

    if status=="pending" or len(history)==0:
        decision="worker_a"
    else:
        decision="worker_b" if history[-1]=="worker_a" else "worker_a"

    new_history=history+[decision]
    if len(new_history)>=REPEAT_THRESHOLD:
        recent=new_history[-REPEAT_THRESHOLD:]
        if len(set(recent))==1:
            print(f"   ⛔ 检测到重复决策 ({decision} × {REPEAT_THRESHOLD})，强制退出")
            return Command(
                goto="__end__",
                update={
                    "final_answer":"任务未完成（检测到重复决策）",
                    "iteration_count":iteration_count,
                    "decision_history":new_history,
                }
            )

    print(f"   🔀 决策: {decision}")
    return Command(
        goto=decision,
        update={
            "iteration_count":iteration_count,
            "decision_history":new_history,
            "worker_status":"pending",
        }
    )


# ==================================================
# 3. Worker A（70% 失败率）
# ==================================================
def worker_a_node(state:LoopState) -> Command[Literal["supervisor"]]:
    print(f"   🔧 [Worker A] 处理中...")
    if random.random()<0.7:
        print(f"   ❌ [Worker A] 失败")
        return Command(
            goto="supervisor",
            update={"worker_status":"failed"}
        )
    print(f"   ✅ [Worker A] 成功")
    return Command(
        goto="supervisor",
        update={"worker_status":"success","worker_result":"Worker A 完成"}
    )


# ==================================================
# 4. Worker B（70% 失败率）
# ==================================================
def worker_b_node(state:LoopState) -> Command[Literal["supervisor"]]:
    print(f"   🔧 [Worker B] 处理中...")
    if random.random()<0.7:
        print(f"   ❌ [Worker B] 失败")
        return Command(
            goto="supervisor",
            update={"worker_status":"failed"}
        )

    print(f"   ✅ [Worker B] 成功")
    return Command(
        goto="supervisor",
        update={"worker_status":"success","worker_result":"Worker B 完成"}
    )


# ==================================================
# 5. 构建图
# ==================================================
builder=StateGraph(LoopState)
builder.add_node("supervisor",supervisor_node)
builder.add_node("worker_a",worker_a_node)
builder.add_node("worker_b",worker_b_node)

builder.add_edge(START,"supervisor")
graph=builder.compile()


# ==================================================
# 6. 测试
# ==================================================
def test_case(name:str,config:dict,seed:int):
    print(f"\n{'=' * 60}")
    print(f"🧪 {name}")
    print("=" * 60)

    random.seed(seed)

    try:
        result=graph.invoke(
            {
                "task":"研究 AI 芯片市场",
                "worker_status":"pending",
                "worker_result":"",
                "iteration_count":0,
                "decision_history":[],
                "final_answer":"",
            },
            config,
        )
        print(f"\n📄 结果: {result['final_answer']}")
        print(f"   迭代次数: {result['iteration_count']}")
        print(f"   决策历史: {result['decision_history']}")

    except Exception as e:
        print(f"\n💥 异常: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_case(
        "测试1：正常情况（应用级迭代控制）",
        config={"configurable":{"thread_id":"test-1"},"recursion_limit":50},
        seed=1,
    )

    test_case(
        "测试2：框架级兜底（recursion_limit=3）",
        config={"configurable":{"thread_id":"test-2"},"recursion_limit":3},
        seed=2,
    )




























