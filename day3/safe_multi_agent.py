"""
给多 Agent 加安全阀
综合运用：浅层回退 + 迭代控制 + 循环检测 + 降级
"""
import os
import random
from typing import TypedDict, Literal, Annotated
import operator
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))


# ==================================================
# 1. 状态定义
# ==================================================
class SafeAgentState(TypedDict):
    task:str
    worker_status:str
    worker_result:str
    current_worker:str
    iteration_count:int
    retry_count:int
    decision_history:list
    failure_history:Annotated[list,operator.add]
    final_answer:str
    exit_reason:str


# ==================================================
# 2. 配置常量
# ==================================================
MAX_ITERATIONS=6
MAX_RETRIES=4
REPEAT_THRESHOLD=3


# ==================================================
# 3. Supervisor 节点（带全套安全阀）
# ==================================================
def supervisor_node(state:SafeAgentState) -> Command[Literal["worker_a","worker_b","worker_c","__end__"]]:
    iteration=state.get("iteration_count",0)+1
    retries=state.get("retry_count",0)
    history=state.get("decision_history",[])
    status=state.get("worker_status","pending")

    print(f"\n{'─' * 50}")
    print(f"🎯 [Supervisor] 迭代 {iteration}/{MAX_ITERATIONS} | 重试 {retries}/{MAX_RETRIES} | 状态: {status}")

    if iteration>MAX_ITERATIONS:
        print(f"   ⛔ [安全阀1] 超过最大迭代次数")
        return Command(
            goto="__end__",
            update={
                "final_answer":state.get("worker_result") or "任务未完成",
                "exit_reason":f"超过最大迭代次数 ({MAX_ITERATIONS})",
                "iteration_count":iteration,
            }
        )

    if retries>MAX_RETRIES:
        print(f"   ⛔ [安全阀2] 超过最大重试次数")
        return Command(
            goto="__end__",
            update={
                "final_answer":state.get("worker_result") or "任务失败",
                "exit_reason":f"超过最大重试次数({MAX_RETRIES})",
                "iteration_count":iteration,
            }
        )

    if status=="success":
        print(f"   ✅ 任务完成")
        return Command(
            goto="__end__",
            update={
                "final_answer":state["worker_result"],
                "exit_reason":"success",
                "iteration_count":iteration,
            }
        )

    if status=="pending" and not history:
        decision="worker_a"
    else:
        workers=["worker_a","worker_b","worker_c"]
        last=history[-1] if history else "worker_c"
        idx=(workers.index(last)+1)%len(workers)
        decision=workers[idx]

    new_history=history+[decision]
    if len(new_history)>=REPEAT_THRESHOLD:
        recent=new_history[-REPEAT_THRESHOLD]
        if len(set(recent))==1:
            print(f"   ⛔ [安全阀3] 重复决策 {decision} × {REPEAT_THRESHOLD}")
            return Command(
                goto="__end__",
                update={
                    "final_answer":state.get("worker_result") or "任务未完成",
                    "exit_reason":f"重复决策({decision})",
                    "iteration_count":iteration,
                    "decision_history":new_history,
                }
            )

    print(f"   🔀 决策: {decision}")
    return Command(
        goto=decision,
        update={
            "iteration_count":iteration,
            "decision_history":new_history,
            "current_worker":decision,
            "worker_status":"pending",
        }
    )


# ==================================================
# 4. 通用 Worker 工厂
# ==================================================
def make_worker(name:str,fail_rate:float):
    """创建一个 Worker 节点函数"""
    def worker_node(state:SafeAgentState) -> Command[Literal["supervisor"]]:
        print(f"   🔧 [{name}] 处理中...")

        if random.random()<fail_rate:
            print(f"   ❌ [{name}] 失败")
            return Command(
                goto="supervisor",
                update={
                    "worker_status":"failed",
                    "retry_count":state.get("retry_count",0)+1,
                    "failure_history":[f"{name}: 服务失败"],
                }
            )

        print(f"   ✅ [{name}] 成功")
        return Command(
            goto="supervisor",
            update={
                "worker_status":"success",
                "worker_result":f"{name} 完成了任务",
            }
        )
    return worker_node


# ==================================================
# 5. 构建图
# ==================================================
builder=StateGraph(SafeAgentState)
builder.add_node("supervisor",supervisor_node)
builder.add_node("worker_a",make_worker("Worker A",0.6))
builder.add_node("worker_b",make_worker("Worker B",0.5))
builder.add_node("worker_c",make_worker("Worker C",0.3))

builder.add_edge(START,"supervisor")
graph=builder.compile()


# ==================================================
# 6. 测试
# ==================================================
def run_test(name:str,seed:int):
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
                "current_worker":"",
                "iteration_count":0,
                "retry_count":0,
                "decision_history":[],
                "failure_history":[],
                "final_answer":"",
                "exit_reason":"",
            },
            config={"configurable":{"thread_id":f"test-{seed}"},"recursion_limit":50},
        )

        print(f"\n{'=' * 60}")
        print("📊 执行总结")
        print("=" * 60)
        print(f"最终结果: {result['final_answer']}")
        print(f"退出原因: {result['exit_reason']}")
        print(f"迭代次数: {result['iteration_count']}")
        print(f"重试次数: {result['retry_count']}")
        print(f"决策历史: {result['decision_history']}")
        print(f"失败历史: {result['failure_history']}")

    except Exception as e:
        print(f"\n💥 异常: {type(e).__name__}: {e}")

if __name__ == "__main__":
    run_test("测试1：可能成功",seed=1)
    run_test("测试2：可能触发重试上限",seed=2)
    run_test("测试3：可能触发迭代上限",seed=3)
