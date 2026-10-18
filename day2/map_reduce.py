"""
任务拆分与结果合并：Map-Reduce 模式
演示：Send API + reducer 实现并行处理
"""
import os
import operator
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

model=ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)


# ==================================================
# 1. 定义状态
# ==================================================
class MapReduceState(TypedDict):
    question:str
    tasks:list[str]
    results:Annotated[list[str],operator.add]
    final_answer:str


# ==================================================
# 2. 规划节点：把问题拆成子任务
# ==================================================
def plan_node(state:MapReduceState) -> dict:
    question=state["question"]
    print(f"\n📋 [Plan] 拆解问题: {question}")

    tasks=[
        f"关于「{question}」的背景信息",
        f"关于「{question}」的最新数据",
        f"关于「{question}」的专家观点",
    ]
    print(f"    拆分为 {len(tasks)} 个子任务")
    for t in tasks:
        print(f"    - {t}")

    return {"tasks":tasks}


# ==================================================
# 3. 路由函数：用 Send 分发任务
# ==================================================
def dispatch_tasks(state:MapReduceState) -> list[Send]:
    sends=[]
    for task in state["tasks"]:
        sends.append(Send("worker",{"task":task}))
    return sends


# ==================================================
# 4. Worker 节点：处理单个子任务
# ==================================================
def worker_node(state:dict) -> dict:
    task=state["task"]
    print(f"\n🔧 [Worker] 处理: {task}")

    result=f"「{task}」的处理结果：这是模拟数据。"
    print(f"   ✅ 完成: {task}")

    return {"results":[result]}


# ==================================================
# 5. 汇总节点：合并所有结果
# ==================================================
def aggregate_node(state:MapReduceState) -> dict:
    results=state["results"]
    print(f"\n📊 [Aggregate] 合并 {len(results)} 个结果")

    final=f"综合 {len(results)} 个来源的信息：\n" + "\n".join(
        f"- {r}" for r in results
    )
    return {"final_answer":final}


# ==================================================
# 6. 构建图
# ==================================================
builder=StateGraph(MapReduceState)

builder.add_node("plan",plan_node)
builder.add_node("worker",worker_node)
builder.add_node("aggregate",aggregate_node)

builder.add_edge(START,"plan")

builder.add_conditional_edges("plan",dispatch_tasks,["worker"])

builder.add_edge("worker","aggregate")
builder.add_edge("aggregate",END)

graph=builder.compile()


# ==================================================
# 7. 测试
# ==================================================
def main():
    print("=" * 60)
    print("🧪 Map-Reduce 模式：任务拆分与结果合并")
    print("=" * 60)

    result=graph.invoke({
        "question":"2026年AI芯片市场",
        "tasks":[],
        "results":[],
        "final_answer":"",
    })

    print(f"\n{'=' * 60}")
    print("📄 最终回答：")
    print("=" * 60)
    print(result["final_answer"])

    print(f"\n📊 原始结果列表（{len(result['results'])} 条）：")
    for i,r in enumerate(result["results"],1):
        print(f"  [{i}] {r}")

if __name__ == "__main__":
    main()
