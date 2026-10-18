"""
信息收集多 Agent 系统
规划 → 并行采集 → 汇总报告
"""
import os
import operator
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

model = ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)


# ==================================================
# 1. 定义状态
# ==================================================
class ResearchState(TypedDict):
    question:str 
    tasks:list[str]
    findings:Annotated[list[dict],operator.add]
    report:str


# ==================================================
# 2. 规划节点：拆解问题
# ==================================================
def plan_node(state:ResearchState) -> dict:
    question=state["question"]
    print(f"\n📋 [Plan] 研究问题: {question}")

    prompt=f"""请把以下研究问题拆解成 3-4 个独立的采集任务：

问题：{question}

请以 JSON 数组格式返回任务列表，每个任务是一个字符串。例如：
["任务1", "任务2", "任务3"]

只返回 JSON 数组，不要其他内容。"""

    response=model.invoke([HumanMessage(content=prompt)])

    import json
    try:
        tasks=json.loads(response.content)
    except:
        tasks=[
            f"{question}的背景信息",
            f"{question}的最新数据",
            f"{question}的专家观点",
        ]

    print(f"   拆分为 {len(tasks)} 个采集任务")
    for t in tasks:
        print(f"     - {t}")

    return {"tasks":tasks}


# ==================================================
# 3. 分发节点：Send 并行分发
# ==================================================
def dispatch_tasks(state:ResearchState) -> list[Send]:
    sends=[]
    for task in state["tasks"]:
        sends.append(Send("collector",{"task":task}))
    return sends


# ==================================================
# 4. 采集节点：处理单个采集任务
# ==================================================
def collector_node(state:dict) -> dict:
    task=state["task"]
    print(f"\n🔍 [Collector] 采集: {task}")

    prompt=f"""请针对以下采集任务，提供一段简洁的信息摘要（100字以内）：

任务：{task}

摘要："""

    response=model.invoke([HumanMessage(content=prompt)])
    content=response.content

    finding={
        "task":task,
        "content":content,
        "source":"llm_simulated",
    }

    print(f"   ✅ 采集完成: {task}")
    print(f"      内容: {content[:80]}...")

    return {"findings":[finding]}


# ==================================================
# 5. 汇总节点：生成最终报告
# ==================================================
def aggregate_node(state:ResearchState) -> dict:
    findings=state["findings"]
    question=state["question"]

    print(f"\n📊 [Aggregate] 合并 {len(findings)} 个采集结果")

    findings_text="\n\n".join([
        f" 【来源 {i+1}】{f['task']}\n{f['content']}"
        for i,f in enumerate(findings)
    ])

    prompt=f"""请根据以下采集结果，针对问题「{question}」撰写一份简洁的研究报告。

采集结果：
{findings_text}

请用 Markdown 格式撰写报告，包含：
1. 核心发现（2-3 条）
2. 关键数据
3. 简要结论

报告："""

    response=model.invoke([HumanMessage(content=prompt)])
    report=response.content

    print(f"   ✅ 报告生成完成（{len(report)} 字）")
    return {"report":report}


# ==================================================
# 6. 构建图
# ==================================================
builder=StateGraph(ResearchState)

builder.add_node("plan",plan_node)
builder.add_node("collector",collector_node)
builder.add_node("aggregate",aggregate_node)

builder.add_edge(START,"plan")

builder.add_conditional_edges("plan",dispatch_tasks,["collector"])

builder.add_edge("collector","aggregate")
builder.add_edge("aggregate",END)

graph=builder.compile()


# ==================================================
# 7. 测试
# ==================================================
def main():
    print("=" * 60)
    print("🧪 信息收集多 Agent 系统")
    print("=" * 60)

    result=graph.invoke({
        "question":"2026年全球AI芯片市场的竞争格局",
        "tasks":[],
        "findings":[],
        "report":"",
    })

    print(f"\n{'=' * 60}")
    print("📄 最终研究报告：")
    print("=" * 60)
    print(result["report"])

    print(f"\n📊 采集统计：")
    print(f"   采集任务数: {len(result['tasks'])}")
    print(f"   采集结果数: {len(result['findings'])}")

if __name__ == "__main__":
    main()
