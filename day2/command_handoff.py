"""
Command 与 Handoff 基础示例
演示：节点返回 Command 实现路由 + 状态更新
"""
import os
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command

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
class RouteState(TypedDict):
    messages:list
    next_agent:str


# ==================================================
# 2. 定义节点（返回 Command）
# ==================================================
def router_node(state:RouteState) -> Command[Literal["greeter","calculator","__end__"]]:
    """路由节点：根据用户输入决定下一个 Agent"""
    last_msg=state["messages"][-1].content if state["messages"] else ""

    if "你好" in last_msg or "hello" in last_msg.lower():
        print("🔀 [Router] → greeter")
        return Command(
            goto="greeter",
            update={"next_agent":"greeter"}
        )
    elif "计算" in last_msg or "+" in last_msg or "*" in last_msg:
        print("🔀 [Router] → calculator")
        return Command(
            goto="calculator",
            update={"next_agent":"calculator"}
        )
    else:
        print("🔀 [Router] → END")
        return Command(goto=END,update={"next_agent":"end"})

def greeter_node(state:RouteState) -> Command[Literal["__end__"]]:
    """问候 Agent"""
    print("👋 [Greeter] 执行问候")
    response=AIMessage(content="你好！我是问候助手，很高兴认识你。")
    return Command(
        goto=END,
        update={"messages":state["messages"]+[response]}
    )

def calculator_node(state:RouteState) -> Command[Literal["__end__"]]:
    """计算 Agent"""
    print("🧮 [Calculator] 执行计算")
    response=AIMessage(content="计算完成：结果是 100。")
    return Command(
        goto=END,
        update={"messages":state["messages"]+[response                                     ]}
    )


# ==================================================
# 3. 构建图
# ==================================================
builder=StateGraph(RouteState)
builder.add_node("router",router_node)
builder.add_node("greeter",greeter_node)
builder.add_node("calculator",calculator_node)

builder.add_edge(START,"router")

graph=builder.compile()


# ==================================================
# 4. 测试
# ==================================================
def test(input_text:str):
    print(f"\n{'=' * 50}")
    print(f"输入：{input_text}")
    result=graph.invoke({
        "messages":[HumanMessage(content=input_text)],
        "next_agent":""
    })
    print(f"最终消息数：{len(result['messages'])}")
    for msg in result["messages"]:
        print(f"  {type(msg).__name__}: {msg.content[:60]}")

if __name__ == "__main__":
    test("你好")
    test("帮我计算 25 * 4")
    test("今天天气怎么样")
