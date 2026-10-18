"""
使用 create_supervisor 跑通 Supervisor 分派 2 个 Worker
改用 subagents 模式
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph_supervisor import create_supervisor
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

# ==================================================
# 1. 创建 LLM
# ==================================================
model=ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)


# ==================================================
# 2. 定义 Worker 的工具
# ==================================================
@tool
def web_search(query:str) -> str:
    """搜索网络信息。"""
    return "2026年全球AI芯片市场规模约为5000亿美元。"

@tool
def multiply(a:float,b:float) -> float:
    """计算两个数的乘积。"""
    return a*b


# ==================================================
# 3. 用 create_agent 创建两个 Worker
# ==================================================
research_agent=create_agent(
    model=model,
    tools=[web_search],
    system_prompt="你是一个研究专家，使用搜索工具查询信息。",
)

math_agent=create_agent(
    model=model,
    tools=[multiply],
    system_prompt="你是一个数学专家，使用计算工具进行计算。",
)


# ==================================================
# 4. 把 Worker 包装成工具（核心步骤）
# ==================================================
@tool("research_expert",description="研究专家，用于查询当前事件和网络信息。")
def call_research_agent(query:str) -> str:
    """调用研究专家 Agent。"""
    result=research_agent.invoke({
        "messages":[{"role":"user","content":query}]
    })
    return result["messages"][-1].content

@tool("math_expert",description="数学专家，用于执行计算。")
def call_math_agent(query:str) -> str:
    """调用数学专家 Agent。"""
    result=math_agent.invoke({
        "messages":[{"role":"user","content":query}]
    })
    return result["messages"][-1].content


# ==================================================
# 5. 创建 Supervisor
# ==================================================
supervisor=create_agent(
    model=model,
    tools=[call_research_agent,call_math_agent],
    system_prompt="""你是一个团队主管，管理研究专家和数学专家。

需要查询信息时，调用 research_expert。
需要计算时，调用 math_expert。
如果任务需要多个步骤，按顺序调用多个专家。""",
)


# ==================================================
# 6. 测试
# ==================================================
def main():
    print("=" * 60)
    print("🧪 Supervisor 分派 2 个 Worker（subagents 模式）")
    print("=" * 60)

    result=supervisor.invoke({
        "messages":[HumanMessage(content="查一下2026年AI芯片市场规模，然后计算如果年增长率为15%，3年后是多少")]
    })

    print(f"\n💬 最终回答：\n{result['messages'][-1].content}")

    print("\n📨 消息传递流程：")
    for i,msg in enumerate(result["messages"]):
        msg_type=type(msg).__name__
        content=str(msg.content)[:60] if msg.content else "(空)"
        tool_info=""
        if hasattr(msg,"tool_calls") and msg.tool_calls:
            tool_info=f" [调用：{[tc['name'] for tc in msg.tool_calls]}]"
        print(f"  [{i}] {msg_type}{tool_info}: {content}...")

if __name__ == "__main__":
    main()
