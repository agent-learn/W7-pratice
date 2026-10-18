"""
Supervisor/Worker 模式示例
使用工具封装子 Agent（官方推荐方式）
"""
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

# ==================================================
# 1. 创建 LLM
# ==================================================
model=ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

# ==================================================
# 2. 定义 Worker 的工具
# ==================================================
@tool
def web_search(query:str) -> str:
    """搜索网络信息。"""
    return f"关于'{query}'的搜索结果：..."

@tool
def calculate(expression:str) -> str:
    """执行数学计算。"""
    try:
        result=eval(expression,{"__builtins__":{}},{})
        return f"{expression} = {result}"
    except:
        return "计算失败"

# ==================================================
# 3. 创建 Worker Agent
# ==================================================
research_agent=create_agent(
    model=model,
    tools=[web_search],
    system_prompt="你是一个研究专家，擅长搜索和整理信息。",
)

math_agent=create_agent(
    model=model,
    tools=[calculate],
    system_prompt="你是一个数学专家，擅长计算。"
)

# ==================================================
# 4. 把 Worker 包装成工具
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
# 5. 创建 Supervisor（主 Agent）
# ==================================================
supervisor=create_agent(
    model=model,
    tools=[call_research_agent,call_math_agent],
    system_prompt="""你是一个协调者。

根据用户请求，调用合适的专家：
- 需要查询信息时，调用 research_expert
- 需要计算时，调用 math_expert

如果任务需要多个步骤，按顺序调用多个专家。""",
)

# ==================================================
# 6. 测试
# ==================================================
def main():
    print("=" * 60)
    print("🧪 Supervisor/Worker 模式")
    print("=" * 60)

    # 测试1：只调用研究专家
    print("\n--- 测试1：研究任务 ---")    
    result=supervisor.invoke({
        "messages":[HumanMessage(content="帮我查一下2026年AI芯片的最新进展")]
    })
    print(f"回答：{result['messages'][-1].content[:200]}...")

    # 测试2：只调用数学专家
    print("\n--- 测试2：计算任务 ---")
    result=supervisor.invoke({
        "messages":[HumanMessage(content="计算 25 * 48")]
    })
    print(f"回答：{result['messages'][-1].content[:200]}...")

    # 测试3：需要两个专家协作
    print("\n--- 测试3：复合任务 ---")
    result=supervisor.invoke({
        "messages":[HumanMessage(content="查一下2026年全球AI芯片市场规模，然后计算如果年增长率为15%，3年后是多少")]
    })
    print(f"回答：{result['messages'][-1].content[:200]}...")

if __name__ == "__main__":
    main()
