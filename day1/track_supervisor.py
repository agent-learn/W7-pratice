"""
用中间件追踪 Supervisor 的路由决策
"""
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import AgentMiddleware
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing import Any

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

model=ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)


# ==================================================
# 工具
# ==================================================
@tool
def web_search(query:str) -> str:
    """搜索信息。"""
    return "市场规模约5000亿美元。"

@tool
def multiply(a:float,b:float) -> float:
    """计算乘积。"""
    return a*b


# ==================================================
# 追踪中间件
# ==================================================
class RouteTracker(AgentMiddleware):

    def wrap_tool_call(self, request, handler) -> Any:
        tool_name=request.tool_call['name']

        if tool_name in ("research_expert","math_expert"):
            print(f"\n🔀 [路由] Supervisor 分派到: {tool_name}")
            print(f"    参数：{request.tool_call['args']}")

        result=handler(request)

        if tool_name in ("research_expert","math_expert"):
            print(f"   ✅ [返回] {tool_name} 完成")

        return result


# ==================================================
# Worker
# ==================================================
research_agent=create_agent(
    model=model,
    tools=[web_search],
    system_prompt="你是研究专家。",
)

math_agent=create_agent(
    model=model,
    tools=[multiply],
    system_prompt="你是数学专家。",
)

@tool("research_expert",description="研究专家。")
def call_research_agent(query:str) -> str:
    result=research_agent.invoke({"messages":[{"role":"user","content":query}]})
    return result["messages"][-1].content

@tool("math_expert",description="数学专家。")
def call_math_agent(query:str) -> str:
    result=math_agent.invoke({"messages":[{"role":"user","content":query}]})
    return result["messages"][-1].content


# ==================================================
# Supervisor（带中间件）
# ==================================================
supervisor=create_agent(
    model=model,
    tools=[call_research_agent,call_math_agent],
    system_prompt="需要查询时调用 research_expert，需要计算时调用 math_expert。",
    middleware=[RouteTracker()],
)

def main():
    config={"configurable":{"thread_id":"track-1"}}
    result=supervisor.invoke(
        {"messages":[HumanMessage(content="查AI芯片市场规模，算15%增长3年后多少")]},
        config,    
    )
    print(f"\n💬 最终回答: {result['messages'][-1].content}")

if __name__ == "__main__":
    main()
