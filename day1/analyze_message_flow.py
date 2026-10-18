"""
分析 Supervisor 系统的消息传递流程
使用官方推荐的 subagents 模式
"""
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

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
    """搜索信息。"""
    return "2026年全球AI芯片市场规模约为5000亿美元。"

@tool
def multiply(a:float,b:float) -> float:
    """计算乘积。"""
    return a*b


# ==================================================
# 3. 用 create_agent 创建 Worker
# ==================================================
research_agent=create_agent(
    model=model,
    tools=[web_search],
    system_prompt="你是研究专家，使用搜索工具查询信息。",
)

math_agent=create_agent(
    model=model,
    tools=[multiply],
    system_prompt="你是数学专家，使用计算工具进行计算。",
)


# ==================================================
# 4. 把 Worker 包装成工具（核心）
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
checkpointer=MemorySaver()

supervisor=create_agent(
    model=model,
    tools=[call_research_agent,call_math_agent],
    system_prompt="""你是一个团队主管，管理研究专家和数学专家。

需要查询信息时，调用 research_expert。
需要计算时，调用 math_expert。
如果任务需要多个步骤，按顺序调用多个专家。""",
    checkpointer=checkpointer,
)


# ==================================================
# 6. 分析消息传递
# ==================================================
def analyze_flow():
    config={"configurable":{"thread_id":"flow-analysis-1"}}

    print("=" * 60)
    print("📨 消息传递流程分析")
    print("=" * 60)

    input_msg={
        "messages":[
            HumanMessage(
                content="查一下2026年AI芯片市场规模，然后计算如果增长率为15%，3年后是多少"
            )
        ]
    }

    step=0
    for chunk in supervisor.stream(input_msg,config,stream_mode="updates"):
        for node_name,update in chunk.items():
            step+=1
            print(f"\n{'─' * 50}")
            print(f"Step {step} | 节点: {node_name}")

            if "messages" in update:
                for msg in update["messages"]:
                    msg_type=type(msg).__name__
                    print(f"  📩 {msg_type}")

                    if isinstance(msg,AIMessage):
                        if msg.tool_calls:
                            for tc in msg.tool_calls:
                                print(f"     🛠️ 调用工具: {tc['name']}")
                                print(f"        参数：{tc['args']}")
                        if msg.content:
                            print(f"     💬 内容: {msg.content[:100]}...")

                    elif isinstance(msg,ToolMessage):
                        print(f"     🔧 工具: {msg.name}")
                        print(f"     📄 结果: {str(msg.content)[:100]}...")

                    elif isinstance(msg,HumanMessage):
                        print(f"     👤 用户: {msg.content[:100]}...")

    print("\n" + "=" * 60)
    print("📊 最终消息列表")
    print("=" * 60)

    state=supervisor.get_state(config)
    for i,msg in enumerate(state.values["messages"]):
        msg_type=type(msg).__name__
        content=str(msg.content)[:60] if msg.content else "(无内容)"
        tool_info=""
        if isinstance(msg,AIMessage) and msg.tool_calls:
            tool_info=f"  [tool_calls: {[tc['name'] for tc in msg.tool_calls]}]"
        print(f"  [{i}] {msg_type}{tool_info}: {content}")

if __name__ == "__main__":
    analyze_flow()
