"""
多 Agent 日志追踪
演示：结构化日志 + stream 三种模式
"""
import os
import time
import json
import uuid
from datetime import datetime
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import AgentMiddleware
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import Any

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

model=ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)


# ==================================================
# 1. 结构化日志器
# ==================================================
class StructuredLogger:

    def __init__(self,trace_id:str):
        self.trace_id=trace_id
        self.logs=[]
        self.start_time=time.time()

    def log(self,event:str,**kwargs):
        entry={
            "timestamp":datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "trace_id":self.trace_id,
            "event":event,
            "elapsed_ms":int((time.time()-self.start_time)*1000),
            **kwargs,
        }
        self.logs.append(entry)
        print(f"[{entry['timestamp']}] {event}: {json.dumps(kwargs,ensure_ascii=False)}")

    def summary(self):
        print(f"\n{'=' * 60}")
        print(f"📊 日志汇总 (trace_id={self.trace_id})")
        print("=" * 60)
        for log in self.logs:
            print(f"  [{log['elapsed_ms']:>5}ms] {log['event']}")


# ==================================================
# 2. 追踪中间件
# ==================================================
class TracingMiddleware(AgentMiddleware):

    def __init__(self,agent_name:str,logger:StructuredLogger):
        self.agent_name=agent_name
        self.logger=logger

    def wrap_model_call(self,request,handler):
        start=time.time()
        result=handler(request)
        elapsed=time.time()-start

        self.logger.log(
            "model_call",
            agent=self.agent_name,
            elapsed_ms=int(elapsed*1000),
        )
        return result

    def wrap_tool_call(self,request,handler):
        tool_name=request.tool_call['name']
        start=time.time()
        result=handler(request)
        elapsed=time.time()-start

        self.logger.log(
            "tool_call",
            agent=self.agent_name,
            tool=tool_name,
            elapsed_ms=int(elapsed*1000),
        )
        return result


# ==================================================
# 3. 工具定义
# ==================================================
@tool
def web_search(query:str) -> str:
    """搜索信息。"""
    time.sleep(0.3)
    return f"关于'{query}'的搜索结果：AI芯片市场约5000亿美元。"

@tool
def calculate(expression:str) -> str:
    """计算表达式。"""
    time.sleep(0.1)
    try:
        result=eval(expression,{"__builtins__":{}},{})
        return f"{expression} = {result}"
    except:
        return "计算失败"


# ==================================================
# 4. 构建多 Agent 系统
# ==================================================
class TraceState(TypedDict):
    messages:list
    next_agent:str
    iteration:int

def build_traced_system(logger:StructuredLogger):

    research_agent=create_agent(
        model=model,
        tools=[web_search],
        system_prompt="你是研究专家，使用搜索工具查询信息。",
        middleware=[TracingMiddleware("research_expert",logger)],
    )

    math_agent=create_agent(
        model=model,
        tools=[calculate],
        system_prompt="你是数学专家，使用计算工具进行计算。",
        middleware=[TracingMiddleware("math_expert",logger)]
    )

    @tool("research_expert",description="研究专家，用于查询信息。")
    def call_research(query:str) -> str:
        logger.log("worker_invoked",worker="research_expert",query=query[:50])
        result=research_agent.invoke({"messages":[{"role":"user","content":query}]})
        return result["messages"][-1].content


    @tool("math_expert",description="数学专家，用于计算。")
    def call_math(query:str) -> str:
        logger.log("worker_invoked",worker="math_expert",query=query[:50])
        result=math_agent.invoke({"messages":[{"role":"user","content":query}]})
        return result["messages"][-1].content

    supervisor=create_agent(
        model=model,
        tools=[call_research,call_math],
        system_prompt="需要查询时调用 research_expert，需要计算时调用 math_expert。",
        middleware=[TracingMiddleware("supervisor",logger)],
    )

    return supervisor


# ==================================================
# 5. 测试三种 stream 模式
# ==================================================
def test_stream_updates(supervisor,logger):
    print("\n" + "=" * 60)
    print("🔍 stream_mode='updates'（追踪状态变化）")
    print("=" * 60)    

    for chunk in supervisor.stream(
        {"messages":[HumanMessage(content="查AI芯片市场规模，然后算15%增长3年后多少")]},
        stream_mode="updates",
    ):
        for node_name,update in chunk.items():
            logger.log("node_executed",node=node_name)
            print(f"  📦 节点: {node_name}")
            if "messages" in update:
                for msg in update["messages"]:
                    msg_type=type(msg).__name__
                    print(f"     {msg_type}")

def test_stream_values(supervisor,logger):
    print("\n" + "=" * 60)
    print("🔍 stream_mode='values'（追踪完整状态）")
    print("=" * 60)

    for state in supervisor.stream(
        {"messages":[HumanMessage(content="计算 25 * 48")]},
        stream_mode="values",
    ):
        msg_count=len(state.get("messages",[]))
        print(f"  状态快照: 消息数={msg_count}")

def test_stream_debug(supervisor,logger):
    print("\n" + "=" * 60)
    print("🔍 stream_mode='debug'（最详细的事件流）")
    print("=" * 60)    

    for event in supervisor.stream(
        {"messages":[HumanMessage(content="你好")]},
        stream_mode="debug",
    ):
        event_type=event.get("type","unknown")
        print(f"  事件类型: {event_type}")
        if event_type=="task":
            payload=event.get("payload",{})
            print(f"      节点: {payload.get('name','N/A')}")


# ==================================================
# 6. 主函数
# ==================================================
def main():
    trace_id=f"trace-{uuid.uuid4().hex[:8]}"
    logger=StructuredLogger(trace_id)

    print("=" * 60)
    print(f"🧪 多 Agent 日志追踪 (trace_id={trace_id})")
    print("=" * 60)

    supervisor=build_traced_system(logger)

    test_stream_updates(supervisor,logger)

    test_stream_values(supervisor,logger)

    logger.summary()

    print(f"\n{'=' * 60}")
    print("📈 性能分析")
    print("=" * 60)

    model_calls=[l for l in logger.logs if l["event"]=="model_call"]
    tool_calls=[l for l in logger.logs if l["event"]=="tool_call"]

    print(f"模型调用次数: {len(model_calls)}")
    print(f"工具调用次数: {len(tool_calls)}")

    if model_calls:
        total_model_ms=sum(l["elapsed_ms"] for l in model_calls)
        print(f"模型调用总耗时: {total_model_ms}ms")

    if tool_calls:
        total_tool_ms=sum(l["elapsed_ms"] for l in tool_calls)
        print(f"工具调用总耗时: {total_tool_ms}ms")

    print(f"总耗时：{logger.logs[-1]['elapsed_ms']}ms")

if __name__ == "__main__":
    main()
