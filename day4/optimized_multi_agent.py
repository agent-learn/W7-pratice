"""
优化后的多 Agent 系统
对比：优化前 vs 优化后
"""
import os
import time
from typing import TypedDict, Annotated
import operator
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
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
# 1. 全局缓存
# ==================================================
_search_cache={}

@tool
def cached_search(query:str) -> str:
    """带缓存的搜索工具。"""
    if query in _search_cache:
        print(f"   💾 [缓存命中] {query}")
        return _search_cache[query]

    time.sleep(0.5)
    result=f"'{query}'的搜索结果"
    _search_cache[query]=result
    print(f"   🌐 [网络请求] {query}")
    return result


# ==================================================
# 2. 优化前：串行多 Agent
# ==================================================
class SlowState(TypedDict):
    questions:list[str]
    results:Annotated[list[str],operator.add]

def slow_supervisor(state:SlowState) -> dict:
    results=[]
    for q in state["questions"]:
        result=cached_search.invoke({"query":q})
        results.append(result)
    return {"results":results}

def build_slow_system():
    builder=StateGraph(SlowState)
    builder.add_node("supervisor",slow_supervisor)
    builder.add_edge(START,"supervisor")
    builder.add_edge("supervisor",END)
    return builder.compile()


# ==================================================
# 3. 优化后：并行多 Agent + 缓存
# ==================================================
class FastState(TypedDict):
    questions:list[str]
    results:Annotated[list[str],operator.add]

def fast_dispatch(state:FastState) -> list[Send]:

    return [Send("worker",{"question":q}) for q in state["questions"]]

def fast_worker(state:dict) -> dict:

    question=state["question"]
    result=cached_search.invoke({"query":question})
    return {"results":[result]}

def build_fast_system():
    builder=StateGraph(FastState)
    builder.add_node("worker",fast_worker)
    builder.add_conditional_edges(START,fast_dispatch,["worker"])
    builder.add_edge("worker",END)
    return builder.compile()


# ==================================================
# 4. 对比测试
# ==================================================
def main():
    print("=" * 60)
    print("🧪 多 Agent 系统优化对比")
    print("=" * 60)

    # 有重复的问题（模拟真实场景：用户可能重复询问）
    questions = [
        "AI芯片市场规模",
        "Python最新版本",
        "AI芯片市场规模",      # 重复
        "LangGraph教程",
        "Python最新版本",      # 重复
    ]

    print("\n--- 优化前（串行，无缓存）---")
    _search_cache.clear()
    slow_system=build_slow_system()
    start=time.time()
    slow_result=slow_system.invoke({"questions":questions,"results":[]})
    slow_time=time.time()-start
    print(f"耗时：{slow_time:.2f}s")

    print("\n--- 优化后（并行 + 缓存）---")
    _search_cache.clear()
    fast_system=build_fast_system()
    start=time.time()
    fast_result=fast_system.invoke({"questions":questions,"results":[]})
    fast_time=time.time()-start
    print(f"耗时：{fast_time:.2f}s")

    print(f"\n{'=' * 60}")
    print("📊 优化效果")
    print("=" * 60)
    print(f"优化前：{slow_time:.2f}s")
    print(f"优化后：{fast_time:.2f}s")
    print(f"加速比：{slow_time/fast_time:.1f}x")
    print(f"结果一致性：{len(slow_result['results'])==len(fast_result['results'])}")

if __name__ == "__main__":
    main()
