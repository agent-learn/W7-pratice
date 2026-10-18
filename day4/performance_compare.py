"""
多 Agent 性能优化对比
演示：串行 vs 并行、无缓存 vs 有缓存
"""
import os
import time
import asyncio
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
# 1. 工具（带延迟模拟）
# ==================================================
@tool
def slow_search(query:str) -> str:
    """模拟慢速搜索（1秒延迟）。"""
    time.sleep(0.1)
    return f"'{query}'的搜索结果：模拟数据"


# ==================================================
# 2. 串行版本
# ==================================================
class SerialState(TypedDict):
    queries:list[str]
    results:Annotated[list[str],operator.add]

def serial_worker(state:dict) -> dict:
    """串行处理：一次处理一个 query"""
    query=state["query"]
    result=slow_search.invoke({"query":query})
    return {"results":[result]}

def build_serial_graph():
    """串行图：一个 query 一个节点"""
    builder=StateGraph(SerialState)

    def process_all(state:SerialState) -> dict:
        results=[]
        for query in state["queries"]:
            result= slow_search.invoke({"query":query})
            results.append(result)
        return {"results":results}

    builder.add_node("process",process_all)
    builder.add_edge(START,"process")
    builder.add_edge("process",END)

    return builder.compile()


# ==================================================
# 3. 并行版本（Send API）
# ==================================================
class ParallelState(TypedDict):
    queries:list
    results:Annotated[list[str],operator.add]

def dispatch_queries(state:ParallelState) -> list[Send]:
    return [Send("worker",{"query":q}) for q in state["queries"]]

def parallel_worker(state:dict) -> dict:
    query=state["query"]
    result=slow_search.invoke({"query":query})
    return {"results":[result]}

def build_parallel_graph():
    builder=StateGraph(ParallelState)
    builder.add_node("worker",parallel_worker)
    builder.add_conditional_edges("__start__",dispatch_queries,["worker"])
    builder.add_edge("worker",END)

    return builder.compile()


# ==================================================
# 4. 性能对比
# ==================================================
def compare_serial_vs_parallel():
    print("=" * 60)
    print("🧪 性能对比：串行 vs 并行")
    print("=" * 60)

    queries = ["查询1", "查询2", "查询3", "查询4", "查询5"]

    # 串行
    print("\n--- 串行版本 ---")
    serial_graph=build_serial_graph()
    start=time.time()
    serial_result=serial_graph.invoke({"queries":queries,"results":[]})
    serial_time=time.time()-start
    print(f"串行耗时：{serial_time:.2f}s")
    print(f"结果数：{len(serial_result['results'])}")

    # 并行
    print("\n--- 并行版本 ---")
    parallel_graph=build_parallel_graph()
    start=time.time()
    parallel_result=parallel_graph.invoke({"queries":queries,"results":[]})
    parallel_time=time.time()-start
    print(f"并行耗时：{parallel_time:.2f}s")
    print(f"结果数：{len(parallel_result['results'])}")

    # 对比
    print(f"\n{'=' * 60}")
    print("📊 对比结果")
    print("=" * 60)
    print(f"串行：{serial_time:.2f}s")
    print(f"并行：{parallel_time:.2f}s")
    print(f"加速比：{serial_time/parallel_time:.1f}x")

# ==================================================
# 5. 缓存对比
# ==================================================
class CachedSearch:

    def __init__(self):
        self.cache={}
        self.hits=0
        self.misses=0

    def search(self,query:str) -> str:
        if query in self.cache:
            self.hits+=1
            return self.cache[query]
        self.misses+=1
        result=slow_search.invoke({"query":query})
        self.cache[query]=result
        return result

def compare_no_cache_vs_cache():
    print("\n" + "=" * 60)
    print("🧪 性能对比：无缓存 vs 有缓存")
    print("=" * 60)

    # 有重复的查询
    queries = ["查询A", "查询B", "查询A", "查询C", "查询B", "查询A"]

    print("\n--- 无缓存 ---")
    start=time.time()
    for q in queries:
        slow_search.invoke({"query":q})
    no_cache_time=time.time()-start
    print(f"无缓存耗时：{no_cache_time:.2f}s")

    print("\n--- 有缓存 ---")
    cached=CachedSearch()
    start=time.time()
    for q in queries:
        cached.search(q)
    cache_time=time.time()-start
    print(f"有缓存耗时：{cache_time:.2f}s")
    print(f"缓存命中：{cached.hits}，未命中：{cached.misses}")

    print(f"\n加速比：{no_cache_time/cache_time:.1f}x")


# ==================================================
# 6. 主函数
# ==================================================
def main():
    compare_serial_vs_parallel()
    compare_no_cache_vs_cache()

if __name__ == "__main__":
    main()
