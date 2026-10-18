# W7

## 本周学习内容

### Day 1:

* 多Agent架构：Supervisor/Worker模式
* 跑通 Supervisor 分派 2 个 Worker
* 分析消息传递流程

### Day 2:

* Agent 间通信协议
* 任务拆分与结果合并
* 做“信息收集”多 Agent

### Day 3:

* 浅层回退机制（Worker 失败 → Supervisor 重分派）
* 避免死循环（最大迭代次数）
* 给多 Agent 加安全阀

### Day 4:

* &#x20;多 Agent 日志追踪
* &#x20;性能考量



项目结构
├── day1/
├── day2/
├── day3/
├── day4/
├── requirements.txt
├── .gitignore
├── README.md
运行环境

* Python 3.10+
* LangChain >= 1.4.0
* 其余详见requirements.txt

