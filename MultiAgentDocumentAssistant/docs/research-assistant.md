# 多 Agent 科研论文助手

新增 `research/` 是独立可选入口。原 `agents/`、`models/`、`retrieval/`、
`mydatasets/`、Hydra 配置和 benchmark 脚本保持不变，旧命令继续使用原实现。
解析复用原项目的 PyMuPDF 技术栈；新增抽取函数保留行分隔和页码，避免旧数据集
加载器去除换行、截断页面后丢失章节信息。旧 ColBERT / ColPali 索引不自动迁移。

## 安装和运行

建议使用单独 Python 3.11 / 3.12 环境，避免与原实验环境的固定 Torch 版本冲突：

```powershell
python -m venv .venv-research
.\.venv-research\Scripts\Activate.ps1
python -m pip install -r requirements-research.txt
docker compose -f compose.research.yaml up -d
```

将 `.env.research.example` 中的值配置为环境变量。程序**不会自动读取 .env 文件**。
PowerShell 示例：

```powershell
$env:POSTGRES_URI = 'postgresql://research:research@localhost:5432/research'
$env:NEO4J_PASSWORD = 'research-password'
$env:QWEN_API_KEY = '<your-key>'
$env:QWEN_MODEL = '<model-id-available-in-your-account>'
$env:QWEN_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
$env:DEEPSEEK_API_KEY = '<your-key>'
$env:DEEPSEEK_MODEL = '<model-id-available-in-your-account>'
$env:DEEPSEEK_BASE_URL = 'https://api.deepseek.com'
$env:CLAUDE_API_KEY = '<your-key>'
$env:CLAUDE_MODEL = '<model-id-available-in-your-account>'
```

Compose 是绑定本机端口的开发配置；部署时替换示例密码。卷保留三个数据库的内容。
首次执行会初始化表、约束、集合及 checkpoint schema，并下载 embedding/reranker 权重。
当前统一 runtime 启动所有存储和检索模型；即使使用引用快速通道，也需要运行依赖服务。
首次构图会检查 DeepSeek 配置；实际推理按任务路由选择模型，不自动替换缺失的供应商。

```powershell
python -m research --namespace lab-a ingest paper1.pdf paper2.pdf
python -m research --namespace lab-a run --thread project-001 '比较论文的方法、实验局限并提出可验证的新想法'
python -m research --namespace lab-a run --thread question-001 --task qa '论文采用了哪些基线？'
python -m research --namespace lab-a run --thread ideas-001 --task ideas '提出可验证的后续实验'
python -m research --namespace lab-a run --thread review-001 --pause '分析论文实验结论'
python -m research --namespace lab-a status --thread review-001
python -m research --namespace lab-a resume --thread review-001 --decision accept
```

`--decision reject` 丢弃暂停的结果，不写入本次长期记忆。进程异常或服务故障修复后，
使用 `resume --thread ...`（不带 decision）从最后的 checkpoint 继续。已完成线程不能
重新提交；新问题使用新的 `--thread`。同一线程避免并发运行多个进程。
线程 ID 对 namespace 与外部 thread 名做联合散列；所有检索和记忆查询都限定 namespace。
这是数据分区机制，不是网络服务身份认证；本项目提供的是本地 CLI。

引用格式化可直接调用，也可用 `--task citation --citation-json '<JSON>'`：

```python
from research.skills import format_citation
print(format_citation('Paper title', ['A. Author'], '2024', style='bibtex'))
```

格式化只处理用户提供的元数据，支持简化 APA 和 BibTeX，不等于文献真实性核验。

## 工作流

```mermaid
flowchart TD
    A[问题与任务类型] --> B[三层记忆召回 / RAG]
    B --> C{research?}
    C -->|是| S[LangGraph Supervisor]
    S --> L[文献 Agent]
    S --> N[分析 Agent]
    S --> I[Idea Agent]
    S --> R[审查 Agent]
    L --> S
    N --> S
    I --> S
    R --> S
    S --> G[检查四角色输出完整性]
    C -->|qa / ideas| F[专职 Sub-Agent 快速通道]
    G --> D[合成回答]
    F --> D
    D --> V[Self-Reflection 引用校验]
    V -->|失败，最多两次| E[Claude 修订]
    E --> V
    V -->|通过或达到重试上限| H[可选持久化中断]
    C -->|citation| T[确定性格式化]
    T --> H
    H --> M[写入记忆 / 保留策略]
```

四个专职 Agent 是具名 LangGraph 子图。Supervisor 使用官方 `create_supervisor`，
通过 handoff 调度；若模型提前结束，完整性节点补齐未执行的角色。已完成的角色不会
再次调用模型。快速通道由明确的 `--task` 控制，避免根据短问题盲目判定为简单任务。
快速通道跳过 Supervisor 调度，但 `qa` / `ideas` 仍进行合成和引用审查。

外层图用 `AsyncPostgresSaver` 保存消息、证据、报告、审查结果及重试次数，子图继承
checkpointer。`interrupt()` 与 `Command(resume=...)` 提供人工中断恢复。
暂停发生在持久化长期记忆之前。节点失败后可能重放该节点内的调用；模型请求不是
exactly-once。长期记忆采用确定性 ID upsert，避免 checkpoint 重放产生重复记录。
三个记忆数据库之间没有分布式事务，跨库写入部分失败时需恢复该线程完成重试。

## 长期记忆

| 层 | 存储 | 内容与召回 |
|---|---|---|
| Episodic | Neo4j | 问题、结果、来源；Episode → CITES → Source；保留分数结合词项匹配 |
| Semantic | Qdrant | 仅保存通过引用审查的结论；向量相似检索 |
| Procedural | PostgreSQL JSONB | 三个版本随代码维护的 SKILL 流程；按词项和保留分数召回 |

Importance 为置信度、新颖度、实用性按 0.4/0.2/0.4 加权，并限制在 [0,1]。
当前工作流使用规则先验值，未声称实现经过训练的 importance/novelty 评估器。
保留分数 = `min(1, importance + 0.03*log1p(accesses)) * 2^(-days/30)`。
召回更新访问时间与计数。每次记录结果后自动维护，也可独立运行：

```powershell
python -m research --namespace lab-a maintain
```

压缩为单条记忆的三级抽取式压缩：L0 原文，分数 <0.4 为 L1（1600 字符），
<0.15 为 L2（400 字符）。优先保留完整句，长句必要时截断，来源 ID 始终保留。
分数 <0.03 且 importance <0.8 才删除；作者维护的高重要性 Procedural SKILL 固定保留。
当前不生成跨事件聚类摘要；压缩内容不再恢复成原文，论文原文仍在独立 RAG 索引中。
维护和非语义召回当前会扫描该 namespace 记录，适合原型/中小规模，海量数据需分页批处理优化。

## RAG 与引用校验

- 章节感知切分：识别常见中英标题、数字编号、Markdown 标题；不跨页和章节。
  默认 1400 字符，重叠 180，保留文档哈希、章节、页码、稳定 chunk ID。
- 多表示索引：Qdrant 同一 point 保存 `body` 与 `context` 两个 dense 向量，
  context 包含标题、章节与正文前缀；`lexical` 保存 BM25 sparse 向量。
- Hybrid Search：三个分支各召回 30，RRF 融合后送交 `BAAI/bge-reranker-v2-m3`
  CrossEncoder 重排，默认选取 6 条。这里的多向量是多表示 named vectors，
  不是 ColBERT token-level late interaction。
- Self-Reflection：回答使用 `[24位chunk_id]` 引用；先检查引用 ID 存在性，再由
  Claude 检查所有事实（含无引用事实），返回 claim/source/quote/supported。
  代码校验 quote 是原文中的非空子串；两次修订仍失败则输出证据不足，不将草稿写入语义记忆。
  模型审查仍可能漏判，不能视作形式化事实正确性保证。

默认 dense encoder 为面向英文论文的 `BAAI/bge-small-en-v1.5`，支持通过
`RESEARCH_EMBED_MODEL` 指定 FastEmbed 支持的模型；更换模型必须使用新集合/迁移重建索引，
不能混用不同模型的向量。BM25 默认分词与 dense 默认模型未针对中文专门评测。
章节识别是文本启发式，未实现复杂双栏版面恢复或 OCR；扫描 PDF 会明确报错。
相同文件重复导入幂等，修改后的文件视作新文档，不自动删除旧版本。

## SKILL 与模型路由

`skills/pdf-parse`、`skills/citation-format`、`skills/idea-extract` 各提供 SKILL.md；
`research.skills.SKILLS` 注册对应可执行函数。解析由 ingest 调用，格式化由 fast 节点
调用，Idea 抽取由 ideator 调用。流程也写入 Procedural 记忆供后续任务检索。
更新 SKILL 后，已有 namespace 中的固定记忆不会自动覆盖；需显式迁移旧流程记录。

| 条件 | 模型 |
|---|---|
| 简短文献整理 | Qwen |
| Supervisor / 分析 / Idea，或输入 >6000 字符 | DeepSeek |
| 审查 / 修订，或输入 >24000 字符 | Claude |

按实际传入提示长度动态选择；客户端设置超时及有界请求重试。供应商失败不会静默
降级到其他账户；修复配置后依靠 checkpoint 恢复。模型名由账户配置，不绑定可能过期的型号。

## 验证与边界

```powershell
python -m pytest tests -q
```

离线测试使用实际 LangGraph Supervisor 与 Qdrant 本地引擎，模型与 embeddings 替身，
覆盖四角色 handoff、快路径、重建恢复、引用失败上限、隔离、幂等和记忆衰减压缩。
PostgreSQL / Neo4j 测试需单独配置测试服务：

```powershell
$env:RESEARCH_TEST_POSTGRES_URI = $env:POSTGRES_URI
$env:RESEARCH_TEST_NEO4J_URI = 'bolt://localhost:7687'
python -m pytest tests/test_services.py -q
```

服务测试使用随机 namespace；应指向可用于测试的数据库。
本次本机没有可用的 PostgreSQL / Neo4j 服务、供应商密钥，也未下载大型 embedding /
reranker 权重，因此真实三库、供应商和 BGE 权重端到端测试未执行。
没有声称达到特定检索质量、延迟或成本收益；需用项目论文数据建立评测集后测量。
官方 Supervisor 当前内部仍依赖被标记弃用的 create_react_agent，依赖限制 LangGraph <2。

接口参考：[LangGraph 持久化](https://docs.langchain.com/oss/python/langgraph/persistence)、
[Supervisor](https://github.com/langchain-ai/langgraph-supervisor-py)、
[Qdrant Hybrid Queries](https://qdrant.tech/documentation/search/hybrid-queries/)、
[BGE 模型](https://huggingface.co/BAAI/bge-reranker-v2-m3)。
