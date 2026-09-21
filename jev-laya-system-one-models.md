# Jev 与 Laya：System One 决策模型总结

> 整理日期：2026-09-21

---

## 一、背景：什么是 System One Model（系统一模型）

传统大模型（GPT、Claude、DeepSeek）的目标一直是「跟人聊天」和「帮人干活」，本质是**逐字生成文本**。

2026 年 9 月，前 OpenAI 研究员 **Diogo Almeida**（ChatGPT 共同发明人之一）创办的 **TypeSafe AI** 发布了 **Jev**——一个「不说人话」的模型。灵感来自 Kahneman《思考，快与慢》：

| | 系统一（System 1） | 系统二（System 2） |
|---|---|---|
| 人类类比 | 直觉判断（一眼看出别人在生气） | 深度推理（算 17 × 24） |
| 模型对应 | **Jev / Laya**：直接输出结构化决策 | 传统 LLM：生成文字 |
| 输出 | JSON 概率分布 | 自然语言文本 |

核心思想：**所有需要「判断」而不需要「写作」的场景，都不该用生成式 LLM**。

---

## 二、Jev 详解

### 2.1 三种「AI 原语」（问题类型）

调用方式：传入 `state`（上下文）+ `questions`（问题），返回结构化 JSON。

#### ① Noul — 是/否判断
返回 0~1 的概率。

```json
// 请求
{
  "state": "我连续 3 天无法连接 Stripe 账户，正在丢失销售额，请尽快处理！",
  "questions": {
    "is_urgent": {
      "type": "noul",
      "instructions": "这条消息是否传达了紧急性或时效性"
    }
  }
}
// 响应
{ "is_urgent": { "type": "noul", "noul": 0.95 } }
```
代码里直接 `if noul > 0.8` 触发告警。

#### ② Choice — 多选一（最多 255 个选项）
返回选中项 + 每个选项的概率分布。

```json
{
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "这个工单应该分配给哪个团队处理",
      "criteria": {
        "billing":   "付款、发票、退款相关问题",
        "technical": "Bug、系统故障、集成问题",
        "sales":     "定价、账户咨询"
      }
    }
  }
}
// 响应：choice=billing, confidence=0.8, probabilities={billing:0.87, sales:0, technical:0.13}
```

#### ③ Score — 自定义量表打分（2~10 个等级）
分数可落在两个等级之间（如 1.04 = 介于 1 级和 2 级之间，偏向 1 级）。

```json
{
  "questions": {
    "frustration": {
      "type": "score",
      "instructions": "客户看起来有多沮丧",
      "criteria": ["冷静，只是在陈述事实", "不满但还算礼貌", "非常愤怒，言辞激烈"]
    }
  }
}
// 响应：score=1.04, confidence=0.94, probabilities={"0":0, "1":0.96, "2":0.04}
```

**三种问题可以在一次请求里同时问**，问 1 个和问 10 个响应时间几乎无差别。

### 2.2 为什么快——技术原理

| 维度 | 传统 LLM | Jev |
|---|---|---|
| 输出方式 | 逐 token 生成文字，写完再校验格式 | **并行采样**，直接在预定义选项上输出概率分布 |
| 训练目标 | RLHF（回答读起来通顺）/ RLVR（可验证任务做对） | **RLCD 校准决策强化学习**（说到做到：说 90% 把握，实际就约 90% 对） |
| 响应速度 | 3 - 329 秒 | **70 - 500 毫秒**（快近 200 倍） |
| 输入价格 | $10 / 百万 token | **$0.042 / 百万 token** |
| 输出价格 | 约输入价 5 倍 | **免费**（无文字输出） |
| 结构化错误率 | 0.58% - 45.5% | **0%（数学保证）** |

关键点：**概率是校准的（calibrated）**，程序可以直接拿概率做自动化决策（如 >0.8 自动处理、<0.5 转人工）——这是它区别于「提示词里要求输出 JSON」的根本原因。

### 2.3 使用方式

1. **官方 Playground**：网页上填 state + questions，直接看结果
2. **官方 API / SDK**（Python / JavaScript / HTTP）：

```python
# pip install typesafe-sdk
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient()  # 自动读取环境变量 TYPESAFE_API_KEY
response = client.system_one(
    state="客户说：我被重复扣款了，订单号 A-104，请退款。",
    questions={
        "department": Choice(instructions="...", criteria={...}),
        "is_urgent": Noul(instructions="..."),
    },
)
print(response.answers["department"].choice)  # "billing"
print(response.answers["is_urgent"].noul)     # 0.95
```

3. **第三方平台**：Vercel AI Gateway（新增 `experimental_evaluate` 接口）、Cloudflare Workers AI
4. **AI 编程工具接入**：安装 TypeSafe Skills 技能包后，在 Claude Code / Codex 等工具里用 `/typesafe-ai` 命令或直接说「用 Jev 帮我判断」

### 2.4 典型应用场景

| 场景 | 说明 |
|---|---|
| 工单/消息分类 | 一次请求同时判断：分给哪个部门 + 是否紧急 + 情绪如何（1018 篇论文分类仅 $0.08） |
| **AI Agent 决策中间层** | Agent 每步的「调哪个工具、操作是否有风险」交给 Jev，需要写代码才上 LLM（浏览器自动化 7.1 秒搜完航班，$0.0039） |
| 内容审核/风控 | 违规判断 + 类型 + 等级一次搞定，每条 < $0.001 |
| LLM 输出质检 | 用便宜模型检查贵模型：跑题检测、质量打分 |
| 实时游戏决策 | 地铁跑酷超人类操作、超级马里奥、星际争霸第一关——传统 LLM 延迟根本做不到 |

### 2.5 实测结论（鱼皮测评）

- **数字华容道对决 DeepSeek V4.1 Flash**：Jev 完成速度遥遥领先，token 和价格更低
- **1000 封邮件批量分类**：Jev 15.6 秒（64 封/秒，$0.0177）vs DeepSeek 48.9 秒（20 封/秒，$0.0207）
- **连连看游戏**：能通关但慢（5 分钟），瓶颈不在 Jev 的决策，而在「看」——Jev 只能吃文本/JSON，Canvas 渲染的游戏拿不到 DOM 数据就废了。社区玩游戏的通用思路是**先用代码把游戏状态转成结构化数据，再喂给 Jev**
- **697 篇长文打标签**：几分钟完成，批量分类场景极佳
- 总花费不到 2 块钱

### 2.6 能力边界（重要）

- ❌ 不能生成文字
- ❌ 不能做算术推理
- ❌ 日期比较不靠谱
- ⚠️ **判断准不准取决于喂给它的结构化输入质量**（garbage in, garbage out）

---

## 三、Laya 详解（开源替代）

> 本节数据来自官方仓库 README 原文：https://github.com/NandhaKishorM/laya

### 3.1 是什么

**Laya** 是 Convai Innovations 开发的**多语言、非自回归 System 1 决策引擎**（Apache 2.0 开源），对任意 state（文本、邮件、工单、JSON）回答类型化问题——**单次前向传播，T4 上 33 ms**。因为没有文本生成，所以「没有东西需要解析，也没有东西可以幻觉」。

训练方式：RLCD 强化学习，以严格真值评分规则（strictly proper scoring rules）为奖励 + GRPO 风格策略梯度。

### 3.2 三个模型检查点 + Router

| 检查点 | Encoder | 参数量 | 上下文 | 用途 |
|---|---|---|---|---|
| `laya` | ModernBERT-large | 421M | 512 | 英文 |
| `laya-multilingual` | mmBERT-base | 322M | 1024 | 100+ 语言，快 2 倍 |
| `laya-typed-decisions` | ModernBERT-large | 421M | 1024 | typed-decisions 工作流 |

内置 **Router**（纯 Python 脚本/语言检测，<0.5 ms）按请求自动选检查点（如检测到天城文 → multilingual）。预加载可避免切换语言时 7.4 s (CPU) / 10.3 s (T4) 的重载；支持 `max_loaded=2`（LRU）、`unload()`、`attach()`（复用已有 agent 的显存）。

### 3.3 三种决策原语（与 Jev 一致）

- `choice` — 顶层标签 + 每选项概率 + 置信度（路由、意图、主题分类）
- `score` — 有序量表上的期望等级 + 分布（紧急度、沮丧度）
- `noul` — 校准的 P(true)，0.0–1.0（垃圾邮件、钓鱼、越狱、流失检测）

### 3.4 安装与使用

```bash
pip install laya
```

**Router 模式（推荐）：**

```python
from laya import Router
router = Router(preload=True)
res_en = router.predict(state, questions)            # 英文 -> laya (39.5 ms)
res_hi = router.predict({"body": "..."}, questions)  # 印地语 -> multilingual (32.8 ms)
res_td = router.predict(state, questions, model="typed-decisions")
```

**直接 SDK：**

```python
agent = laya.load("convaiinnovations/laya")
result = agent.predict(state, questions)  # 一次前向传播，~35 ms GPU
```

内置问题预设：`laya.router_questions()`、`guard_questions()`、`moderation_questions()`、`triage_questions()`。官方示例带**置信度门控**：低于 0.85 转人工升级。

### 3.5 性能（Tesla T4 实测）

| 场景 | laya | multilingual |
|---|---|---|
| 单问题 | 39.5 ms | **32.8 ms** |
| 10 条批量 | 158.6 ms（15.9/条） | **72.3 ms（7.2/条）** |
| 50 条批量 | 771 ms（15.4/条） | **337 ms（6.8/条）** |

吞吐：单张 T4 **103–332 问/秒**。

### 3.6 基准成绩

- MASSIVE 意图分类：英文 **0.783**（laya）vs 0.657；其他 13 语言 0.306 vs **0.451**
- XNLI：英文 **0.860** vs 0.843；其他 14 语言 0.521 vs **0.731**
- 可用语言数（>3 倍随机水平）：23/51 vs **45/51**
- ⚠️ 英文检查点在非英文语种上崩得很惨——高棉语「0.000 准确率 @ 0.952 置信度」
- typed-decisions 基准（400 案例，2000 决策）：`laya-typed-decisions` **0.766** 准确率、Brier **0.062**（base laya 0.362、multilingual 0.342、随机猜测 0.318、teacher 上限 0.735）。分原语：noul 0.857 / choice 0.733 / score 0.723

### 3.7 校准（重要：开箱即用是过自信的）

两个基础检查点**出厂都过自信**。需按（问题类型 × 选项数）做温度重拟合，之后平均 ECE **0.466 → 0.081**（laya）、0.314 → 0.106（multilingual，出厂无拟合温度）。

### 3.8 训练 / 微调

官方提供 Kaggle 2xT4 notebook 覆盖完整闭环（数据集 → 训练 → 温度拟合 → 评估 → 推送 HF），约 30k 问题 × 4 epoch 耗时 4–5 小时。README 明确强调：基础模型是「**快速特化的底座，不是零样本决策引擎**」——正式用于垂直场景前必须微调。

### 3.9 与 Jev 的正面对比（README 官方数据）

- **延迟**：单问 32.8 ms vs Jev 公布的 236–276 ms p50 → **约 6–7 倍快**（另有 7.8 倍的说法）
- **typed-decisions 硬准确率**：Laya 0.766 vs Jev 0.727（Laya 胜）；但 Jev 软准确率更高（0.580 vs 0.471）、原始 ECE 更好（0.144 vs 0.213）
- AG News：0.950 vs 0.910；DAIR Emotion：0.595 vs 0.480（Laya 均胜）
- **高基数标签 Jev 明显领先**：Banking77（77 类）——Jev 0.870 vs Laya 0.425。根因是 Laya 固定的 `head_max_len` token 预算（英文 192 / 多语言 256），77 个选项平均每个只分到 3–4 个 token；把 `head_max_len` 提到 512+ 或做层级拆分可缓解
- **成本**：Laya Apache 2.0 自托管 $0；Jev 闭源 API（$0.042 / 1M tokens）

### 3.10 生态资源

| 资源 | 地址 |
|---|---|
| **官方仓库（PyPI：`pip install laya`）** | github.com/NandhaKishorM/laya |
| HuggingFace 模型 | huggingface.co/convaiinnovations/laya |
| 官网 | laya.convaiinnovations.com |
| Node.js/TS 运行库（ONNX Runtime） | github.com/receptron/laya |

---

## 四、Jev vs Laya 对比总表

| 维度 | Jev（TypeSafe AI） | Laya（开源） |
|---|---|---|
| 发布方 | TypeSafe AI（Diogo Almeida，$40M 种子轮） | Convai Innovations，Apache 2.0 |
| 开源 | ❌ 闭源商业 API | ✅ 开源自托管 |
| 架构 | 并行采样（细节未公开） | 非自回归、单次前向传播（ModernBERT/mmBERT + Router） |
| 延迟 | 官方宣传 70–500 ms（Laya README 引用 p50 为 236–276 ms） | 32.8 ms（T4 实测），约 6–7 倍快 |
| 成本 | 输入 $0.042/M token，输出免费 | $0（本地推理，322M–421M 参数） |
| 硬准确率（typed-decisions） | 0.727 | **0.766** |
| 软准确率 / 原始校准 | **0.580 / ECE 0.144** | 0.471 / ECE 0.213 |
| 高基数分类（Banking77） | **0.870** | 0.425（`head_max_len` 限制所致，可调参缓解） |
| 多语言 | 未见公开数据 | 100+ 语言（multilingual 检查点，45/51 语种可用） |
| 出厂状态 | 概率已校准（RLCD 保证） | **过自信**，需温度重拟合后才可信 |
| 适用 | 快速接入、生产环境、高基数复杂分类 | 数据敏感、零成本、边缘部署、可微调特化 |
| 接入 | Playground / API / SDK / Vercel / Cloudflare / Skills | `pip install laya` / HF / Node.js ONNX |

---

## 五、对我的启示（Agent 开发视角）

1. **「智能的 if 语句」心智模型**：Jev/Laya 不是聊天模型的替代品，而是**判断层的专用件**。生成文字 → LLM；快速判断 → System One 模型，两者互补。
2. **Agent 架构升级点**：目前 Agent（包括我在做的生图/对话 Agent、图生视频 Agent）每一步都要调一次大模型做路由/工具选择判断。把这类判断换成 Jev/Laya，延迟从秒级降到毫秒级，成本降 1-2 个数量级——这是 Agent 决策层优化的直接抓手。
3. **接口设计范式**：`state + questions(类型化原语)` 这个 API 设计很干净，自己写 Spring AI / LangChain 应用时可以借鉴这种「声明式判断」的封装思路。
4. **关键前提**：喂给它的 state 必须是准确的结构化数据。玩游戏的案例证明了瓶颈永远在「感知」（把现实世界转成 JSON），不在「决策」。
5. **动手路线**：先在 TypeSafe Playground 免费体验 → Python/JS SDK 跑通官方 Quick Start → `pip install laya` 本地跑通（Apache 2.0 零成本；注意 README 明说基础模型是「特化底座，不是零样本引擎」，且出厂过自信需温度校准）→ 在自己的 Agent 项目里把「工具路由」这一步替换成决策模型试试。

---

## 六、参考链接

- TypeSafe 官方公告：https://typesafe.ai
- Jev 中文资料合集：https://github.com/yzfly/awesome-jev-zh
- Jev 深度讨论：https://github.com/mizzlelover/jev-hub
- **Laya 官方仓库（本文 Laya 部分数据来源）**：https://github.com/NandhaKishorM/laya
- Laya Node.js 运行库：https://github.com/receptron/laya
- Laya HuggingFace 模型：https://huggingface.co/convaiinnovations/laya
- Laya 官网：https://laya.convaiinnovations.com
- Laya PyPI 决策引擎：https://github.com/NandhaKishorM/laya
- TechCrunch 报道："A new kind of AI model from a ChatGPT inventor is thrilling readers"
