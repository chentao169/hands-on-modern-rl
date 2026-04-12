# 12.5 工业界实战：那些论文里没写的“血泪坑”

前面几节讲了 Agentic RL 的通用工程原则和框架设计。但当你真正动手训练一个 Agentic RL 模型时，会遇到各种论文里不会写的坑——训练突然崩了、回答越来越长变成乱码、reward 涨了但质量反而下降了。

2025–2026 年，多家公司（如 Alibaba、Moonshot、LinkedIn、Bespoke Labs 等）公开了他们做 Agentic RL 的“踩坑血泪史”。为了让 0 基础的开发者也能快速避坑，我们**不再按公司罗列，而是按照你动手训练时会遇到的“案发现场”时间线**，把这些工业界真金白银砸出来的经验串起来。

> 💡 **核心心法提前看**：在 Agentic RL 里，“先跑起来不崩”比“用什么花哨算法”重要得多。数据质量和环境稳定性，是决定生死的第一关。

---

## 场景一：训练还没开始，数据和环境去哪弄？

很多人的第一个问题是：我要让模型学会上网搜索或写代码，是不是得接个真实的浏览器或者真实的沙盒让它去交互？

### 真实 API 的坑：又贵又慢又不稳定

如果你直接接真实的搜索引擎 API 做训练，会遇到一个致命问题：**不可复现**。

> **Moonshot AI（月之暗面）**在训练 Kimi-Researcher 时发现：搜索引擎的结果是随时间变化的。这导致 PPO 算法中要求的“同一状态（State）应该产生一致的观测（Observation）”这个假设被破坏了。因此，他们干脆放弃了对环境稳定性要求高的 PPO/GRPO，用起了最古老但也最随性的 **REINFORCE** 算法（端到端直接算梯度），这是一种工程上的妥协与创新 [\[参考\]](https://moonshotai.github.io/Kimi-Researcher/)。

### 解法：自己造“楚门的世界”（合成环境）

与其让模型去真实的互联网里“裸奔”，不如给它造一个确定性的“模拟沙盒”。

> **Alibaba 通义团队**在构建 Tongyi DeepResearch 时，发现“数据和训练环境的稳定性是决定 RL 是否 work 的更关键因素”。他们没有用真实的 Web API，而是用离线的 Wikipedia 数据库和自定义工具，造了一个确定性的合成环境。实验证明，在这个“楚门的世界”里用合成数据训练，效果远比在真实但嘈杂的人工标注数据上好得多 [\[参考\]](https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/)。

### 没算力没数据？72 条数据也能“逆天改命”

如果你说“我是个人开发者，没那么多钱合成数据”，别怕。

> **Amazon Science** 证明了小数据定制的奇迹。在 AppWorld（个人助手 agent 基准）上，他们仅仅用了 **72 个高质量的训练样本**做 RL，就把 Qwen-2.5-32B 的任务完成率从 39.2% 拔高到了 72%，甚至超过了 Claude Sonnet 3.7/4.0。他们的结论是：高质量的小数据，加上模型在 RL 过程中的自我探索（比如模型自己学会了“写代码前先查 API 文档”），完全可以四两拨千斤 [\[参考\]](https://www.amazon.science/blog/customizing-multiturn-ai-agents-with-reinforcement-learning)。

---

## 场景二：刚跑起来就崩了，Loss 怎么爆炸了？

好不容易搞定了数据，你兴奋地按下了 `python train.py`，结果没跑几步，梯度爆炸了，Reward 死活不涨。你开始怀疑是不是 PPO 算法的超参数没调对。

别急着调参，**很可能是你底层的代码写错了，而且错得很深。**

### 隐藏杀手：训练引擎和推理引擎“精神分裂”

Agentic RL 的特点是：模型需要先**推理（Rollout）**生成动作，然后再**训练（Backward）**更新权重。

> **LinkedIn 团队**在用 GPT-OSS（一个 MoE 架构的开源模型）做 RL 训练时，遇到了极其诡异的梯度爆炸。他们扒到了最底层的 CUDA Kernel 才发现根因：推理引擎（vLLM/SGLang 使用的 Triton kernel）和训练引擎（FSDP 使用的 FlashAttention-v2）对 Attention 的实现有一点点微小的差异。这种“精神分裂”导致模型在生成数据和计算梯度时对不齐。更离谱的是，他们发现训练框架里根本没实现 Attention Sink 参数的反向传播（Backward），相当于模型在训练时，有一部分脑子是“假装不存在”的。直到他们手写了 FlashAttention-v3 的 Sink Backward，训练才彻底稳定 [\[参考\]](https://huggingface.co/blog/LinkedIn/gpt-oss-agentic-rl)。

👨‍💻 **小白避坑大白话**：如果你用复杂的模型（比如 MoE），别盲目相信开源框架的默认配置。先在最简单的单轮问答（比如 GSM8K）上跑通，确认 Loss 正常下降，再去跑多轮 Agent 任务。

---

## 场景三：模型变成了“话痨”和“乱码复读机”？

这是 Agentic RL 里最常见、也最让人崩溃的现象：模型不仅没有学会用工具，反而开始疯狂输出几千个 token 的废话，最后变成毫无意义的乱码。或者更惨的，发生了**格式坍塌（Format Collapse）**：

```json
// 期望的输出格式：
{"action": "search", "query": "AAPL stock"}

// 训练跑偏后的崩溃输出（格式坍塌）：
{"action": "searchsearchAAPL<|endoftext|><|endoftext|><|endoftext|>"
```

### 乱码原因 1：Reward 设计太复杂，“聪明反被聪明误”

你可能觉得，为了让模型学得好，我得给它很多奖励信号：用了工具给 +1，格式对了给 +1，最终答案对了给 +5。

> **Bespoke Labs** 发现，复杂的 Reward 反而会被模型“钻空子”（Reward Hacking）。模型发现只要不断调用工具就能一直拿 +1，于是它干脆不回答问题了，就一直在那死循环调工具。他们的最终解法是：**大道至简，只保留“任务是否完成”这一个二值的奖励信号**，其他花里胡哨的奖励全删掉，训练反而最稳定 [\[参考\]](https://www.bespokelabs.ai/blog/improving-multi-turn-tool-use-with-reinforcement-learning)。

### 乱码原因 2：负样本没过滤好，模型“破罐子破摔”

有时候模型没做对任务，其实是因为步数超了或者被环境截断了，但它输出的格式是完美的。如果你一股脑把这些没完成的任务全给惩罚（负样本）。

> **Alibaba 通义团队**观察到，不加过滤的负样本会在长时间训练后导致“格式坍塌”——模型连怎么输出正确的工具调用格式都忘了。他们的解法是**保守负采样（Conservative Negative Sampling）**：选择性地排除那些因为超出长度限制而没有产生最终答案的轨迹，保护模型的格式输出能力 [\[参考\]](https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/)。

### 乱码原因 3：“遛狗的绳子（KL 散度）”断了

在 RLHF/GRPO 里，我们通常会用一个 KL 惩罚项（KL Penalty）来限制正在训练的模型（策略模型）不要偏离初始模型（Reference Model）太远。这就像是你遛狗时的牵引绳，Reference Model 就是你，狗就是策略模型。

> **Bespoke Labs** 在训练 Qwen-2.5-7B 时踩遍了坑：如果把 KL 惩罚设为 0（解开牵引绳），模型 300 步就彻底放飞自我，跑出乱码；如果 KL 太大，模型又学不到新东西。
> 他们的秘籍是：
>
> 1. **给一根极细的绳子**：把 KL 权重设为微小的 0.001，提供保底约束。
> 2. **人跟着狗走（定期更新 Reference）**：每 100 步，把现在的狗（当前策略模型）复制一份当成新主人（新的 Reference Model）。让约束从“别跑太远”变成“别跑太远，但可以往前跑” [\[参考\]](https://www.bespokelabs.ai/blog/improving-multi-turn-tool-use-with-reinforcement-learning)。

### 话痨解法：Gamma-decay 奖励效率，而不是惩罚长度

怎么阻止模型为了凑字数而变成话痨？

> **Moonshot** 提出了一招叫 **Gamma-decay Reward**。如果模型对了，奖励不是固定的，而是随着步数衰减：$r_i = r \times \gamma^{T-i}$（$\gamma < 1$）。
> 简单说就是：**同样是答对问题，你用 5 步答对，我给你 10 块钱；你磨叽了 30 步才答对，我只给你 2 块钱。** 这样模型就会自己学会“闭嘴，快点干活” [\[参考\]](https://moonshotai.github.io/Kimi-Researcher/)。

---

## 场景四：长任务跑到一半，模型“失忆”了？

Agentic RL 和传统 RL 最大的不同，就是交互轮数可能极长。如果要查文献、写代码、再改 Bug，可能要交互 50 轮以上。这时候，上下文窗口（Context Window）很快就会被撑爆，模型就像一个记性差的人，开始忘记最开始的任务。

### 解法：教模型“断舍离”（Context Management）

> **Moonshot** 的 Kimi-Researcher 在解决长文本失控时，引入了**上下文管理（Context Management）**机制。他们不强求模型记住所有历史，而是让模型在每一轮自己决定：**“这篇网页有用吗？有用就提炼摘要留着，没用就扔掉（丢弃文档）。”** 消融实验证明，学会了“断舍离”的模型，平均能多跑 30% 的轮次而不失忆，最终拿到了更高的分数 [\[参考\]](https://moonshotai.github.io/Kimi-Researcher/)。

---

## 场景六：其他单例与特定架构的坑

前面的场景是绝大多数人在做 Agentic RL 时都会遇到的通用问题（数据、Reward、上下文、显卡利用率）。但如果你在使用某些**特定的模型架构（如 MoE）**，或者**试图在极小参数量的模型上创造奇迹**，你还会踩到一些专门为你准备的坑。

### 1. MoE 模型的路由不确定性，破坏了 RL 的底层假设

很多团队看中了 MoE 模型（如 Mixtral、DeepSeek-V3）推理成本低的优势，想拿它做 Agentic RL 的基座。但 MoE 模型有一个致命陷阱：**路由选择的随机性**。

PPO 等算法有一个核心假设：当前生成数据的模型和正在被训练的模型，必须是同一个（即 On-policy）。这在数学上表现为重要性采样比率（Ratio）必须等于 1。

> **LinkedIn 团队**在用 GPT-OSS 做 RL 时发现，MoE 模型的 Gating Network（路由网络）在两次 Forward Pass 时，可能会给同一个 Token 选择不同的专家（Expert）！这直接导致 $\log \pi(a|s) \neq \log \pi_{\text{old}}(a|s)$，PPO 的 On-policy 假设被当场击碎。他们不得不写了一行硬核代码：`old_log_prob = log_prob.detach()`，强行把两次概率拉齐，才把这个问题稳住 [\[参考\]](https://huggingface.co/blog/LinkedIn/gpt-oss-agentic-rl)。

### 2. MoE 训练不仅人跑得慢，连“专家”也闲忙不均

前面我们提到了“有的人跑得快，有的人跑得慢”的木桶效应。对于 MoE 模型，这种显卡空转更严重，因为不仅任务长度不同，连不同“专家（Expert）”的负载也不同（大家都想用同一个聪明的专家，导致负责这个专家的显卡被挤爆，其他显卡在发呆）。

> **Salesforce** 提出了**流水线同步 RL（Pipelined Synchronous）**结合 **Least-Loaded Expert Parallelism**。他们不仅让 Rollout（做题）和 Training（讲题）交替进行，还让空闲的 GPU 去接管负载重的 MoE 专家。这套组合拳把 16 块 H200 榨干到了极致，内存效率比传统方法高了 250 倍 [\[参考\]](https://www.salesforce.com/blog/efficient-rl-training-agentic-era/)。

### 3. 太迷信 RL，小模型碰到了“推理天花板”

很多新手有一种错觉：“只要我用 RL 训得足够久，哪怕是个 1B 的小模型，也能像 GPT-4 一样聪明。”
但 RL 的本质是“激发模型已有的潜力”，而不是“无中生有注入新知识”。

> **Amazon Science** 的实验发现：32B 模型从 RL 中获益极大，因为模型本身聪明，能生成高质量的轨迹（Rollout），形成正反馈循环。但 3B 的小模型在做某些任务时，甚至连“这个问题无法回答”都判断不出来。这种基础能力的缺失，RL 怎么惩罚都没用。这就好比你让一个小学生去考微积分，你再怎么扣他分，他也写不出来，因为他压根没学过。所以：对于小模型，遇到瓶颈了，老老实实回去做 SFT 或者蒸馏，别死磕 RL [\[参考\]](https://www.amazon.science/blog/customizing-multiturn-ai-agents-with-reinforcement-learning)。

### 4. 不能全靠 RL，要分阶段培养（三阶段管线）

既然小模型不能直接硬上 RL，大模型直接做 RL 也容易崩溃，那正确的姿势是什么？

> **Alibaba 通义团队**没有把希望全寄托在 RL 上，而是设计了稳扎稳打的“三步走战略”（**Agentic CPT → Agentic SFT → Agentic RL**）。他们在预训练（CPT）阶段，就把工具调用的轨迹重写成文本喂给模型；在 SFT 阶段教它怎么思考；最后才用 RL 做提纯。他们发现，如果在起步阶段模型就没有基本的工具使用能力，直接上 RL 就是一场灾难 [\[参考\]](https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/)。

---

## 实战避坑 Checklist (TL;DR) {#tricks}

如果你马上要动手写代码了，请把这张表贴在显示器旁边：

| 现象 / 目标               | 工业界救命药方                                                               | 来源参考              |
| ------------------------- | ---------------------------------------------------------------------------- | --------------------- |
| **没数据/想省钱**         | 用离线数据搞“合成环境”；几十条高质量小数据也能出奇迹。                       | Alibaba / Amazon      |
| **Loss 莫名爆炸**         | 检查底层 CUDA Kernel，确保推理（Rollout）和训练（Backward）是完全一致的。    | LinkedIn              |
| **模型变复读机/乱码**     | **极简 Reward**（只奖成败）；加上**长度过滤**（把太长的输出直接扔掉）。      | Bespoke Labs          |
| **策略偏离/瞎探索**       | 设微小的 KL 惩罚（0.001）；**每 100 步把当前模型设为新的 Reference Model**。 | Bespoke Labs          |
| **回答越来越长（话痨）**  | 用 **Gamma-decay 衰减奖励**（越快解决，给钱越多）。                          | Moonshot              |
| **格式坍塌（忘记格式）**  | 采用**保守负采样**，过滤掉因超长等客观原因失败但格式正确的轨迹。             | Alibaba               |
| **长任务导致 OOM/失忆**   | 教模型做“摘要过滤”，主动丢弃没用的历史信息（Context Management）。           | Moonshot              |
| **GPU 等待时间太长**      | 把没跑完的超长任务强制切断，存进 Buffer，下次用新权重接着跑。                | Moonshot / Salesforce |
| **MoE 训练报错/利用率低** | 修正 Gating 的 On-policy 假设（detach）；引入专家并行和流水线同步。          | LinkedIn / Salesforce |
| **小模型训练卡瓶颈**      | 不要硬上 RL，先用 SFT 或蒸馏提升基础能力；采用 **CPT -> SFT -> RL** 三阶段。 | Amazon / Alibaba      |

## 参考资料 {#references}

- Zhu J, Sang H, et al. "[Unlocking Agentic RL Training for GPT-OSS: A Practical Retrospective](https://huggingface.co/blog/LinkedIn/gpt-oss-agentic-rl)." Hugging Face Blog, 2026. —— LinkedIn 团队在 GPT-OSS MoE 模型上的 Agentic RL 调试实践，包含 attention sink backward 实现。
- Zhuang R, Vu T, et al. "[Improving Multi-Turn Tool Use with Reinforcement Learning](https://www.bespokelabs.ai/blog/improving-multi-turn-tool-use-with-reinforcement-learning)." Bespoke Labs Blog, 2025. —— GRPO 训练多轮工具调用的详细踩坑记录和稳定性 recipe。
- Moonshot AI. "[Kimi-Researcher: End-to-End RL Training for Emerging Agentic Capabilities](https://moonshotai.github.io/Kimi-Researcher/)." 2025. —— 端到端 REINFORCE 训练自主研究智能体，包含 partial rollout 和 context management 机制。
- Tongyi DeepResearch Team. "[Tongyi DeepResearch: From Chatbot to Autonomous Agent](https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/)." 2025. —— 三阶段 Agentic CPT → SFT → RL 管线，30B MoE 模型的 Deep Research Agent。[GitHub](https://github.com/Alibaba-NLP/DeepResearch)
- Salesforce AI Research. "[Building Efficient RL Training for the Agentic Era](https://www.salesforce.com/blog/efficient-rl-training-agentic-era/)." 2026. —— SFR-RL 的流水线同步架构和 MoE Expert Parallelism 优化。
- Subramanian S, Xu P, Wang Y. "[Customizing Multiturn AI Agents with Reinforcement Learning](https://www.amazon.science/blog/customizing-multiturn-ai-agents-with-reinforcement-learning)." Amazon Science Blog, 2026. —— 小数据（72 题）RL 定制 Agent 的实践，证明数据质量 > 数量。

---

通过这些血泪坑的梳理，你应该对 Agentic RL 训练有了一个具象的感受：“别整虚的，先跑通、别乱码、快点干活”。下一节，我们将看到这些技术的最前沿综合应用——[深度研究智能体：Deep Research Agent](./deep-research-agent)，看看 Agentic RL 如何训练出能自主进行长程研究的 AI。
