# Part 3: 大模型强化学习 — 知识总结

## 这一 Part 我们学了什么？

这三章是全书的核心转折点。我们把前六章积累的 RL 理论，应用到了大语言模型对齐和智能体训练的真实场景中。学完后，你应该掌握以下核心知识点：

- **RLHF 工程流水线**：SFT → RM → RL 三阶段，混合奖励函数 $R = R_{\text{RM}} + \alpha R_{\text{format}} + \beta R_{\text{length}}$，奖励黑客的防御，RLAIF 用 AI 替代人类标注。
- **DPO 隐式奖励**：$r(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$。奖励函数藏在策略模型的概率比值里，不需要单独训练奖励模型。
- **DPO 损失**：让好回答的隐式奖励高于坏回答，把 RL 问题转化为分类问题。两个模型搞定，不需要 Critic 和 RM。
- **DPO 家族**：KTO 只需好/坏标签（不需配对），SimPO 去掉参考模型，ORPO 合并 SFT 和对齐。
- **GRPO 组内归一化**：$A_i = \frac{r_i - \text{mean}(r_1, \ldots, r_k)}{\text{std}(r_1, \ldots, r_k)}$。对同一个 prompt 生成 $k$ 个回答，用组内 $z$-score 替代 Critic。只需要 1 个模型。
- **DAPO 四大改进**：Clip-Higher（给低概率动作更多上升空间）、Token 级损失（按 token 计算梯度）、动态采样（过滤已掌握的 prompt）、Overlong 过滤。
- **RLVR**：在数学、代码等客观任务中用规则验证器（答案匹配、单元测试）取代人工标注的奖励模型。DeepSeek-R1-Zero 证明纯 RLVR 训练就能涌现推理能力。
- **Agentic RL**：ORM（只在最后给奖励）vs PRM（每步给奖励），工具调用 RL 的"SFT 教格式 + RL 教策略"流程。

下面让我们逐章复习这些内容。

## 第 8 章：RLHF 完整流水线——从理论到工程

### 奖励函数设计

在真实的 RLHF 系统中，奖励函数远不止"一个 RM 模型打分"这么简单。通常是一个混合公式：

$$R_{\text{total}} = R_{\text{RM}} + \alpha R_{\text{format}} + \beta R_{\text{length}} + \gamma R_{\text{correctness}}$$

奖励函数的粒度也很重要：Sequence-level（整个回答一个分数）、Step-level（每步打分，即 PRM）、Token-level（每个 token 都有奖励信号）。

### 奖励黑客：模型学会骗分

奖励黑客（Reward Hacking）的经典表现包括：长度膨胀、重复刷分、格式作弊。防御手段包括分析长度与奖励的相关性、统计高频短语、定期做人工评估。KL 散度惩罚 $-\beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ 是一道防线。

### RLAIF：用 AI 替代人类

RLAIF 用更强大的模型来替代人类做偏好标注。Constitutional AI 让模型自己对生成的回答做批评和修订，形成数据闭环：部署模型 → 收集用户反馈 → 识别薄弱环节 → 用 AI 构造偏好数据 → 重新训练。

## 第 9 章：对齐与推理强化（DPO + GRPO + RLVR）

### 从 RLHF 到 DPO：一个关键的数学等价

传统 RLHF 的流程很复杂：先用人类偏好数据训练一个奖励模型（Reward Model），再用 PPO 训练语言模型来最大化这个奖励，同时加一个 KL 散度惩罚防止模型跑偏。整个过程需要四个模型同时跑，显存和工程复杂度都很高。

DPO 的突破在于一个优美的数学发现：在 KL 约束下的 RL 目标

$$\max_\pi \mathbb{E}_{x,y \sim \pi}[r(x,y)] - \beta D_{\text{KL}}(\pi \| \pi_{\text{ref}})$$

的闭式最优解恰好是

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)$$

把两边取对数再整理，奖励函数可以用策略的概率比来表达：

$$r(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

由于 $Z(x)$ 只依赖提示词 $x$ 不依赖回答 $y$，在 Bradley-Terry 偏好模型 $P(y_w \succ y_l|x) = \sigma(r(x,y_w) - r(x,y_l))$ 中，$Z(x)$ 会自动消掉。于是我们得到了 DPO 损失：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

### GRPO：用组内统计替代 Critic

PPO 需要一个 Critic 网络来估计优势函数 $A(s,a)$。在 LLM 场景中，这个 Critic 本身就是一个大模型，显存开销巨大。GRPO（Group Relative Policy Optimization）提出了一个巧妙的替代方案：

对同一个 prompt $x$，让模型生成 $k$ 个回答 $y_1, y_2, \ldots, y_k$，用奖励函数给每个回答打分 $r_1, r_2, \ldots, r_k$。然后计算组内归一化的优势：

$$A_i = \frac{r_i - \text{mean}(r_1, \ldots, r_k)}{\text{std}(r_1, \ldots, r_k)}$$

这和 PPO 用 Critic 计算 $A = Q - V$ 的逻辑完全一致——都是"比基准好多少"——只是 GRPO 用组内统计代替了 Critic 网络。

### DAPO：让 GRPO 更强的四个改进

**Clip-Higher** 把上下裁剪范围解耦，给探索更多呼吸空间。**Token 级损失** 按 token 求和梯度，精确定位出错位置。**动态采样** 过滤已掌握的 prompt，保持训练难度梯度。**Overlong 过滤** 移除超出长度限制的样本。

### RLVR：用规则验证取代人工标注

GRPO/DAPO 不再依赖 RM——只要有一个能打分的东西就行。在数学推理、代码生成等客观任务中，这个打分者可以是一个规则验证器：数学题直接比对最终答案，代码题跑一遍单元测试。DeepSeek-R1-Zero 的实验证明，一个基础模型不加任何 SFT，只用 RLVR 训练，就能涌现出 chain-of-thought 推理能力。

## 第 10 章：Agentic RL——让模型学会使用工具

### 多轮交互与信用分配

传统的 RL 对齐是"单轮"的，但真实的 Agent 是多轮交互的。**ORM**（Outcome Reward Model）只在最后给奖励，信号太稀疏。**PRM**（Process Reward Model）给每一步都打分，提供密集信号但标注成本更高。

### 工具调用的 RL 训练

Web Agent 和 Code Agent 是 Agentic RL 的典型场景。训练流程通常是"SFT 教格式 + RL 教策略"：先用监督学习教会模型如何调用工具，再用 RL 教会模型何时、如何使用工具来完成任务。

## 小结

Part 3 展现了一条清晰的演进路线：RLHF 需要 4 个模型 → DPO 用隐式奖励砍到 2 个 → GRPO 用组内归一化砍到 1 个 → RLVR 用规则验证彻底去掉 RM。每一步都是"用更简单的机制替代更复杂的组件"，但数学上保持等价甚至更强。

同时，RL 从"对齐人类偏好"扩展到了"激发推理能力"和"训练智能体"——从单轮对话到多轮工具交互，RL 正在大模型的后训练阶段扮演越来越核心的角色。

> **下一站**：[Part 4: 前沿与进阶专题](/chapter11_vlm_rl/intro)
