# 11.4 视觉生成模型的 RL 后训练

前三节我们一直在讨论 VLM 的**理解**侧：模型看一张图，回答一个问题，RL 的目标是让它看得更准、答得更稳。现在我们转到视觉 AI 的另一侧：**生成**。给模型一段文字，它要生成一张图，或者一段视频。

这件事表面上像“让模型画得更好看”。但在真实应用里，用户很少只要“好看”。用户真正想要的是：主体要对，数量要对，位置关系要对，细节要对，整体风格还要自然。

例如 prompt 写的是：

> 玻璃走廊里有三把红色雨伞，右侧墙上有一块蓝色指示牌。

模型生成了一张很漂亮的玻璃走廊，但只有两把伞，指示牌也不是蓝色。这张图应该给高分还是低分？如果只看审美，它可能很好；如果看指令遵循，它明显失败。

因此，视觉生成 RL 的核心问题是：

> **能不能把“生成得好”拆成可以学习、可以比较、可以优化的反馈信号？**

本节会从零开始建立这件事的语言：什么是生成模型里的 action，什么是 reward model，为什么 Diffusion 可以被看成一个 MDP，DDPO 怎么更新模型，以及为什么 reward 设计往往比算法本身更关键。

![DDPO Training Teaser](./images/ref-ddpo-teaser.jpg)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>图 1：DDPO 论文/项目展示的 RL 后训练效果。不同 reward 会把 Diffusion 模型推向不同的生成偏好，直观说明视觉生成 RL 的关键：reward 设计会直接塑造最终图像分布。来源：<a href="https://github.com/kvablack/ddpo-pytorch" target="_blank" rel="noopener noreferrer">DDPO GitHub</a></em>
</div>

## 本节先建立几个最小概念

为了避免后面混乱，我们先把本节反复出现的词讲清楚。

**Prompt** 是用户输入的条件，可以是一句话，也可以是一张参考图加上一段文字。视觉生成模型的任务，是根据 prompt 生成图像或视频。

**Sample** 是模型生成出来的候选结果。同一个 prompt 可以采样很多次，每次得到的图都可能不同。

**Latent** 是模型内部使用的压缩表示。很多 Diffusion 模型并不直接在像素空间里去噪，而是在 latent 空间里生成，再由 decoder 还原成图像。

**Reward model** 是一个打分器。它输入 prompt 和生成结果，输出一个分数，表示这张图有多符合某种目标。这个目标可以是“更符合文字”，也可以是“更符合人类偏好”，还可以是“有没有满足某个细粒度约束”。

**Policy** 在 RL 里表示“智能体如何选择动作”。在语言模型中，policy 是下一个 token 的概率分布；在 Diffusion 生成中，policy 可以理解为每一步如何从当前噪声状态采样下一步 latent。

这几个概念放在一起，就是视觉生成 RL 的基本闭环：

1. 给定一个 prompt。
2. 生成模型采样一个图像或视频。
3. reward model 对生成结果打分。
4. RL 算法更新生成模型，让下一次更可能生成高 reward 的结果。

这个闭环看起来简单，但每一步都比文本问答更难。

## 为什么不能直接复用 VLM 理解侧的 RL？

第 11.1 节里，VLM 回答视觉问题，很多任务可以写出明确的规则奖励。例如目标框和标准答案的 IoU 高，就给高分；答案格式正确，就给格式分。

视觉生成没有这么幸运。它面对的是连续、高维、主观的输出空间。

| 维度     | 视觉理解 RL                   | 视觉生成 RL                                |
| -------- | ----------------------------- | ------------------------------------------ |
| 输入输出 | 图像输入，文本回答            | 文本或图像条件输入，图像/视频输出          |
| 动作空间 | 离散 token                    | 连续 latent、去噪轨迹或视频 token          |
| 奖励来源 | 答案正确性、格式、推理过程    | 文本对齐、视觉质量、人类偏好、细粒度约束   |
| 可验证性 | 很多任务可以写规则 reward     | 部分属性可验证，但整体审美和偏好很难二值化 |
| 主要风险 | 格式投机、语言 reward hacking | reward 过拟合、模式单一化、对齐和美学冲突  |

这里最重要的区别有两个。

第一，**动作空间更大**。语言模型一步选择一个 token；Diffusion 模型一步要预测整张 latent 的去噪方向。一个 token 是离散选择，一个 latent 是高维连续向量。这会让策略梯度的方差更大，训练也更难稳定。

第二，**奖励更难定义**。问答任务常常有标准答案，但生成图像没有唯一标准答案。两张图都可能合理，只是偏好不同。更麻烦的是，不同目标之间会冲突：更漂亮的图未必更听话，更严格满足属性检查的图也未必自然。

所以，生成侧 RL 的第一步不是选算法，而是把“好图”拆成足够小的可学习信号。

## 先从 Diffusion 的采样过程看起

Diffusion 模型的生成过程可以理解为“从噪声出发，逐步去噪”。

最开始，模型有一个接近随机噪声的 latent，记作 $x_T$。然后模型一步一步生成：

$$
x_T \rightarrow x_{T-1} \rightarrow \cdots \rightarrow x_1 \rightarrow x_0
$$

这里 $x_0$ 是最终图像对应的 latent。再经过 decoder，就得到用户看到的图像。

每一步去噪时，模型会看三个东西：

| 符号  | 含义                  |
| ----- | --------------------- |
| $x_t$ | 当前还带噪声的 latent |
| $t$   | 当前去噪步数          |
| $c$   | prompt 或条件信息     |

模型要决定的是下一步 latent：

$$
x_{t-1}\sim p_\theta(x_{t-1}\mid x_t,t,c)
$$

这行公式的意思是：给定当前噪声状态 $x_t$、时间步 $t$ 和 prompt $c$，模型用参数 $\theta$ 定义一个概率分布，并从中采样下一步 $x_{t-1}$。

如果只从生成模型角度看，这是一个采样公式。如果从 RL 角度看，它已经很像一个 policy。

## 把 Diffusion 翻译成 MDP 语言

DDPO（Denoising Diffusion Policy Optimization）的关键观察是：Diffusion 的采样过程可以被看成一个有限长度的 MDP。

这个翻译非常重要。我们逐项来看：

| RL 里的概念       | Diffusion 里的对应                        |
| ----------------- | ----------------------------------------- |
| 状态 $s_t$        | 当前 latent、时间步和 prompt：$(x_t,t,c)$ |
| 动作 $a_t$        | 采样下一步 latent，或预测去噪方向         |
| 轨迹 $\tau$       | 一整条去噪链路：$x_T,\ldots,x_0$          |
| 奖励 $R$          | 最终图像由 reward model 给出的分数        |
| 策略 $\pi_\theta$ | Diffusion 模型的去噪分布 $p_\theta$       |

因此，一次生成就像一个 episode：

$$
\tau=(x_T,x_{T-1},\ldots,x_0)
$$

episode 结束以后，reward model 才看到最终图像，并给出分数：

$$
R=r_\phi(x_0,c)
$$

注意这里的 $r_\phi$ 不是生成模型本身，而是另一个打分模型。它的参数是 $\phi$，生成模型的参数是 $\theta$。

这样一来，生成模型的目标可以写成：

$$
J(\theta)=\mathbb{E}_{\tau\sim p_\theta}\left[r_\phi(x_0,c)\right]
$$

这句话读作：我们希望在模型自己采样出来的轨迹上，最终图像的平均 reward 尽可能高。

## DDPO：用策略梯度更新去噪策略

有了上面的 MDP 翻译，DDPO 就不神秘了。它本质上是在 Diffusion 采样轨迹上做策略梯度。

我们先把一条去噪轨迹的概率写出来。为了简化记号，下面默认 prompt $c$ 是给定的：

$$
p_\theta(\tau\mid c)
=
p(x_T)\prod_{t=1}^{T}
p_\theta(x_{t-1}\mid x_t,t,c)
$$

这行公式有两个意思。

第一，初始噪声 $x_T$ 通常从标准高斯分布采样，它不依赖模型参数 $\theta$。第二，真正由模型控制的是每一步去噪分布 $p_\theta(x_{t-1}\mid x_t,t,c)$。

生成模型想最大化最终 reward：

$$
J(\theta)
=
\mathbb{E}_{\tau\sim p_\theta(\tau\mid c)}
\left[R(\tau,c)\right]
$$

其中 $R(\tau,c)=r_\phi(x_0,c)$，也就是 reward model 对最终图像的打分。

把期望写成积分，就是：

$$
J(\theta)
=
\int p_\theta(\tau\mid c)R(\tau,c)\,d\tau
$$

现在对 $\theta$ 求梯度：

$$
\nabla_\theta J(\theta)
=
\int \nabla_\theta p_\theta(\tau\mid c)R(\tau,c)\,d\tau
$$

这里用一个常见技巧，叫 **log-derivative trick** 或 **score-function trick**：

$$
\nabla_\theta p_\theta(\tau\mid c)
=
p_\theta(\tau\mid c)\nabla_\theta\log p_\theta(\tau\mid c)
$$

代回去：

$$
\nabla_\theta J(\theta)
=
\int p_\theta(\tau\mid c)
\nabla_\theta\log p_\theta(\tau\mid c)
R(\tau,c)\,d\tau
$$

也就是：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau\sim p_\theta}
\left[
\nabla_\theta\log p_\theta(\tau\mid c)R(\tau,c)
\right]
$$

这一步很关键。reward model 可以是不可微的，也可以是一个黑盒打分器；策略梯度不需要对 reward 本身求导，只需要知道“这条轨迹得了多少分”。

接下来展开轨迹的 log probability：

$$
\log p_\theta(\tau\mid c)
=
\log p(x_T)
+
\sum_{t=1}^{T}
\log p_\theta(x_{t-1}\mid x_t,t,c)
$$

由于 $\log p(x_T)$ 不依赖 $\theta$，求梯度时它会消失：

$$
\nabla_\theta\log p_\theta(\tau\mid c)
=
\sum_{t=1}^{T}
\nabla_\theta
\log p_\theta(x_{t-1}\mid x_t,t,c)
$$

所以最朴素的策略梯度就是：

$$
\nabla_\theta J
=
\mathbb{E}\left[
\sum_{t=1}^{T}
\nabla_\theta \log p_\theta(x_{t-1}\mid x_t,t,c)
\cdot R(\tau,c)
\right]
$$

这就是 REINFORCE 在 Diffusion 轨迹上的形式：如果某条去噪轨迹最后拿到高 reward，就提高这条轨迹中每一步采样动作的概率；如果 reward 低，就降低它们的概率。

### Baseline 和 advantage 为什么可以减？

直接用 $R(\tau,c)$ 更新会有很大方差。一个 prompt 可能天然更容易生成高分图，另一个 prompt 可能天然更难。我们更关心的是：这次采样结果是否比同类样本更好。

因此可以减去一个 baseline $b(c)$：

$$
\hat{A}=R(\tau,c)-b(c)
$$

为什么这样不会改变梯度的期望？因为：

$$
\mathbb{E}_{\tau\sim p_\theta}
\left[
\nabla_\theta\log p_\theta(\tau\mid c)b(c)
\right]
=
b(c)\int p_\theta(\tau\mid c)
\nabla_\theta\log p_\theta(\tau\mid c)d\tau
$$

再用同一个 log-derivative trick：

$$
=
b(c)\int \nabla_\theta p_\theta(\tau\mid c)d\tau
=
b(c)\nabla_\theta \int p_\theta(\tau\mid c)d\tau
=
b(c)\nabla_\theta 1
=0
$$

所以，减掉不依赖具体动作的 baseline，不会引入偏差，只会降低方差。

实际训练里，$\hat{A}$ 可以有几种常见做法：

| Advantage 做法        | 含义                                   |
| --------------------- | -------------------------------------- |
| $R-\bar{R}$           | 减去同一 batch 的平均 reward           |
| $R-b(c)$              | 减去 prompt 级别的历史平均 reward      |
| $R-V_\psi(x_t,t,c)$   | 减去 value model 对当前状态的预测      |
| normalize 后的 reward | 对 batch reward 做标准化，让尺度更稳定 |

加入 advantage 后，DDPO 常用的策略梯度可以写成：

$$
\nabla_\theta J
=
\mathbb{E}\left[
\sum_{t=1}^{T}
\nabla_\theta \log p_\theta(x_{t-1}\mid x_t,t,c)
\cdot \hat{A}_t
\right]
$$

如果只用终局 reward，那么每一步可以共享同一个 $\hat{A}$。如果训练了 value model，也可以给不同时间步不同的 $\hat{A}_t$。

### 这和 Diffusion 的 log probability 怎么对应？

在很多 Diffusion 实现中，每一步反向转移可以写成一个高斯分布：

$$
p_\theta(x_{t-1}\mid x_t,t,c)
=
\mathcal{N}\left(
\mu_\theta(x_t,t,c),
\sigma_t^2 I
\right)
$$

这里 $\mu_\theta$ 是模型预测出来的去噪均值，$\sigma_t$ 是这一步的噪声尺度。于是这一动作的 log probability 近似是：

$$
\log p_\theta(x_{t-1}\mid x_t,t,c)
=
-
\frac{1}{2\sigma_t^2}
\left\|
x_{t-1}-\mu_\theta(x_t,t,c)
\right\|_2^2
+ \text{const}
$$

这解释了伪代码里的 `step.logprob` 是什么：它不是一句抽象的 RL 符号，而是当前模型在第 $t$ 步采样出这个 $x_{t-1}$ 的对数概率。

### 从最大化目标到最小化 loss

深度学习框架通常最小化 loss，而策略梯度是在最大化 $J(\theta)$。所以实现时会写成负号：

$$
\mathcal{L}_{\text{pg}}
=
-
\mathbb{E}\left[
\sum_{t=1}^{T}
\log p_\theta(x_{t-1}\mid x_t,t,c)
\cdot \hat{A}_t
\right]
$$

最小化这个 loss 等价于最大化策略梯度目标。直观地看：

| 情况                 | loss 会推动什么                      |
| -------------------- | ------------------------------------ |
| $\hat{A}_t>0$        | 提高这一步采样动作的 log probability |
| $\hat{A}_t<0$        | 降低这一步采样动作的 log probability |
| $\hat{A}_t\approx 0$ | 基本不更新这一步                     |

这里和第 5 章 REINFORCE 的思想完全一致，只是动作从“选择 token”变成了“选择下一步 latent”。

### 为什么还需要 KL 约束？

如果只最大化 reward，模型很容易走偏。原因很简单：reward model 本身并不完美。模型可能找到一些 reward model 喜欢、但人类并不真正喜欢的模式。

所以实际训练常常保留一个参考模型 $p_{\text{ref}}$，并惩罚当前模型偏离它太远：

$$
\mathcal{L}_{\text{DDPO}}
=
\mathcal{L}_{\text{pg}}
+
\beta\,
\mathbb{E}\left[
\sum_{t=1}^{T}
\mathrm{KL}\left(
p_\theta(\cdot\mid x_t,t,c)
\|p_{\text{ref}}(\cdot\mid x_t,t,c)
\right)
\right]
$$

这个式子可以分成两部分理解：

| 项         | 作用                                     |
| ---------- | ---------------------------------------- |
| 策略梯度项 | 让高 reward 的采样轨迹更可能出现         |
| KL 项      | 限制模型不要为了追 reward 而远离原始模型 |

这和 RLHF、DPO、GRPO 里的思想相同：让模型变好，但不要把模型训飞。

### DDPO 的最小训练流程

用算法步骤写出来，DDPO 可以拆成六步：

1. 从 prompt 数据集中取一批 prompt。
2. 用当前 Diffusion 模型生成图像，并记录每一步采样的 log probability。
3. 用 reward model 给最终图像打分。
4. 用 batch 平均值、value model 或其他 baseline 计算 advantage。
5. 用策略梯度更新 Diffusion 模型。
6. 用 KL 约束把模型拉回参考分布附近。

对应的极简伪代码如下：

```python
for prompt in prompt_loader:
    trajectory = diffusion.sample_with_logprob(prompt)
    image = trajectory.final_image

    reward = reward_model(prompt, image)
    advantage = reward - reward_baseline.update(reward)

    policy_loss = 0.0
    for step in trajectory.steps:
        policy_loss -= step.logprob * advantage.detach()

    kl_loss = kl_to_reference_model(trajectory)
    loss = policy_loss + beta * kl_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

这段代码省略了很多工程细节，但保留了最重要的结构：**先采样，再打分，再把最终分数分配回去噪轨迹**。

## Reward Model：生成 RL 的真正瓶颈

到这里，算法已经有了。但生成 RL 的困难往往不在“能不能写出策略梯度”，而在“reward 到底可信吗”。

如果 reward model 太弱，它给不出有效方向；如果 reward model 有偏，生成模型就会学到偏差；如果 reward 太复杂，不同目标之间还会互相拉扯。

视觉生成的 reward 通常来自三类信号。

### 第一类：人类偏好

人类偏好数据最常见的形式是成对比较。给定同一个 prompt，让用户在两张候选图之间选择更喜欢的一张。

![Pick-a-Pic Preference UI](./images/ref-pick-a-pic-ui.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>图 2：Pick-a-Pic 的人类偏好采集界面。用户对同一 prompt 下的两张候选图进行偏好选择，这类数据可以训练 PickScore 等 reward model。来源：<a href="https://stability.ai/research/pick-a-pic" target="_blank" rel="noopener noreferrer">Stability AI Research</a></em>
</div>

这样的数据可以写成：

$$
\mathcal{D}_{\text{pref}}=\{(c,x^+,x^-)\}
$$

其中 $x^+$ 是用户更喜欢的图，$x^-$ 是被比较后落选的图。reward model 的训练目标通常是让 $x^+$ 的分数高于 $x^-$：

$$
\mathcal{L}_{\text{rm}}
=
-\mathbb{E}_{\mathcal{D}_{\text{pref}}}
\log\sigma\left(r_\phi(c,x^+)-r_\phi(c,x^-)\right)
$$

这就是 Bradley-Terry 风格的偏好建模。它不要求人类给绝对分数，只要求比较两张图哪张更好。

这种信号的优点是接近真实用户偏好。缺点是成本高，而且偏好数据会继承标注人群的审美、文化和任务分布。

### 第二类：文本-图像对齐

文本-图像对齐检查的是：图像是否真的符合 prompt。

这可以从粗到细拆成几层：

| 层次     | 例子                               | 可能的检查方法               |
| -------- | ---------------------------------- | ---------------------------- |
| 全局语义 | 是否大体生成了指定场景             | CLIP Score、VLM 判断         |
| 对象存在 | prompt 中的关键对象是否出现        | 检测器、VLM 问答             |
| 属性匹配 | 颜色、材质、大小是否正确           | 细粒度 caption 后逐项比对    |
| 关系匹配 | 左右、上下、遮挡、交互关系是否正确 | 关系抽取、VLM judge          |
| 数量匹配 | 指定数量是否正确                   | 计数模型、目标检测、VLM 检查 |

这一层和前几节的 VLM RL 直接接上了。一个训练得更会看图的 VLM，可以被拿来当 captioner、judge 或 reward model，帮助生成模型判断“有没有画对”。

### 第三类：视觉质量

视觉质量检查的是图像本身是否自然、清晰、有良好的构图和光影。常见信号包括 aesthetic score、无参考图像质量评估、人工排序等。

它很有用，但不能单独使用。因为视觉质量模型通常更容易奖励“看起来高级”的图，而不是“严格遵守 prompt”的图。生成模型如果只追这个分数，可能会变得更漂亮，但更不听话。

### Reward 不是一个越复杂越好的公式

把所有 reward 加权求和很自然：

$$
R_{\text{total}}
=
w_1R_{\text{align}}
+w_2R_{\text{quality}}
+w_3R_{\text{instruction}}
$$

但这个公式只是起点，不是答案。多组件 reward 最大的问题是：每个分量都可能被模型投机利用，而且分量之间可能互相冲突。

一个更稳的工程做法是分层使用 reward：

1. 先用规则或 VLM 检查硬约束，例如数量、颜色、对象是否满足。
2. 再用偏好模型对合格样本排序。
3. 最后用人工抽查或离线 benchmark 找 reward model 的盲区。

这样 reward 就不再是一个万能分数，而是一套筛选和校准流程。

![PickScore Ranking Examples](./images/ref-pickscore-ranking.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>图 3：PickScore 用偏好模型重新排序候选生成结果。它说明视觉 reward 不只是离线评测数字，也可以直接改变采样或排序阶段展示给用户的结果。来源：<a href="https://stability.ai/research/pick-a-pic" target="_blank" rel="noopener noreferrer">Stability AI Research</a></em>
</div>

## 两种用 reward 的方式：训练时用，还是推理时用？

有了 reward model 以后，不一定马上做 RL 微调。它有两种常见用法。

第一种是**推理时使用**，也叫 reward-guided sampling 或 reranking。比如同一个 prompt 生成 $N$ 张图，用 reward model 排序，选择分数最高的图。这个方法简单、安全，适合先验证 reward model 是否靠谱。

第二种是**训练时使用**，也就是 DDPO、DPOK 这类 RL fine-tuning。模型不只是被筛选，而是真的更新参数，把偏好内化到生成策略里。

| 方法                    | 它做什么                         | 优点                  | 缺点                     |
| ----------------------- | -------------------------------- | --------------------- | ------------------------ |
| Best-of-$N$ / reranking | 多生成几张，再用 reward model 选 | 实现简单，不改模型    | 推理成本高，能力不固化   |
| Reward-guided sampling  | 采样过程中用 reward 引导方向     | 比纯 reranking 更主动 | 每次生成仍然要额外评估   |
| RL fine-tuning          | 用 reward 更新模型参数           | 可以内化偏好          | 训练更贵，也更容易不稳定 |

实践中，经常先做 reranking。如果 reward model 连排序都排不好，就不应该直接拿它做 RL。

## 视频生成：同一个问题，多了一条时间轴

视频生成可以看成图像生成的扩展，但不能简单理解为“多生成几张图”。视频多了一条时间轴，所以 reward 也要多一层。

一段视频要同时满足三件事：

1. 每一帧都要清晰、自然、符合 prompt。
2. 相邻帧之间要连贯，主体不能突然变化。
3. 整段视频要表达 prompt 中的事件顺序。

因此，视频 reward 常常写成：

$$
R_{\text{video}}
=
\alpha \cdot \frac{1}{T}\sum_t R_{\text{frame}}(x_t,c)
+ \beta \cdot \frac{1}{T-1}\sum_t R_{\text{temporal}}(x_t,x_{t+1})
+ \gamma \cdot R_{\text{overall}}(\{x_t\}_{t=1}^T,c)
$$

这三个分量分别对应：

| 分量                  | 检查什么                       |
| --------------------- | ------------------------------ |
| $R_{\text{frame}}$    | 单帧质量和单帧文本对齐         |
| $R_{\text{temporal}}$ | 帧间一致性和运动自然性         |
| $R_{\text{overall}}$  | 整段视频是否完成 prompt 的事件 |

视频 RL 的困难也随之增加：

| 挑战          | 为什么更难                      | 常见缓解思路                          |
| ------------- | ------------------------------- | ------------------------------------- |
| 时序一致性    | 单帧都好，不代表连起来合理      | 光流一致性、轨迹一致性、视频 VLM 评估 |
| 长 horizon    | 视频 token 和 latent 数远超图像 | 分段优化、短片段 reward shaping       |
| 计算成本      | 每次采样和打分都更贵            | latent 空间训练、低帧率评估、候选重排 |
| 文本-视频对齐 | prompt 可能包含先后顺序         | 分段 caption、事件级 reward           |

直觉上，图像生成的错误常常是“某个地方画错了”；视频生成的错误常常是“前后关系断了”。这就是为什么视频 reward 更依赖片段级和整体级评估。

## On-Policy 蒸馏：把 RL 得到的能力固化下来

RL 微调后的模型可能更符合偏好，但它也可能更慢、更贵，或者只适合某个特定采样设置。On-policy 蒸馏的目标，是把 RL 后模型在当前分布上产生的高质量样本，重新变成更便宜的监督学习信号。

可以把它理解成三步：

1. 用 RL 后的 teacher 模型在线生成样本。
2. 用 reward model 或规则过滤留下高质量样本。
3. 让 student 模型学习这些样本，用更低成本复现 teacher 的行为。

这和第 8 章的蒸馏思想一致：强模型负责探索和筛选，弱模型负责把能力压缩成更便宜的推理路径。区别在于，视觉生成蒸馏通常发生在 latent、去噪轨迹或视频 token 空间里，而不是普通文本 token 空间里。

## 与前面章节的联系

视觉生成 RL 看起来和 VLM 问答离得很远，但它复用了本书前面几条主线。

| 前面章节               | 在视觉生成 RL 中的对应                                      |
| ---------------------- | ----------------------------------------------------------- |
| 第 5 章 REINFORCE      | DDPO 把去噪链路当成策略轨迹，用终局 reward 更新每一步采样   |
| 第 7 章 Reward Hacking | 生成模型可能讨好 reward model，却牺牲真实用户意图           |
| 第 8 章 RLVR           | 细粒度属性、数量、关系可以变成局部可验证信号                |
| 第 10 章 Agentic RL    | 长 horizon 信用分配、多组件 reward 和 KL 约束都再次出现     |
| 第 11.1-11.3 节 VLM RL | VLM 可以反过来做生成模型的 judge、captioner 和 reward model |

最后这一点尤其重要。理解模型和生成模型不是两条完全分开的线。VLM 学会看图以后，可以检查生成图是否符合 prompt；生成模型可以合成更丰富的数据，反过来训练 VLM。到了多模态后训练阶段，“看”和“生成”会越来越像一个闭环。

## 小结

视觉生成 RL 的目标不是简单地让模型“画得更漂亮”，而是把用户意图拆成可学习的反馈信号，让生成模型在偏好、规则和多模态评估下持续改进。

本节最重要的结论有四个：

1. **Diffusion 可以被看成 MDP**：去噪轨迹就是 episode，最终图像得到 reward，策略梯度把 reward 分配回每一步。
2. **DDPO 的核心是翻译问题**：把去噪概率看成 policy，把最终图像分数看成 reward，就能使用策略梯度。
3. **Reward model 是生成 RL 的瓶颈**：人类偏好、文本对齐和视觉质量都重要，但必须防止 reward hacking。
4. **训练和推理都可以用 reward**：reranking 更安全，RL fine-tuning 更能固化能力，视频生成则进一步放大了时序和计算问题。

到这里，我们覆盖了 VLM 的理解和生成两个方向的 RL 训练。下一章，我们将进入更广阔的前沿趋势：[具身智能、自博弈与离线 RL](../chapter13_future_trends/intro)。

## 参考资料

- Black K, Janner M, Du Y, et al. "[Training Diffusion Models with Reinforcement Learning](https://arxiv.org/abs/2305.13301)." ICLR 2024. —— DDPO，把 Diffusion 去噪过程建模为 MDP，用策略梯度优化。首次证明 RL 可以显著提升 Diffusion 模型的指令遵循能力。
- Fan Y, Watkins O, Du Y, et al. "[DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models](https://arxiv.org/abs/2305.16381)." NeurIPS 2023. —— 将策略优化与 KL 正则化结合，在线 RL 微调文本到图像扩散模型。
- Clark K, et al. "[Directly Fine-Tuning Diffusion Models on Differentiable Rewards](https://arxiv.org/abs/2309.17400)." ICLR 2024. —— 用可微 reward 直接微调 Diffusion 模型，无需 RL。
- Prabhudesai M, et al. "[Aligning Text-to-Image Diffusion Models with Reward Backpropagation](https://arxiv.org/abs/2407.08737)." 2024. —— 通过 reward 反向传播来对齐 Diffusion 模型。（注：初版 arXiv:2310.03739 已被作者撤回，此处引用更新版。）
- Wu X, et al. "[Human Preference Score v2: A Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis](https://arxiv.org/abs/2306.09341)." NeurIPS 2023. —— HPS v2，大规模人类偏好评分数据集和 reward model。
- Kirstain S, et al. "[Pick-a-Pic: Open Dataset of Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2305.01569)." NeurIPS 2023. —— 人类对生成图像的偏好数据集，用于训练 reward model。
- Girdhar R, et al. "[Emu Video: Factorizing Text-to-Video Generation by Explicit Image Conditioning](https://arxiv.org/abs/2311.10709)." ECCV 2024. —— 视频生成模型，涉及视频质量的多维度评估。
- Wu X, et al. "[Boosting Text-to-Video Generative Model with MLLMs Feedback](https://neurips.cc/virtual/2024/poster/96722)." NeurIPS 2024. —— 用多模态 LLM 反馈作为 reward model，通过 RL 微调视频生成模型。
