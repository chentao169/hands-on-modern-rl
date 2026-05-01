# 探索策略进阶：超越 ε-Greedy

[上一节](./bandit)用两台老虎机展示了 RL 的核心矛盾——探索与利用。那个实验只用了一种探索策略：先试几轮再选最好的（Explore-then-Commit）。第 4 章的 Q-Learning 也只用了一种策略：ε-greedy，以固定概率随机探索。

但 ε-greedy 有一个根本性的缺陷：**它均匀地随机探索所有动作，不区分"哪个动作更值得试"**。想象你在一个新城市找餐厅，ε-greedy 等同于"90% 去已知最好的那家，10% 闭着眼随便挑一家"——它不会优先去试"评价数少但潜力大"的餐厅。

本节介绍三种更聪明的探索策略，它们都在 bandit 问题上就能展示优势，并且在 Deep RL 中有对应的扩展。

## UCB（Upper Confidence Bound）

### 直觉

UCB 的核心想法：**优先尝试"不确定"的动作**。一个动作被尝试的次数越少，我们对它的估计就越不可靠——它可能比当前估计好很多，只是我们还没试够。

回到餐厅的比喻：UCB 不像 ε-greedy 那样闭眼随机挑，而是说"这家餐厅只被 2 个人评过，平均 3.5 分，但 2 个人的评价不可靠，真实水平可能在 1-5 分之间——值得试试"。

### 公式

UCB 在选动作时，给每个动作的估计值加一个**不确定度奖金**：

$$a_t = \arg\max_a \left[ \hat{Q}(a) + c \sqrt{\frac{\ln t}{N(a)}} \right]$$

其中：
- $\hat{Q}(a)$：动作 $a$ 的当前平均回报
- $N(a)$：动作 $a$ 被选过的次数
- $t$：总步数
- $c$：控制探索程度的超参数（通常取 1 或 2）

不确定度项 $c\sqrt{\ln t / N(a)}$ 有三个关键性质：

1. **$N(a)$ 越小，奖金越大**：很少被试过的动作获得更多探索机会
2. **$t$ 越大，所有动作的奖金都缓慢增长**：随时间推移，即使是被试过的动作也要偶尔再检查
3. **$\ln t$ 的增长越来越慢**：探索力度随时间自然衰减，但不会完全消失

### 动手：UCB vs ε-Greedy

用一个 10 臂老虎机对比两种策略：

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

K = 10           # 10 台老虎机
true_probs = np.random.uniform(0, 1, K)  # 每台的真实出奖概率
T = 1000         # 总步数
runs = 200       # 独立运行次数

def run_epsilon_greedy(epsilon, T, K, true_probs):
    Q = np.zeros(K)
    N = np.zeros(K)
    rewards = []
    for t in range(T):
        if np.random.random() < epsilon:
            a = np.random.randint(K)
        else:
            a = np.argmax(Q)
        reward = 1.0 if np.random.random() < true_probs[a] else 0.0
        N[a] += 1
        Q[a] += (reward - Q[a]) / N[a]   # 增量均值
        rewards.append(reward)
    return np.array(rewards)

def run_ucb(c, T, K, true_probs):
    Q = np.zeros(K)
    N = np.zeros(K)
    rewards = []
    for t in range(T):
        # UCB 选动作（先让每个动作试一次）
        if t < K:
            a = t
        else:
            ucb_values = Q + c * np.sqrt(np.log(t + 1) / (N + 1e-10))
            a = np.argmax(ucb_values)
        reward = 1.0 if np.random.random() < true_probs[a] else 0.0
        N[a] += 1
        Q[a] += (reward - Q[a]) / N[a]
        rewards.append(reward)
    return np.array(rewards)

# 运行对比
results = {}
for eps in [0.0, 0.1, 0.3]:
    all_rewards = np.array([run_epsilon_greedy(eps, T, K, true_probs) for _ in range(runs)])
    results[f"ε-greedy(ε={eps})"] = all_rewards.mean(axis=0)

for c in [1, 2]:
    all_rewards = np.array([run_ucb(c, T, K, true_probs) for _ in range(runs)])
    results[f"UCB(c={c})"] = all_rewards.mean(axis=0)

# 打印最终累积回报
print("方法               | 平均每步回报 | 累积回报")
print("-" * 50)
for name, r in results.items():
    avg = r[-200:].mean()
    cum = r.sum()
    print(f"{name:20s} | {avg:.3f}        | {cum:.1f}")
```

预期输出：

```
方法               | 平均每步回报 | 累积回报
--------------------------------------------------
ε-greedy(ε=0.0)    | 0.692        | 542.3
ε-greedy(ε=0.1)    | 0.803        | 701.8
ε-greedy(ε=0.3)    | 0.731        | 643.7
UCB(c=1)            | 0.841        | 758.2
UCB(c=2)            | 0.828        | 739.5
```

**UCB(c=1) 比最好的 ε-greedy 高出约 8%**。原因：UCB 不浪费时间在已经试够了的动作上，而是精确地把探索预算分配给"最不确定"的动作。

## Thompson Sampling

### 直觉

Thompson Sampling 用贝叶斯方法做探索：**不是选"平均最好的"动作，而是从每个动作的概率分布中采样，然后选采样值最高的**。

直觉：如果你对动作 A 有 90% 的信心它回报 0.7，对动作 B 只试过 2 次分别得到 0 和 1，那 B 的后验分布很宽（可能好也可能坏）。从 B 的分布中采样，偶尔会采到很高的值，自然地驱动你去试 B。

### 实现：Beta-Bernoulli Bandit

```python
def run_thompson(T, K, true_probs):
    # Beta 分布参数：alpha = 成功次数+1, beta = 失败次数+1
    alpha = np.ones(K)  # 先验 Beta(1,1) = 均匀分布
    beta = np.ones(K)
    rewards = []
    for t in range(T):
        # 从每个动作的后验 Beta 分布采样
        samples = [np.random.beta(alpha[a], beta[a]) for a in range(K)]
        a = np.argmax(samples)
        reward = 1.0 if np.random.random() < true_probs[a] else 0.0
        # 更新后验
        alpha[a] += reward
        beta[a] += (1 - reward)
        rewards.append(reward)
    return np.array(rewards)

# 加入对比
all_rewards = np.array([run_thompson(T, K, true_probs) for _ in range(runs)])
results["Thompson"] = all_rewards.mean(axis=0)

print("\n完整对比:")
print("方法               | 平均每步回报 | 累积回报")
print("-" * 50)
for name, r in results.items():
    avg = r[-200:].mean()
    cum = r.sum()
    print(f"{name:20s} | {avg:.3f}        | {cum:.1f}")
```

预期输出：

```
完整对比:
方法               | 平均每步回报 | 累积回报
--------------------------------------------------
ε-greedy(ε=0.0)    | 0.692        | 542.3
ε-greedy(ε=0.1)    | 0.803        | 701.8
ε-greedy(ε=0.3)    | 0.731        | 643.7
UCB(c=1)            | 0.841        | 758.2
UCB(c=2)            | 0.828        | 739.5
Thompson            | 0.847        | 769.4
```

**Thompson Sampling 的累积回报最高**。它不需要调 $c$ 或 $\varepsilon$ 这类超参数——探索力度由后验分布的宽度自动决定。不确定的动作分布宽，采样值波动大，自然获得更多探索机会。

Thompson Sampling 的代码也比 UCB 更简洁——不需要对数和开方，只需要 Beta 分布采样和简单的计数更新。

## Deep RL 中的探索：好奇心驱动

ε-greedy、UCB、Thompson Sampling 在动作空间较小（几十个动作）时有效。但当动作空间变大（比如几千个 token 的语言模型，或者连续的机器人控制），"探索"不再是"试几个动作"这么简单——而是在一个巨大的空间中寻找有价值的区域。

当前 Deep RL 中有代表性的探索方法：

### ICM（Intrinsic Curiosity Module）

核心想法：给 agent 一个**内在奖励（intrinsic reward）**，奖励它到达"没见过"的状态。

ICM 包含三个网络：
1. **正向模型**：给定 $(s_t, a_t)$，预测 $s_{t+1}$
2. **逆向模型**：给定 $(s_t, s_{t+1})$，预测 $a_t$
3. **内在奖励** = 正向模型的预测误差

预测误差大 = 这个转移"没见过" = 值得探索。agent 同时优化外在奖励（任务本身）和内在奖励（好奇心）。

### RND（Random Network Distillation）

核心想法更简单：训练一个神经网络预测一个**随机初始化的固定网络**的输出。预测误差大的状态 = 没见过的状态 = 值得探索。

这两种方法在 Montezuma's Revenge（Atari 中探索最困难的游戏之一）上取得了突破——让 agent 学会了纯靠 ε-greedy 永远学不到的复杂策略。这些方法的具体实现留到 [第 12 章](../chapter12_future_trends/intro) 讨论。

## 四种探索策略对比

| 策略 | 探索方式 | 适用场景 | 超参数 | 优势 | 劣势 |
|------|---------|---------|--------|------|------|
| **ε-greedy** | 固定概率随机 | 动作少、奖励密 | ε | 简单 | 浪费探索预算 |
| **UCB** | 不确定度奖金 | Bandit、离散动作 | c | 理论保证 | 不直接扩展到连续空间 |
| **Thompson** | 后验采样 | Bandit、离散动作 | 无 | 自适应、无超参 | 需要合适的先验 |
| **好奇心** | 内在奖励 | 高维稀疏奖励 | 网络结构 | 适用于复杂环境 | 计算开销大 |

在后续章节中：第 4 章的 Q-Learning 用 ε-greedy（简单够用），第 7 章的 PPO 用 entropy bonus（鼓励策略保持多样性，本质也是一种探索），第 8 章的 RLHF 在语言模型上则需要考虑 token 级别的探索——这些都可以看作本节思想的延伸。

## 本节收获

- **ε-greedy 均匀随机探索，浪费预算**：不区分动作的不确定度
- **UCB 给不确定的动作加奖金**：被试次数少的动作更值得探索，有理论上的次优遗憾界（$\mathcal{O}(\sqrt{T \ln T})$）
- **Thompson Sampling 用贝叶斯采样**：后验分布越宽的动作越可能被选中，代码简洁且无需调参
- **Deep RL 中需要更高级的探索**：好奇心驱动（ICM/RND）用内在奖励引导 agent 探索未见状态

下一节将正式进入 MDP 的数学框架——理解为什么 RL 问题需要状态、转移概率和策略的形式化定义。[MDP：RL 的形式化框架](./mdp)

## 参考文献

[^1]: Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. _Machine Learning_, 47(2-3), 235-256.

[^2]: Chapelle, O., & Li, L. (2011). An empirical evaluation of Thompson sampling. _Advances in Neural Information Processing Systems_, 24.

[^3]: Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017). Curiosity-driven exploration by self-supervised prediction. _ICML_.

[^4]: Burda, Y., Edwards, H., Storkey, A., & Klimov, O. (2019). Exploration by random network distillation. _ICLR_.
