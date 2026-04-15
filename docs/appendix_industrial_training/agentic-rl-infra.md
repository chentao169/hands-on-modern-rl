# B.4 Agentic RL 基础设施

> 前面 B.1-B.3 讲了 LLM RL 的基础设施。但那些都假设模型只是在 GPU 上老老实实生成文本。如果你训练的是 Agent——会调工具、执行代码、多轮交互——事情就完全不一样了。

## 一问一答 vs. 多轮行动

回忆一下第 8 章的 GRPO 训练。给模型一道数学题，模型一次性写完解答，然后检查答案对不对。整个过程一轮就结束了，模型从头到尾只做一件事：生成文本。所有计算都在 GPU 上完成，干净利落。

但如果你想训练一个会修 bug 的 Agent，事情就变得复杂了。模型拿到一段有错误的代码，需要先读代码，判断哪里有问题，修改代码，跑测试看看改对了没有。如果测试没过，它还得继续改。一个任务可能要来回五六个回合才能完成，而每一步之间模型都要等——读文件要等磁盘 IO，跑测试要等沙箱执行，搜资料要等网络响应。这些操作不在 GPU 上，延迟从几十毫秒到几秒不等。

这就是 Agentic RL 和 LLM RL 的根本区别。前者是"一问一答"，模型只管说话；后者是*多轮行动*（multi-turn interaction），模型要在环境里做事——写代码、执行代码、搜索、调 API——然后根据结果决定下一步做什么。听起来区别不大，但对基础设施来说，复杂度直接上了一个数量级。因为 LLM RL 的训练循环是纯 GPU 的，一切都在显存里完成。而 Agentic RL 的训练循环牵扯到 GPU（模型推理）、CPU（工具执行）、网络（API 调用）三种完全不同的资源，它们之间的协调、隔离、容错、存储都是全新的工程问题。

## 沙箱隔离

Agent 会执行代码，这是它区别于普通聊天模型的核心能力，也是最大的安全隐患。

在训练过程中，模型会尝试各种各样的策略来获取更高分数。如果不对它加以限制，它可能发现一条"捷径"：生成一段 `os.system("rm -rf /")` 来删除训练服务器上的文件，或者读取环境变量里的 API key。这些行为当然不是恶意的——模型只是在探索所有可能的动作空间。但后果是灾难性的。想象一下，你正在跑一个持续三天的训练任务，凌晨两点 Agent 的某次探索直接把训练集群的文件系统清空了，这种感觉大概不会太好。

所以 Agent 执行代码必须在*隔离环境*（sandbox）里跑。最常用的方案是 Docker 容器，它提供了文件系统隔离、资源限制和网络限制：

```python
container = client.containers.run(
    "python:3.11-slim",
    command=f"python -c '{code}'",
    detach=True,
    mem_limit="512m",       # 内存限制
    cpu_quota=50000,         # CPU 限制
    network_mode="none",     # 禁止联网
    remove=True,
)
```

这里 `network_mode="none"` 尤其值得注意。训练中的 Agent 不应该真的去访问外网——既不安全，也不可复现。同一个 Agent 第二次跑同样的任务，搜索引擎返回的结果可能已经变了，训练轨迹就无法重现。当然，Web Agent 之类的场景确实需要联网，这时要通过代理做请求过滤，而不是简单地断网。

一个容易被忽视的细节是容器的启动开销。Docker 容器创建一次要大约 100 毫秒。如果同时跑 1000 个 episode，每个都要创建新容器，光启动就花了一分钟。工业级的做法是维护一个*预热容器池*（warm container pool）——提前创建好 N 个容器，用完回收重置而不是销毁，启动开销能降到大约 5 毫秒。

## 多轮轨迹存储

LLM RL 的训练数据很简单：每个样本就是一个三元组 `(prompt, completion, reward)`——题目、回答、分数，一行 JSON 搞定。

Agentic RL 的训练数据则像一棵对话树。一个 episode 可能包含七八轮对话，每轮有模型输出和工具返回。比如模型拿到一个"修复 Python bug"的任务，它先读代码，然后修改，跑测试发现失败，继续修改，再跑测试通过——这些交互过程都需要完整记录下来。不仅如此，你还需要按任务类型检索（看看"数学做得好但代码做得差"的模式），按步骤切片（分析到底是哪一步走错了），以及处理去重和过期（同一个任务不要重复训练，旧轨迹可能因为环境变化而失效）。

规模小的时候（不到一万条轨迹），JSON 文件加 SQLite 就够了。中等规模（一万到一百万条）可以用 Redis 做索引、S3 存数据。超过一百万条就需要分布式数据库（MongoDB 或 DynamoDB）了。如果训练的是多模态 Agent，轨迹里还会有图片和音频——这时候应该存引用（URL）不存原始数据，训练时按需下载，保持轨迹索引在 KB 级别。

## GPU 空等问题

前面 B.2 讨论过 LLM RL 的 GPU 空等问题：生成和训练串行，训练 GPU 有 99% 的时间在等生成完成。Agentic RL 把这个问题推到了更极端。

来看看单条 Agentic 轨迹的时间线：GPU 生成一个动作只要 3 毫秒，然后 CPU 执行工具要 500 毫秒。GPU 在这 500 毫秒里什么都没干，就在等。下一轮又一样：GPU 3 毫秒，CPU 300 毫秒。几轮下来，GPU 实际工作时间还不到 1%。这个问题比 LLM RL 严重得多——LLM RL 的空等发生在 rollout 和 training 之间，Agentic RL 的空等发生在每一轮交互里。

解法和 B.2 一脉相承：同时跑多条轨迹。轨迹 A 在等工具返回的时候，GPU 给轨迹 B 生成动作。轨迹 B 等工具的时候，GPU 给轨迹 C 生成动作。像一条流水线，GPU 一直在忙。这种设计能把 GPU 利用率从大约 1% 拉到 70-80%，吞吐量提升 50-100 倍。

上面解决的是"批次内"的并发。批次之间还有 B.2 讲的 Rollout 和 Training 串行问题。完整方案是两级异步：批次内多条轨迹并发（GPU 和工具交替工作），批次间 Rollout 和 Training 通过数据队列解耦（Rollout 不停生成，Training 不停训练）。

## 案例：Relax

前面讲了通用的问题和解法。现在来看一个把这些想法落地的工业级系统——Relax，小红书 AI Infra 团队开源的多模态 Agentic RL 后训练框架。它是少数支持全模态（文本、图像、音频）Agentic RL 训练的引擎之一。

### 架构

Relax 最核心的设计选择是把训练流程中的每个角色都部署为独立的 Ray Serve 服务。Actor、Rollout、Critic、Reference、Advantages、GenRM 各自独立运行。为什么要这样做？因为 Agentic RL 的组件是异构的：推理需要 GPU、工具执行需要 CPU、编排需要 CPU 和内存。独立部署让每个组件可以按需扩缩、独立容错，不用互相迁就。

```
┌───────────────────────────────────────────────────────────────┐
│  Entrypoints:  train.py                                        │
├───────────────────────────────────────────────────────────────┤
│  Orchestration:  Controller (训练循环) │ Service │ Registry    │
├───────────────────────────────────────────────────────────────┤
│  Components:  Actor │ Rollout │ Critic │ ActorFwd │ GenRM     │
├───────────────────────────────────────────────────────────────┤
│  Engine:  SGLang 推理 │ 奖励函数库 │ 路由 │ 过滤器             │
├───────────────────────────────────────────────────────────────┤
│  Backends:  Megatron-LM (训练) │ SGLang (推理)                 │
├───────────────────────────────────────────────────────────────┤
│  Distributed:  Ray Actor Groups │ DCS (权重同步)               │
└───────────────────────────────────────────────────────────────┘
```

训练后端是 Megatron-LM，支持 B.3 讲过的 TP/PP/CP/EP 全套并行策略。推理后端是 SGLang。两者之间通过 Megatron Bridge 自动做权重格式转换。

### TransferQueue

Relax 最有意思的工程创新是 TransferQueue。

回顾一下 B.2 的异步训练：Rollout 生成数据写入 Buffer，Training 从 Buffer 读取数据训练。传统 Buffer 是批量的——Rollout 生成完整个 batch 才写入，Training 等有数据了才读取。这导致两边总有一个人在等：Rollout 写太快 Buffer 溢出，Training 读太快 Buffer 空了 GPU 闲着。

TransferQueue 把这个交互改成了流式的。Rollout 每生成一个样本就写入队列，Training 端每拿到一个样本就开始处理，不用等整个 batch 生成完。配合 DCS（Distributed Checkpoint Service）做权重同步——Training 每更新一步参数，DCS 通过 NCCL 广播给 Rollout 等组件，和下一次训练计算重叠进行，不占额外时间。

这个设计看起来简单，但它解决的是异步训练中一个很根本的效率瓶颈：Batch 级别的等待变成了 Sample 级别的等待，等待时间缩短了一个数量级。

### 两种执行模式

Relax 提供两种模式来适应不同的硬件条件。

*Collocate 模式*下，Actor 和 Rollout 共享同一组 GPU，轮替使用。Rollout 生成完一个 batch，让出 GPU 给 Training。这适合 GPU 数量有限的情况，而且可以做到严格的 on-policy——模型参数没有任何延迟，Training 永远在用最新版本的模型生成的数据。

*Fully Async 模式*下，各角色跑在独立的 GPU 集群上，通过 TransferQueue 交换数据，通过 DCS 异步同步权重。参数 `--max-staleness` 控制允许多"旧"的数据参与训练——设 0 就是严格 on-policy，设大就允许更多异步换取吞吐。这和 B.2 讨论的"旧数据怎么处理"是同一个问题，只不过 Relax 把它做成了一个可调参数。

### Agentic RL 的几个关键设计

训练 Agentic RL 时有一个容易踩的坑：多轮轨迹里的所有 token 都参与 loss 计算。但这不对——工具返回的结果不是模型生成的，模型不应该为它负责。模型需要学习的是"什么时候调什么工具、怎么理解工具结果"，而不是"怎么输出工具结果"。Relax 用 *loss mask* 来处理这个问题：模型生成的 token 标记为 mask=1 参与训练，工具返回的 token 标记为 mask=0 不参与训练。

另一个设计是环境接口解耦。`BaseInteractionEnv` 只提供 `reset` / `step` / `format_observation` 三个方法，环境实现和 Rollout 逻辑完全分离。换一套工具环境不需要改训练代码——这听起来像是一个理所当然的设计，但在实际项目中，环境和训练逻辑纠缠不清是非常常见的坑。

多模态 Agent 还有一个特殊需求：多轮对话里，第一轮用户发的图片，到第三轮模型还需要看到。Relax 在 Rollout 端维护 `image_data`，在 Training 端维护 `multimodal_train_inputs`，每轮自动合并。

### 弹性扩展

RL 训练的 60-70% 时间花在 Rollout 上。训练跑到一半发现 Rollout 速度跟不上，能不能不停训练动态加几个推理引擎？Relax 通过 HTTP REST API 实现了这一点：

```bash
# 在当前集群里加引擎
curl -X POST http://controller:8000/scale \
  -d '{"target_engine_count": 4, "mode": "ray_native"}'

# 或者注册其他集群已有的引擎（跨集群联邦推理）
curl -X POST http://controller:8000/scale \
  -d '{"engine_urls": ["gpu-cluster-2:8000"], "mode": "external"}'
```

`external` 模式特别有意思——你可以利用其他 GPU 集群上的空闲资源或抢占式实例来加速 Rollout，不用把它们搬到当前集群里。

### 算法、模型和运维

Relax 内置了四种算法：GRPO（见 [8.1-8.2 节](/chapter08_grpo_rlvr/grpo-mechanism)）、GSPO、SAPO 和 OPD（见 [8.5 节](/chapter08_grpo_rlvr/on-policy-distillation)）。加新算法只需要实现一个 Service 类注册到 `ALGOS` 字典。

模型方面支持 Qwen3 全系列（4B、30B-A3B MoE）、Qwen3-VL（视觉语言）、Qwen3-Omni（全模态）和 Qwen3.5。

运维层面有 HealthManager 做心跳监控和两级自动恢复（先尝试原地重启，失败了再全局重启），Metrics Service 把训练指标分发到 TensorBoard / WandB / ClearML，以及通过 Apprise 推送告警到 Slack、微信、邮件。大规模 RL 训练跑起来不难，难的是持续稳定地跑。一个训练任务可能跑几天甚至几周，中间 GPU 故障、网络抖动、OOM 都是常态。没有自动恢复机制的话，半夜被叫起来手动重启可不是什么愉快的体验。

### 和其他框架对比

| 框架     | 出品方      | 特点                             | 多模态 | 异步       |
| -------- | ----------- | -------------------------------- | ------ | ---------- |
| AReaL    | 清华 & 蚂蚁 | 全异步，2.77x 提速               | 否     | 全异步     |
| Agent-R1 | 中科大      | MDP 扩展，过程/结果奖励分离      | 否     | 部分异步   |
| NeMo Gym | NVIDIA      | 科学 Agent 环境                  | 否     | 同步为主   |
| Relax    | 小红书      | TransferQueue + 弹性扩展 + 全模态 | 是     | 全异步流式 |

Relax 是目前唯一同时支持全模态和全异步弹性扩展的 Agentic RL 引擎。论文见 [arxiv.org/abs/2604.11554](https://arxiv.org/abs/2604.11554)，代码见 [github.com/redai-infra/Relax](https://github.com/redai-infra/Relax)。

## 选型建议

最后是一些实践建议。原型验证阶段，TRL 加 subprocess 就够了——先证明训练流程能跑通、reward 信号是对的。中等规模时可以上 veRL 或 OpenRLHF，配合 Docker 沙箱和 asyncio 做异步并发。到了大规模 Agentic 训练，就需要 Relax 或 AReaL 这样的全异步框架了。如果是多模态 Agent，Relax 是目前唯一的选择。

不要一开始就追求最复杂的架构。就像写代码先跑通测试用例再考虑性能优化，基础设施也一样——先跑通，再优化，最后生产化。

## 参考文献

[^relax_paper]: Xiaohongshu AI Infra Team. "[Relax: Reinforcement Learning Engine Leveraging Agentic X-modality](https://arxiv.org/abs/2604.11554)." arXiv:2604.11554, 2026. [GitHub](https://github.com/redai-infra/Relax)

[^1]: HuggingFace Blog, "[Async RL Training Landscape — 16 Open-Source Libraries Compared](https://huggingface.co/blog/async-rl-training-landscape)", 2026.

[^2]: PyTorch Blog, "[A Primer on LLM Post-Training](https://pytorch.org/blog/a-primer-on-llm-post-training/)", 2025.

[^3]: AReaL Team. "[AReaL: Async RL for Language Reasoning](https://arxiv.org/abs/2505.24298)." arXiv:2505.24298, 2025. [GitHub](https://github.com/inclusionAI/AReaL)
