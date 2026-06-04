<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "过度优化"
prev-url: "14-over-optimization"
page-title: 正则化
search-title: "第 15 章：正则化"
meta-description: "保持 RLHF 和后训练更新有用而不降低基础模型的正则化方法。"
next-chapter: "评估"
next-url: "16-evaluation"
---

# 正则化

在本书中，我们学习了许多修改模型以从人类偏好、可验证奖励和其他有价值的信号中学习的工具。
我们使用的所有方法都非常强大，并可能导致模型相对于前一训练阶段的强大通用模型（通常称为参考模型）变化过大。
当模型从给定奖励中学习过多，导致分布外性能下降时，这被称为"过度优化"（如我们在前一章中讨论的）。

在整个 RLHF 优化过程中，使用许多正则化步骤来防止奖励模型的过度优化。
在这些情况下，过度优化看起来像输出无意义文本的模型。
优化"失控"的一些例子包括输出带有极其错误答案但可追随的数学推理、重复文本、切换语言或过多特殊字符的模型。
本章涵盖了用于控制模型优化的不同方法。

截至 2026 年，在大多数 RLHF 实现中使用的最流行变体是从当前策略到参考策略的跨生成样本的 KL 距离。
"KL 距离"是一个通俗术语，用于表达训练过程中的*优化距离*，尽管 KL 散度——衡量两个概率分布分离的底层数学方法——并不满足成为真正距离度量所需的形式属性（将数字称为距离，比称之为分布差异的数值度量更简单）。
许多其他正则化技术已经在文献中出现，然后在该研究方向的下一模型迭代中消失。
也就是说，超出核心生成 KL 距离的正则化通常用于稳定实验设置，然后可以在下一代中简化。
尽管如此，理解约束 RLHF 中优化的工具很重要。

*在本章中，我们使用 $x$ 表示提示，$y$ 表示补全。这种表示法在语言模型文献中很常见，其中方法操作完整的提示-补全对，而不是单个 token。*

当在具有奖励模型 $r_\theta$ 的 RLHF 框架中使用时，一般公式如下：

$$ r = r_\theta - \lambda r_{\text{reg.}} $$ {#eq:rl_start}

参考实现为：

$$
r = r_\theta - \lambda_{\text{KL}} \mathcal{D}_{\text{KL}} \left( \pi_{\text{RL}}(y \mid x) \, \| \, \pi_{\text{ref}}(y \mid x) \right)
$$ {#eq:kl_standard}

## RL 优化中的 KL 散度

数学定义见附录 A 定义。
KL 散度衡量一个概率分布从另一个漂移了多远——当 KL 为零时，两个分布产生相同的输出。
回忆它定义如下：

$$ \mathcal{D}_{\text{KL}}(P || Q) = \sum_{x \in \mathcal{X}} P(x) \log \left(\frac{P(x)}{Q(x)}\right) $$ {#eq:kl_distance_regularization}

在 RLHF 中，感兴趣的两个分布通常是新模型版本的分布，比如 $P(x)$，和参考策略的分布，比如 $Q(x)$。
不同的优化器使用不同的 KL 方向。贯穿本书，最常用的"KL 惩罚"被称为对参考策略的反向 KL。在实践中，这简化为从 RL 模型采样 token 并从参考模型计算概率的蒙特卡洛估计。直观地说，这种反向 KL 具有一个数值特性，当新模型 $P$ 或 $\pi_{\text{RL}}$ 在原始参考模型分配低概率的地方放置显著概率质量时，施加大的惩罚。

另一个 KL 方向在 ML 中仍然经常使用，例如在一些 RL 算法的内部信任区域计算中。这种惩罚直观地在更新不对 $Q$ 或 $\pi_{\text{ref}}$ 中的高似然区域应用概率时惩罚新模型。这更接近于用于蒸馏或行为克隆的目标。

### 参考模型到生成

KL 惩罚最常通过比较训练期间生成的 token 到静态参考模型的距离来实现。
直觉是你正在训练的模型有一种你希望保持接近的风格。
这个参考模型最常是指令微调模型，但也可以是先前的 RL 检查点。
通过简单的替换，我们采样的模型变为 $\pi_{\text{RL}}(x)$ 和 $\pi_{\text{ref}}(x)$，如上 @eq:kl_standard 所示（在标准定义中，当应用于 RL KL 惩罚时，通常为 $P$ 和 $Q$）。
这种 KL 散度惩罚首先应用于对话代理，远早于大语言模型的流行 [@jaques2017sequence]，然而 KL 控制很快被确立为微调预训练模型的核心技术 [@jaques2020human]。

### 实现示例

在实践中，KL 散度的实现通常被近似 [@schulman2016klapprox]，使实现简单得多。
使用上述定义，KL 的求和可以在直接从分布 $P$ 采样时转换为期望（这里 $x$ 是样本空间上的通用随机变量，不是本书其他地方使用的提示符号）。
在这种情况下，$P$ 是当前正在训练的模型的生成分布（即不是参考模型）。
然后，KL 散度的计算变为以下：

$$
\mathcal{D}_{\text{KL}}(P \,||\, Q) = \mathbb{E}_{x \sim P} \left[ \log P(x) - \log Q(x) \right].
$$ {#eq:kl_expectation}

这种基于样本的形式实现起来简单得多，特别是在处理语言模型训练中频繁使用的对数概率时。

```python
# 第 1 步：generate() 自回归地逐 token 采样完整序列
generated_tokens = model.generate(inputs)

# 第 2 步：forward() 在序列上运行单次传递以获取每 token logit（无采样）
logits       = model.forward(generated_tokens[:, :-1]).logits
ref_logits   = ref_model.forward(generated_tokens[:, :-1]).logits

# 第 3 步：将 logit 转换为对数概率
logprobs     = F.log_softmax(logits, dim=-1)
ref_logprobs = F.log_softmax(ref_logits, dim=-1)

# 第 4 步：收集每个模型分配给实际生成的 token 的概率
token_logprobs     = logprobs.gather(-1, generated_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
ref_token_logprobs = ref_logprobs.gather(-1, generated_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

# 第 5 步：求和以获得序列级对数概率；它们的差异近似 KL
seq_logprob     = token_logprobs.sum(dim=-1)
ref_seq_logprob = ref_token_logprobs.sum(dim=-1)

kl_approx = seq_logprob - ref_seq_logprob
kl_full   = F.kl_div(ref_logprobs, logprobs, reduction='batchmean')
```

一些示例实现包括 [TRL](https://github.com/huggingface/trl/blob/5c21de30ae210e4251ead85517ba8dfe3f210e81/trl/trainer/ppo_trainer.py#L1150) 和 [Hamish Ivison 的 Jax 代码](https://github.com/hamishivi/EasyLM/blob/main/EasyLM/models/llama/llama_train_ppo.py#L278)。

## 隐式正则化

本章的其他部分描述了*显式*正则化：实践者故意添加到训练目标中的 KL 惩罚、预训练梯度和边际损失。
越来越多的经验研究表明，基于 RL 的后训练也提供了*隐式*正则化——一种内置的对记忆化和灾难性遗忘的抵抗力，这种抵抗力来自于在线策略优化本身的结构。
这是由于损失更新的性质，即使没有使用任何控制 RL 训练的显式工具，如 KL 惩罚或重放缓冲区。

### SFT 记忆，RL 泛化

后训练社区面临的一个核心问题是：当在单一任务上训练时，模型是学到了可转移到未见变体的泛化规则，还是记忆了训练分布的表面模式？
Chu 等人 2025 [@chu2025sft] 通过一个受控的经验研究回答了这个问题，该研究直接隔离了后训练方法——SFT 对比 RL——对分布外（OOD）泛化的影响。
答案是明确的：RL 学习可转移的规则，而 SFT 记忆训练数据，并在分布偏移下崩溃。

该研究使用两个内置规则变化的环境来理解权衡：

- **GeneralPoints** 是一个算术纸牌游戏，其中模型接收四张扑克牌，必须将它们的数值与运算符（+、-、*、/）组合以达到目标数字（默认 24）。OOD 测试改变了人头牌的计分方式：训练使用一条规则（J、Q、K 都算作 10），评估使用另一条规则（J=11、Q=12、K=13）。

- **V-IRL** 是一个真实世界的视觉导航任务，其中模型遵循语言指令穿越城市街道，沿途识别地标。OOD 偏移将动作空间从绝对方向（北、东）切换到相对方向（左、右）。

在所有任务变体中，RL 随着训练计算量的扩展持续改善 OOD 性能，而 SFT 持续*降低* OOD 性能，尽管改善了分布内性能。
差异的程度是惊人的：在仅使用语言输入的 V-IRL 上，其中 OOD 偏移是从绝对方向坐标到相对方向坐标，RL 将 OOD 每步准确率从 80.8% 提高到 91.8%，而 SFT 将其从 80.8% 坍缩到 1.3%。
SFT 模型比未能泛化更进一步：它破坏了基础模型已经拥有的空间推理能力，坍缩为从指令短语到绝对方向的查找表。

### 通过做来保留：在线策略数据减轻遗忘

前一节表明在单个任务上 RL 泛化而 SFT 记忆。
Chen 等人 2025 [@chen2025retainingdoingroleonpolicy] 提出了互补的问题：当在多个任务上*顺序*训练时，模型是否保留了它已经知道的内容？
他们发现 RL 在目标任务上实现了可比或更高的收益，同时遗忘的比 SFT 少得多，并将这种优势追溯到两个目标优化的根本差异。

要理解为什么两种方法行为如此不同，我们可以通过 KL 散度的视角来看待它们的目标。
在本节中，我们首先展示两种常见的后训练方法可以映射到 KL 散度的两个方向，然后解释将这些作为损失函数的数值行为如何转化为不同的模型行为。

KL 散度定义为两个分布之间的期望对数比，$\mathbb{E}_{x \sim P}\!\left[\log \frac{P(x)}{Q(x)}\right]$，可以用两个方向上的对数差来写：

- **前向 KL**：$\text{KL}(P \| Q) = \mathbb{E}_{x \sim P}\!\left[\log P(x) - \log Q(x)\right]$
- **反向 KL**：$\text{KL}(Q \| P) = \mathbb{E}_{x \sim Q}\!\left[\log Q(x) - \log P(x)\right]$

其中 $P$ 是目标分布，$Q$ 是我们用参数 $\theta$ 建模的分布。
关键区别是我们从哪个分布采样：前向 KL 从目标（或最优）分布 $P$ 采样，而反向 KL 从我们的策略 $Q$ 采样。
在下面的推导中，$P$ 对应于目标 $\pi_\star$（在分析 SFT 时是训练数据分布，或在分析 RL 时是奖励最优策略），$Q$ 对应于学习到的策略 $\pi_\theta$（我们正在训练的内容）。
SFT 将目标放在首位——$\text{KL}(\pi_\star \| \pi_\theta)$——而 RL 翻转顺序——$\text{KL}(\pi_\theta \| \pi_\star)$——改变我们采样的分布。
样本提供了学习的数据。目标——SFT 或 RL——从所述数据塑造模型。

#### SFT 前向 KL

从前向 KL 的定义开始：

$$
\text{KL}(\pi_\star \| \pi_\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \log \pi_\star(y \mid x) - \log \pi_\theta(y \mid x) \right]
$$

将对数差上的期望拆为两项得到：

$$
= \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \log \pi_\star(y \mid x) \right] - \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \log \pi_\theta(y \mid x) \right]
$$

第一项 $\mathbb{E}\!\left[\log \pi_\star(y \mid x)\right]$ 仅依赖于数据分布并等于负熵 $-H(\pi_\star)$——一个不随 $\theta$ 变化的常数。
第二项 $-\mathbb{E}\!\left[\log \pi_\theta(y \mid x)\right]$ 是数据集上的负对数似然，即标准的 SFT 交叉熵损失 $\mathcal{L}_\text{SFT}(\theta)$。代入：

$$
= \underbrace{-H(\pi_\star)}_\text{const} + \mathcal{L}_\text{SFT}(\theta) \propto \mathcal{L}_\text{SFT}(\theta)
$$ {#eq:sft_forward_kl}

由于熵项关于 $\theta$ 是常数，两个损失共享相同的梯度和相同的最小值——最小化 SFT 损失等价于最小化**前向 KL** 散度 $\text{KL}(\pi_\star \| \pi_\theta)$。

#### RL 反向 KL

让我们从标准的 KL 正则化 RL 目标开始：

$$
\max_\pi \; \mathcal{J}_\text{RL}(\theta) = \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi(\cdot \mid x)} \left[ r(x, y) \right] - \beta \cdot \text{KL}\!\left(\pi(\cdot \mid x) \| \pi_\text{ref}(\cdot \mid x)\right)
$$ {#eq:rl_objective_retaining}

提取 $-\beta$ 将最大化转换为最小化：

$$
= \min_\pi \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi(\cdot \mid x)} \left[ \log \frac{\pi(y \mid x)}{\pi_\text{ref}(y \mid x)} - \frac{1}{\beta} r(x, y) \right]
$$ {#eq:rl_min_form}

引入配分函数 $Z(x) = \sum_y \pi_\text{ref}(y \mid x) \exp\!\left(\frac{1}{\beta} r(x,y)\right)$ 将奖励倾斜的参考归一化为有效分布，并加减 $\log Z(x)$，内部期望变为 KL 散度：

$$
= \min_\pi \; \mathbb{E}_{x \sim \mathcal{D}} \left[ \text{KL}\!\left(\pi(\cdot \mid x) \;\middle\|\; \frac{1}{Z(x)} \pi_\text{ref}(\cdot \mid x) \exp\!\left(\tfrac{1}{\beta} r(x,y)\right) \right) - \log Z(x) \right]
$$ {#eq:rl_kl_form}

由于 $\log Z(x)$ 不依赖于 $\pi$，而 KL 散度非负，当且仅当两个分布相等等于零，因此 KL 在 $\pi$ 等于奖励倾斜分布时最小化为零。
因此，奖励 $r(x,y)$ 下的最优策略为：

$$
\pi_\star(y \mid x) = \frac{1}{Z(x)} \pi_\text{ref}(y \mid x) \exp\!\left(\frac{1}{\beta} r(x,y)\right)
$$ {#eq:optimal_policy_retaining}

现在我们可以直接显示与反向 KL 的联系。展开 $\text{KL}(\pi_\theta \| \pi_\star)$ 并代入 $\log \pi_\star(y \mid x) = \log \pi_\text{ref}(y \mid x) - \log Z(x) + \frac{1}{\beta} r(x, y)$：

$$
\begin{aligned}
\text{KL}(\pi_\theta \| \pi_\star) &= \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)} \left[ \log \pi_\theta(y \mid x) - \log \pi_\star(y \mid x) \right] \\
&= \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)} \left[ \log \pi_\theta(y \mid x) - \log \pi_\text{ref}(y \mid x) + \log Z(x) - \frac{1}{\beta} r(x, y) \right] \\
&= - \frac{1}{\beta} \mathbb{E}_{x,y}\!\left[r(x,y)\right] + \text{KL}\!\left(\pi_\theta(\cdot \mid x) \;\middle\|\; \pi_\text{ref}(\cdot \mid x)\right) + \underbrace{\log Z(x)}_\text{const} \\
&\propto - \frac{1}{\beta} \mathbb{E}_{x,y}\!\left[r(x,y)\right] + \text{KL}\!\left(\pi_\theta(\cdot \mid x) \;\middle\|\; \pi_\text{ref}(\cdot \mid x)\right) \\
&= -\frac{1}{\beta} \mathcal{J}_\text{RL}(\theta)
\end{aligned}
$$

等价地，最大化 RL 目标 $\mathcal{J}_\text{RL}(\theta)$ 等同于最小化**反向 KL** 散度 $\text{KL}(\pi_\theta \| \pi_\star)$。

此推导表明 SFT 和 RL 优化了根本不同的目标：SFT 最小化前向 KL，RL 最小化反向 KL。

![前向 KL（SFT）对比反向 KL（RL）的遗忘动力学。"旧"模式代表先验知识，"新"模式代表目标任务。前向 KL 拉伸策略以覆盖目标并将质量从旧模式拉走（右上），而反向 KL 将新模式移向目标而不扰动旧模式（右下）。来自 Chen 等人 2025，经作者许可。](images/retaining_by_doing_mode_intuition.png){#fig:retaining-mode-intuition}

KL 散度的两个方向引入不同的优化压力。

前向 KL 在目标分布具有质量而模型没有质量的地方惩罚模型，这倾向于鼓励**模式覆盖**——模型广泛地散布概率以覆盖目标的所有主要模式。
原因：前向 KL 中的期望在 $\pi_\star$ 下取，因此它严重惩罚模型未能将概率分配给目标具有质量的区域。

反向 KL 仅在模型实际放置质量的区域惩罚模型，这倾向于鼓励**模式寻求**：模型可以集中在一个高概率模式上，同时忽略其他模式。
这里的期望在 $\pi_\theta$ 下取——模型自己的分布——因此 $\pi_\theta(y \mid x) \approx 0$ 的区域对损失的贡献很小，即使 $\pi_\star$ 在那里分配了相当的质量。
同时，它惩罚模型在目标不存在的区域放置质量。

鉴于这种区别，我们可能朴素地期望 SFT 遗忘的比 RL *少*：模式覆盖的前向 KL 应保持目标所有模式上的质量，保留旧知识，而模式寻求的反向 KL 可能坍缩到单一高奖励模式并放弃其他模式。
然而，相反的情况成立。
这种直觉假设单峰策略，但预训练的 LLM 包含多个模式——对于多峰分布，动力学翻转。

考虑具有两种模式的策略：代表先验知识的"旧"模式和针对目标任务的"新"模式（@fig:retaining-mode-intuition）。
前向 KL（SFT）试图覆盖目标分布的两种模式，这推动策略拉伸并*从*旧模式重新分配概率质量，破坏其形状并导致遗忘。
反向 KL（RL）相反，只需要在某些高奖励区域放置质量，因此它可以将采样的新模式移向目标而不触碰旧模式，使先验知识完整。

RL 的模式寻求行为——反向 KL 的结构性属性——保留了模型先验知识的广度并实现了更好的泛化。

总结：

- **SFT（前向 KL）**：$\text{KL}(\pi_\star \| \pi_\theta)$——样本来自目标 $\pi_\star$，一个固定的人类编写补全数据集。对于每个示例，我们问：我们的模型 $\pi_\theta$ 分配给它的概率是多少？模型从不生成任何东西；它学习模仿。这种模式覆盖压力迫使策略广泛地重新分配质量，可能破坏先验知识。

- **RL（反向 KL）**：$\text{KL}(\pi_\theta \| \pi_\star)$——样本来自我们自己的策略 $\pi_\theta$。对于模型生成的每个补全，我们问：这与奖励最优策略 $\pi_\star$ 有多接近？因为模型仅在其自己的生成上训练，更新保持在它已经放置概率质量的地方——奖励信号告诉它哪些生成需要强化，将概率移向 $\pi_\star$ 而不扰动分布的其余部分。

### RL 剃刀：为什么在线 RL 遗忘更少

前一节表明在线策略采样驱动 RL 对遗忘的抵抗力，并将机制追溯到前向与反向 KL 动力学。
对于任何给定任务，存在许多表现良好的不同策略。
Shenfeld 等人 2026 [@shenfeld2026rls] 提供了 RL 泛化的补充视角，引入了 **RL 剃刀**（RL's Razor）论点，它假定以下内容：

> 在新任务的许多高奖励解决方案中，如 RL 等在线策略方法固有地偏向于在 KL 散度上保持更接近原始策略的解决方案。

![偏向 KL 最小解的减少遗忘。（左）在解决新任务的策略中，RL 收敛到那些在 KL 上与基础模型最接近的策略。（右）这种 KL 偏差在匹配新任务性能的情况下产生比 SFT 更高的先前任务保留。来自 Shenfeld、Pari 和 Agrawal 2026。许可证 CC-BY。](images/rl_razor_motivation.png){#fig:rl-razor-motivation}

作者发现过去任务的遗忘与微调策略从初始模型漂移的距离成正比，如通过 KL 散度测量：

$$
\text{遗忘} \approx f\!\left(\mathbb{E}_{x \sim \tau}\!\left[\text{KL}\!\left(\pi_0(\cdot \mid x) \| \pi(\cdot \mid x)\right)\right]\right)
$$ {#eq:rl_razor_forgetting}

在几种 RL 和 SFT 的训练方式中，作者经验证明遗忘与训练和初始策略之间的 KL 散度强相关（$R^2 = 0.96$），**如使用新任务数据测量的**。
这是令人惊讶的，因为 KL 是在*新任务的*输入分布上测量的，而不是在先前任务的保留数据上，但它仍然预测了在先前任务上的性能下降。
在实践中，这为我们提供了一个强大的工具，可以直接从基础和训练策略之间的漂移来估计遗忘——在我们的新专门数据上测量 KL 距离。

为了确定是什么驱动了 RL 策略中较小的 KL 偏移，作者沿两个轴分解 RL 和 SFT 之间的差异——在线策略与离线数据，以及目标是否包括负梯度（当样本低于奖励基线时存在于 RL，在仅强化正确示范的 SFT 中不存在）以将概率推离错误输出。
值得注意的是，他们发现在线策略与离线数据完全解释了泛化性能的差异，而负梯度没有可辨别的影响。

直观上，在线策略方法采样模型已经分配了不可忽略概率的输出，因此每次更新被约束为保持在当前分布附近。
另一方面，SFT 在固定的外部分布上进行训练，该分布可以与模型当前产生的内容任意远，而每个梯度步骤都向那个遥远的目标拉去，无论模型自己的信念是什么。

## 其他类型的正则化

在后训练文献中，许多突出模型包括其他正则化方法，有助于在其设置中达到领先性能。
这两个例子被包括以描绘一些领先模型如何操纵后训练设置以获得稳定优化，而不是作为应该在每个设置中明确工作的工具。
无数更有创意的解决方案可以工作且将会被发现！

### 预训练梯度

另一种看待正则化的方式是，你可能有一个希望模型保持接近的*数据集*，正如 InstructGPT 中所做的 [@ouyang2022training]，"为了修复在公共 NLP 数据集上的性能退化"。
为实现此，他们修改了 RLHF 的训练目标。
取 @eq:rl_start，我们可以通过从用于 RLHF 的 RL 数据集中采样 RL 策略模型的提示 $x$ 的补全 $y$ 来将其转化为优化的目标函数，得到：
$$
J(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}_{\pi_{\text{RL},\theta}}} \left[ r_{\theta}(y \mid x) - \lambda r_{\text{reg.}} \right]
$$ {#eq:objective_regularization}

然后，我们可以在预训练期间使用的标准自回归下个 token 预测损失上，对从预训练语料库（或另一个数据集）中采样的一组文档添加额外的奖励，用于为更高的概率提供奖励，以保持文本连贯性：

$$
J(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}_{\pi_{\text{RL},\theta}}} \left[ r_{\theta}(y \mid x) - \lambda r_{\text{reg.}} \right] + \gamma \mathbb{E}_{x \sim \mathcal{D}_{\text{pretrain}}} \left[ \log(\pi_{\text{RL},\theta}(x)) \right]
$$ {#eq:objective_pretraining}

最近的工作提出使用负对数似然项来平衡直接偏好优化（DPO）的优化 [@pang2024iterative]。
鉴于 DPO 损失的成对性质，可以对奖励模型训练进行相同的损失修改，约束模型以预测准确的文本。

优化作为对 DPO 的修改如下：
$$\mathcal{L}_{\text{DPO+NLL}} = \mathcal{L}_{\text{DPO}}(c_i^w, y_i^w, c_i^l, y_i^l \mid x_i) + \alpha \mathcal{L}_{\text{NLL}}(c_i^w, y_i^w \mid x_i)
$$ {#eq:dpo_nll}

$$
= -\log \sigma \left( \beta \log \frac{P_\theta(c_i^w, y_i^w \mid x_i)}{P_{\text{ref.}}(c_i^w, y_i^w \mid x_i)} - \beta \log \frac{P_\theta(c_i^l, y_i^l \mid x_i)}{P_{\text{ref.}}(c_i^l, y_i^l \mid x_i)} \right) - \alpha \frac{\log P_\theta(c_i^w, y_i^w \mid x_i)}{|c_i^w| + |y_i^w|},
$$ {#eq:dpo_nll_expanded}

其中 $P_{\theta}$ 是可训练的策略模型，$P_{\text{ref.}}$ 是固定的参考模型（通常是 SFT 检查点），$(c_i^w, y_i^w)$ 和 $(c_i^l, y_i^l)$ 表示提示 $x_i$ 的获胜和失败补全。
第一项是标准的 DPO 逻辑损失：它使用对数似然比的差异 $\log \tfrac{P_{\theta}}{P_{\text{ref.}}}$ 增加获胜和失败之间的边际，而 $\beta$ 控制此偏好信号从参考拉开有多强烈。
第二项是对获胜补全的长度归一化负对数似然惩罚，由 $\alpha$ 加权，有助于在绝对语言建模意义上保持偏好文本的高似然，而不仅仅是比被拒绝样本相对更好。

### 基于边际的正则化

在 RLHF 栈的其他部分，控制优化的定义不太明确。
大多数奖励模型除了标准的对比损失函数外没有正则化。
直接对齐算法通过 $\beta$ 参数不同地处理对 KL 散度的正则化（见[直接对齐章节](https://rlhfbook.com/c/08-direct-alignment)）。

Llama 2 为奖励模型训练提出了边际损失 [@touvron2023llama]：

$$
\mathcal{L}(\theta) = - \log \left( \sigma \left( r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x) - m(y_c, y_r) \right) \right)
$$ {#eq:margin_loss}

其中 $m(y_c, y_r)$ 是两个数据点 $y_c$ 和 $y_r$ 之间的边际，表示两个标注者评分差的数值差。
这通过让标注者对输出在数值尺度上评分或使用量化的排名方法（如 [Likert 量表](https://en.wikipedia.org/wiki/Likert_scale)）实现。

奖励边际已在直接对齐文献中被大量使用，如奖励加权 DPO；奖励感知偏好优化（RPO），将奖励模型分数集成到遵循 DPO 损失的更新规则中 [@adler2024nemotron]；以及 REBEL [@gao2024rebel]，在回归损失公式中具有奖励增量加权。
