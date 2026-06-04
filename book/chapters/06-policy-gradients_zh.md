<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "奖励建模"
prev-url: "05-reward-models"
page-title: 强化学习
search-title: "第 6 章：强化学习"
meta-description: "用于 RLHF 和 LLM 后训练的策略梯度方法，包括 PPO、REINFORCE、RLOO、GRPO 及实现细节。"
next-chapter: "推理与推理时扩展"
next-url: "07-reasoning"
lectures:
  - video: "https://www.youtube.com/watch?v=K_Sj_-1BUMM&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=4"
    label: "第 3 讲：理解 LLM 上的策略梯度算法"
  - video: "https://www.youtube.com/watch?v=i-AIMpZHgeg&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=5"
    label: "第 4 讲：实现 LLM 的 RL 算法"
---

# 强化学习

在 RLHF 过程中，强化学习算法根据奖励模型的反馈缓慢更新模型的权重。
策略（正在训练的模型）对训练集中的提示生成补全，然后奖励模型对它们评分，强化学习优化器基于此信息采取梯度步骤（概述见 @fig:rlhf-overview）。
本章解释了用于从奖励模型对在线策略数据给出的信号中学习的各种算法的数学和权衡。
这些算法运行多个 epoch，通常跨越更大提示集的数千或数百万批次，在每批之间进行梯度更新。

## 强化学习在 RLHF 中的角色

使 RLHF 在语言模型上流行的算法是策略梯度强化学习算法。
这些算法（如近端策略优化 PPO、群组相对策略优化 GRPO 和 REINFORCE）使用最近生成的样本来更新其模型（而不是像 Deep Q-Networks (DQN) 等在 AlphaGo 等流行项目中使用的算法那样将分数存储在重放缓冲区中）。
在本节中，我们将介绍策略梯度算法的基础知识以及它们如何在现代 RLHF 框架中使用。

在机器学习层面上，本节是 RLHF 过程中复杂度最高的主题。
然而，与大多数现代 AI 模型一样，其成功的最大决定因素是作为流程输入提供的数据。

![RLHF 训练循环概述。数据集中的提示被传递给被调优的策略，该策略生成补全。奖励模型对此补全评分，而冻结的初始模型（通常是 RL 之前的指令微调模型）对相同文本计算对数概率以计算 KL 惩罚，防止过度漂移。组合的奖励信号然后驱动对策略参数的强化学习更新。](images/rlhf-overview.png){#fig:rlhf-overview}

当 RLHF 随 ChatGPT 出现时，人们广泛知道他们使用了 PPO 的一种变体，许多初步努力都建立在此基础上。
随着时间推移，多个研究项目展示了 REINFORCE 风格算法的前景 [@ahmadian2024back] [@wang2024helpsteer2p]，因其比 PPO 更简单、无需单独的价值模型（节省内存从而减少所需 GPU 数量）且具有更简单的优势估计（无广义优势估计 GAE，后者是用于降低策略梯度算法方差的计算优势方法）而受到推崇。
更多算法已经出现，包括群组相对策略优化（GRPO），它在推理任务中特别流行，但总体上许多这些算法可以被调优以适应特定任务。
在本章中，我们涵盖核心策略梯度设置和上述三种算法，因为它们在建立经典 RLHF 文献中起着核心作用。

在最简单的形式下，RLHF 的 RL 阶段需要两个模型：一个策略（正在训练的模型）和一个对其输出评分的奖励模型（如上一章所述）。
RL 之前策略的副本用作参考模型来计算 KL 惩罚（此模型被冻结，即不通过自动微分引擎的梯度更新）。
此处涵盖的最复杂算法 PPO 添加了第四个模型——一个学习到的价值函数，用于估计动作中每个 token 有多好，也是在训练期间更新的大语言模型。
本章中的算法主要区别在于它们如何估计称为*优势*的量——模型当前动作（补全）相对于平均值有多好的度量——以及它们如何约束策略更新以使优化数值稳定。
此 RLHF 过程的视觉概览（无价值模型）显示在 @fig:rlhf-overview 中。

关于符号定义，请参阅问题设置章节。

*本章使用强化学习文献中的 $(s, a)$ 符号，其中 $s$ 表示状态，$a$ 表示动作。在语言模型上下文中，你经常会看到使用 $(x, y)$，其中 $x$ 是提示，$y$ 是补全。$(s, a)$ 框架更通用——这些算法是为每个时间步采取动作的序列决策问题设计的。然而，许多 RLHF 实现将整个补全视为单一动作，使得 $(x, y)$ 符号同样有效。*

***RL 速查表：**本章所有核心 RL 损失函数的单页参考可在 [rlhfbook.com/rl-cheatsheet](https://rlhfbook.com/rl-cheatsheet) 获取。*

## 策略梯度算法

本章的核心是致力于理解以下方程形式。
这个方程计算我们正在训练的语言模型 $\pi_\theta$ 的梯度 $\Delta \theta$：

$$\Delta \theta \propto \Psi_t \, \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$ {#eq:policy_gradient_intuition}

这里，方程由两个关键组成部分构成：
1. $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$——参数空间中哪个方向使动作 $a_t$ 更可能。
2. $\Psi_t$——它有多好？一个对结果评分的标量。

当你把这些组合在一起时，是的，通过乘以这些量，你得到策略梯度更新。
一些事情很简单，比如 $\Psi_t > 0$ 更新参数使 $a_t$ 更可能，$\Psi_t < 0$ 更新参数使其更不可能。
策略梯度计算哪些参数对动作有贡献，以及我们是应该使其在未来更可能还是更不可能发生。
本章其余部分深入探讨了不同的实现方法，以及使其适用于 LLM 的特定技巧。

现在，让我们进一步形式化。
强化学习算法设计为最大化跨越状态 $s \in \mathcal{S}$ 和动作 $a \in \mathcal{A}$ 轨迹的未来折扣奖励（更多符号见附录 A 定义）。
智能体的目标，通常称为*回报*，是给定时间 $t$ 处折扣未来奖励的总和（其中 $\gamma\in [0,1]$ 是优先考虑近期奖励的因子）：

$$G_t = R_{t+1} + \gamma R_{t+2} + \cdots = \sum_{k=0}^\infty \gamma^k R_{t+k+1}.$$ {#eq:return_definition}

回报定义也可以估计为：
$$G_{t} = \gamma{G_{t+1}} + R_{t+1}.$$ {#eq:recursive_return}

这个回报是学习价值函数 $V(s)$ 的基础，该函数是给定当前状态的估计未来回报：

$$V(s) = \mathbb{E}\left[G_t \mid S_t = s \right].$$ {#eq:value_function}

所有策略梯度算法优化策略 $\pi_\theta(a\mid s)$ 以最大化期望回报；此目标可以使用诱导的价值函数 $V^{\pi_\theta}(s)$ 来表达。

其中 $d^{\pi_\theta}(s)$ 是策略 $\pi_\theta(a \mid s)$ 诱导的状态-访问分布，我们最大化的目标可以写为：
$$
J(\theta)
\;=\;
\sum_{s} d^{\pi_\theta}(s) V^{\pi_\theta}(s),
$$ {#eq:policy_objective}

在有限 MDP 中，这是所有状态的求和，但在实践中我们从不精确计算它。
相反，我们通过从当前策略采样 rollout 来从数据中估计它。
在 RLHF 中，这通常意味着从数据集中采样提示 $x_i$ 并生成补全 $y_i \sim \pi_\theta(\cdot\mid x_i)$，然后取经验平均，如：

$$
\hat{J}(\theta) = \frac{1}{B}\sum_{i=1}^{B} R(x_i, y_i),
$$ {#eq:empirical_batch_estimate}

或者，在具有每步奖励的 MDP 视角中，

$$
\hat{J}(\theta) = \frac{1}{B}\sum_{i=1}^{B} \sum_{t=0}^{T_i} \gamma^t r_{i,t}.
$$ {#eq:empirical_mdp_estimate}

在实践中，语言模型的 RLHF 设置 $\gamma = 1$（无折扣），因为优化的单位是集体补全，而不是单个 token——此选择在本章后面的 MDP 与 Bandit 部分进一步讨论。

策略梯度算法的核心是计算关于当前策略下有限时间期望回报的梯度。
使用此期望回报 $J$，参数更新可以如下计算，其中 $\alpha$ 是学习率：

$$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$$ {#eq:policy_update}

核心实现细节是如何计算所述梯度。

### 推导策略梯度

我们想要最大化的 RL 目标的另一种表述方式如下：
$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ R(\tau) \right],
$$ {#eq:policy_objective_expectation}

其中 $\tau = (s_0, a_0, s_1, a_1, \ldots)$ 是一条轨迹，$R(\tau) = \sum_{t=0}^\infty r_t$ 是轨迹的总奖励。或者，我们可以将期望写为对所有可能轨迹的积分：
$$
J(\theta) = \int_\tau p_\theta (\tau) R(\tau) d\tau
$$ {#eq:policy_objective_integral}

注意我们可以如下表达轨迹概率，其中 $\pi_\theta(a_t|s_t) p(s_{t+1}|s_t, a_t)$ 是从一个状态和动作转移到下一组状态的转移概率：
$$
p_\theta (\tau) = p(s_0) \prod_{t=0}^\infty \pi_\theta(a_t|s_t) p(s_{t+1}|s_t, a_t),
$$ {#eq:trajectory_probability}

如果我们将目标（@eq:policy_objective_expectation）关于策略参数 $\theta$ 的梯度：
$$
\nabla_\theta J(\theta) = \int_\tau \nabla_\theta p_\theta (\tau) R(\tau) d\tau
$$ {#eq:policy_gradient_integral}

注意我们可以使用[对数导数技巧](https://andrewcharlesjones.github.io/journal/log-derivative.html)将积分的梯度重写为期望：
$$
\begin{aligned}
\nabla_\theta \log p_\theta(\tau) &= \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} &\text{（来自链式法则）} \\
\implies \nabla_\theta p_\theta(\tau) &= p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) &\text{（重排）}
\end{aligned}
$$ {#eq:log_chain_rule}

使用这个对数导数技巧：
$$
\begin{aligned}
\nabla_\theta J(\theta) &= \int_\tau \nabla_\theta p_\theta (\tau) R(\tau) d\tau \\
&= \int_\tau p_\theta (\tau) \nabla_\theta \log p_\theta (\tau) R(\tau) d\tau \\
&= \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log p_\theta (\tau) R(\tau) \right]
\end{aligned}
$$ {#eq:policy_gradient_expectation}

其中最后一步使用轨迹分布 $p_\theta(\tau)$ 下期望的定义：对于任何函数 $f$，$\mathbb{E}_{\tau \sim p_\theta}[f(\tau)] = \int_\tau f(\tau)\,p_\theta(\tau)\,d\tau$（或离散情况下的求和）。
将其写为期望是有用的，因为我们可以用蒙特卡洛 rollout 近似它，例如 $\frac{1}{B}\sum_{i=1}^{B} f(\tau_i)$ 对于轨迹 $\tau_i \sim \pi_\theta$。

回到推导，展开轨迹的对数概率：

$$
\log p_\theta (\tau) = \log p(s_0) + \sum_{t=0}^\infty \log \pi_\theta(a_t|s_t) + \sum_{t=0}^\infty \log p(s_{t+1}|s_t, a_t)
$$ {#eq:trajectory_log_prob}

现在，如果我们取上述的梯度，我们得到：

- $\nabla_\theta \log p(s_0) = 0$（初始状态不依赖于 $\theta$）
- $\nabla_\theta \log p(s_{t+1}|s_t, a_t) = 0$（环境转移动力学不依赖于 $\theta$）
- 只有 $\nabla_\theta \log \pi_\theta(a_t|s_t)$ 保留

因此，轨迹对数概率的梯度简化为：
$$
\nabla_\theta \log p_\theta (\tau) = \sum_{t=0}^\infty \nabla_\theta \log \pi_\theta(a_t|s_t)
$$ {#eq:trajectory_log_grad}

达到这个方程是实现中的关键点。
在这里，我们已经走得足够远，可以看到轨迹分布的梯度简化为语言模型策略概率的梯度之和（即我们正在训练的模型给出的 token 概率）。
在实践中，这产生了策略梯度方程的常见形式。
它们最终看起来像是损失中的对数概率之和，然后我们通过自动微分计算梯度。
一个你会反复看到的简短片段大致如下：

```python
seq_log_probs = (token_log_probs * completion_mask).sum(dim=-1)
loss = -(seq_log_probs * advantages).mean()
loss.backward()
```

你会在本章中看到这个模式。现在，回到正式的策略梯度数学。

将其代入 @eq:policy_gradient_expectation，我们得到：
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^\infty \nabla_\theta \log \pi_\theta(a_t|s_t) R(\tau) \right]
$$ {#eq:policy_gradient_returns}

人们经常使用策略梯度的更一般公式：
$$
g = \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^\infty \nabla_\theta \log \pi_\theta(a_t|s_t) \Psi_t \right]
$$ {#eq:general_gradient}

其中 $\Psi_t$ 可以是以下内容（奖励通常也可以通过 $\gamma$ 进行折扣），这一分类法采用自 Schulman 等人 2015 [@schulman2015high]：

1. $R(\tau) = \sum_{t=0}^{\infty} r_t$：轨迹的总奖励。
2. $\sum_{t'=t}^{\infty} r_{t'}$：动作 $a_t$ 之后的奖励，也被描述为回报 $G$。
3. $\sum_{t'=t}^{\infty} r_{t'} - b(s_t)$：前述公式的基线版本。
4. $Q^{\pi}(s_t, a_t)$：状态-动作价值函数。
5. $A^{\pi}(s_t, a_t)$：优势函数，如果能够准确计算，会产生最低的理论方差。
6. $r_t + \gamma V^{\pi}(s_{t+1}) - V^{\pi}(s_t)$：时间差分（TD）残差。

*基线*是用于减少策略更新方差的值（下面会详细讨论）。

对于语言模型，这些概念中的一些不太适用。
例如，对于确定性策略 $\pi$，状态价值为 $V^{\pi}(s_t) = Q^{\pi}(s_t, \pi(s_t))$（对于最优价值函数，有 $V^*(s_t)=\max_{a_t} Q^*(s_t,a_t)$）。对于随机策略，类似的恒等式为 $V^{\pi}(s_t) = \mathbb{E}_{a_t \sim \pi(\cdot\mid s_t)}\!\left[Q^{\pi}(s_t,a_t)\right]$。
Bellman 方程将 Q 与 V 相关联：一般来说 $Q^\pi(s_t,a_t) = \mathbb{E}\!\left[r_t + \gamma V^\pi(s_{t+1}) \mid s_t, a_t\right]$，但对于状态转移确定性的语言模型，这简化为 $Q(s_t,a_t) = r_t + \gamma V(s_{t+1})$。
优势函数衡量动作 $a_t$ 比平均值好多少：

$$A(s_t,a_t) = Q(s_t,a_t) - V(s_t) = r_t + \gamma V(s_{t+1}) - V(s_t)$$ {#eq:advantage_trick}

最后形式正是时间差分（TD）残差（上述第 6 项）——RL 中的一个基本量，衡量价值函数预测与实际发生之间的差距，驱动价值函数更新朝向更准确的估计。在实践中，学习到的价值函数 $\hat{V}$ 用于通过此 TD 误差估计优势。

### 普通策略梯度

普通策略梯度实现通过关于策略参数微分来优化上述 $J(\theta)$ 表达式。
相对于总体回报的简单版本是：

$$\nabla_\theta J(\theta) = \mathbb{E}_\tau \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) G_t \right]$$ {#eq:vanilla_policy_gradient}

普通策略梯度算法的一个常见问题是梯度更新的高方差，这可以通过多种方式缓解。
高方差来自于通过从通常易受噪声影响的环境中的少量 rollout 集估计回报 $G$ 来计算梯度更新（例如从温度 $>0$ 的语言模型生成的随机性质）。
在稀疏奖励的领域中，回报估计的方差更高，因为更多样本是 0 或 1，而不是紧密聚集的。
为了缓解这一点，使用各种技术来归一化价值估计，称为*基线*。
基线通过多种方式实现这一点，有效地通过状态相对于下游动作的价值进行归一化（例如在优势的情况下，它是 Q 值与价值之间的差异）。
最简单的基线是批次奖励的平均值或移动平均值。
即使是这些与动作无关的基线也可以在不改变期望梯度的情况下减少方差，因为 $\mathbb{E}_{a \sim \pi(a|s)}\!\left[b(s) \nabla_\theta \log \pi_\theta(a|s)\right] = 0$ 对于任何状态依赖的 $b(s)$，显著改善了学习信号。

本章讨论的许多策略梯度算法都建立在策略梯度的优势公式之上：

$$\nabla_\theta J(\theta) = \mathbb{E}_\tau \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) A^{\pi_\theta}(s_t, a_t) \right]$$ {#eq:advantage_policy_gradient}

### REINFORCE

REINFORCE 算法很可能是一个逆向首字母缩略词，但它所代表的算法组件与现代强化学习算法非常相关。
定义在开创性论文 *Simple statistical gradient-following algorithms for connectionist reinforcement learning* [@williams1992simple]中：

> 名称是"REward Increment = Nonnegative Factor X Offset Reinforcement X Characteristic Eligibility"的首字母缩略词。

这个的三个组成部分是如何进行*奖励增量*，也就是策略梯度步骤。
它有三个更新规则部分：

1. 非负因子：这是学习率（步长），必须为正数，例如下面的 $\alpha$。
2. 偏移强化：这是奖励的基线 $b$ 或其他归一化因子，以改善稳定性。
3. 特征资格：这将标量奖励信号归因于产生动作的参数。Williams 将此资格项记为 $e$（非指数函数）。在现代策略梯度符号中，它对应于 $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$。

因此，形式看起来非常熟悉：

$$ \Delta_\theta = \alpha(r - b)e $$ {#eq:REINFORCE_BASIC}

使用更现代的符号和广义回报 $G$，REINFORCE 操作符表现为：

$$
\nabla_{\theta}\,J(\theta)
\;=\;
\mathbb{E}_{\tau \sim \pi_{\theta}}\!\left[
    \sum_{t=0}^{T}
    \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t)\,(G_t - b(s_t))
\right],
$$ {#eq:REINFORCE_with_baseline}

这里，值 $G_t - b(s_t)$ 是策略在当前状态下的*优势*，因此我们可以用优势 $A$ 重新表述策略梯度，形式如下：

$$
\nabla_{\theta}\,J(\theta)
\;=\;
\mathbb{E}_{\tau \sim \pi_{\theta}}\!\left[
    \sum_{t=0}^{T}
    \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t)\,A_t
\right],
$$ {#eq:REINFORCE_with_advantage}

REINFORCE 是普通策略梯度的一种具体实现，使用梯度的蒙特卡洛估计器。

![语言模型的基本 REINFORCE 架构。成形奖励将奖励模型分数与来自参考模型的 KL 惩罚相结合。我们在本章中以此结构为基础。](images/reinforce_tikz.png){#fig:reinforce-arch}

### REINFORCE Leave One Out (RLOO)

REINFORCE Leave One Out 相对于标准 REINFORCE 的核心实现细节是，它取批次中*其它*样本的平均奖励来计算基线——而不是对批次中所有奖励取平均 [@huang2024putting], [@ahmadian2024back], [@kool2019buy]。
通过从当前样本自己的基线中排除其奖励，RLOO 基线独立于被评估的动作，这保持了梯度估计器的完全无偏性。

关键的是，这仅在每个状态（提示）生成多个轨迹（补全）时才有效，这在微调语言模型与 RL 的多个领域中都是常见做法。

具体来说，对于 REINFORCE Leave-One-Out（RLOO）基线，给定 $K$ 个采样的轨迹（以提示为条件的采取的动作）$a_1, \dots, a_K$，对给定提示 $s$，我们明确定义基线为以下*每个提示*：

$$
b(s, a_k) = \frac{1}{K-1}\sum_{i=1, i\neq k}^{K} R(s, a_i),
$$ {#eq:RLOO_baseline}

导致优势为：

$$
A(s, a_k) = R(s, a_k) - b(s, a_k).
$$ {#eq:RLOO_advantage}

等价地，这可以表达为：

$$
A(s, a_k) = \frac{K}{K - 1}\left(R(s, a_k) - \frac{1}{K}\sum_{i=1}^{K} R(s, a_i)\right).
$$ {#eq:RLOO_advantage_alt}

这是一个简单的、低方差的*每提示*优势估计，与群组相对策略优化（GRPO）（在近端策略优化 PPO 之后简要讨论）中使用的群组相对优势密切相关。
在实践中，GRPO 风格的训练主要区别在于它如何应用 KL 正则化器（作为显式损失项而非折叠到奖励中）以及是否使用 PPO 风格的比率裁剪。
具体而言，经典的 GRPO 实现在损失级别应用 KL 惩罚，而 RLOO 或传统策略梯度的推导将 KL 惩罚应用于奖励本身。
随着从 RLHF 过渡到推理和可验证奖励的强化学习（RLVR），KL 惩罚的普遍性总体上有所下降，许多 RLHF 代码的推理改编完全将其关闭。
尽管如此，RLOO 的优势可以与 PPO 的裁剪结合使用，显示了这些算法有多么相似。

RLOO 和其他不使用价值网络的算法——价值网络是一个额外的模型副本（评论家），为每个 token 预测标量值 $V(s_t)$——在计算损失时将相同的序列级优势（或奖励）分配给每个 token。
使用学习到的价值网络的算法，如 PPO，为每个 token 单独分配不同的值，从在 EOS token 处实现的最终奖励进行折扣。
使用 KL 距离惩罚，RLOO 对补全聚合每个 token 的 KL 并将该标量折叠到序列奖励中，因此结果优势被广播到所有 token。
PPO 在计算 $A_t$ 之前从每 token 奖励中减去每 token KL，提供 token 级别的信用分配。
GRPO 通常保留序列级优势，但向损失添加一个单独的每 token 项，而不是从奖励中减去它。
这些细节和权衡将在本章后面讨论。

![REINFORCE Leave-One-Out（RLOO）架构。每个提示的多个补全提供了留一基线用于优势估计，无需学习价值函数。](images/rloo_tikz.png){#fig:rloo-arch}

### 近端策略优化（PPO）

近端策略优化（PPO）[@schulman2017proximal] 是深度 RL 成功背后（如 OpenAI 的 Five，掌握了 DOTA 2 [@berner2019dota] 和大量研究）的基础算法之一。
PPO 最大化的目标，关于优势和策略概率，如下：

$$J(\theta) = \min\left(\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}A, \text{clip} \left( \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}, 1-\varepsilon, 1+\varepsilon \right) A \right).$$ {#eq:PPO_EQN}

这里，$\pi_\theta(a|s)$ 是当前正在优化的策略，$\pi_{\theta_{\text{old}}}(a|s)$ 是用于收集训练数据的策略（即前一次迭代的策略）。
这两个策略之间的比率源于*重要性采样*，它允许我们重用在旧策略下收集的数据来估计新策略的梯度。

回忆策略梯度的优势公式（@eq:advantage_policy_gradient）我们有：
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) A^{\pi_\theta}(s_t, a_t) \right].$$ {#eq:advantage_policy_gradient_recall}

此期望是对从 $\pi_\theta$ 采样的轨迹取的，但在实践中，我们希望在一个批次的数据上采取多个梯度步骤，这些数据是从固定策略 $\pi_{\theta_{\text{old}}}$ 收集的。
为了纠正这种分布不匹配，我们乘以重要性权重 $\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$，它重新加权样本以考虑它们在当前策略下比数据收集策略下更可能或更不可能的程度。
没有约束，优化这种重要性加权的目标可能导致破坏性的大策略更新，当比率偏离 1 很远时。
PPO 通过将比率裁剪到范围 $[1-\varepsilon, 1+\varepsilon]$ 来解决这个问题，确保策略在单次更新中不能变化得太剧烈。

为完整起见，PPO 通常被写为时间步上的*期望*裁剪替代目标：

$$
J(\theta)
=
\mathbb{E}_{t}\left[
\min\left(\rho_t(\theta)A_t,\ \text{clip}(\rho_t(\theta),1-\varepsilon,1+\varepsilon)A_t\right)
\right],
\qquad
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}.
$$ {#eq:PPO_EQN_EXPECTED}

目标通常通过简单添加负号转换为损失函数，使优化器寻求使其尽可能负。

对于语言模型，目标（或损失）按每个 token 计算，这直观上可以基于如何计算整个自回归预测序列的概率——通过概率的乘积——来理解。
从那开始，常见的实现是使用*对数概率*，这使计算在现代语言建模框架中更简单。
在实践中，人们计算 token 对数概率的差异并对其进行指数化以恢复策略比率 $\rho_t$。

$$ J(\theta) = \frac{1}{|a|} \sum_{t=0}^{|a|} \min\left(\frac{\pi_\theta(a_{t}|s_t)}{\pi_{\theta_{\text{old}}}(a_{t}|s_t)}A_{t}, \text{clip} \left( \frac{\pi_\theta(a_{t}|s_t)}{\pi_{\theta_{\text{old}}}(a_{t}|s_t)}, 1-\varepsilon, 1+\varepsilon \right) A_{t} \right).  $$  {#eq:PPO_EQN_EXPANDED}

这是 PPO 的每 token 版本，也适用于其他策略梯度方法，但在本章后面的实现部分进一步探讨。
这里，对动作中 token 数量取平均的项 $\frac{1}{|a|}$ 来自常见的实现实践，但不在损失的正式推导中（见 [@liu2025understanding]）。

![PPO 框架。学习到的价值函数支持广义优势估计（GAE）用于每 token 优势，与裁剪的替代目标一起使用。](images/ppo_tikz.png){#fig:ppo-arch}

在这里，我们将解释此损失函数在给定各种优势和策略比率时触发的不同情况。
在实现层面，PPO 的内部计算涉及两个主要项：1）具有学习到的优势的标准策略梯度，和 2）基于最大步长的裁剪策略梯度。

要理解不同情况如何出现，我们可以将策略比率定义为：

$$\rho(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$$ {#eq:PPO_POL_RATIO}

策略比率是 PPO 及相关算法的核心。
它源自计算策略的梯度，并以非常直观的方式控制参数更新。
对于任何批次的数据，策略比率在该批次的第一个梯度步骤从 1 开始，因为此时 $\pi_{\theta}$ 与 $\pi_{\theta_{\text{old}}}$ 相同。然后，在下一次梯度步骤中，如果该梯度步骤增加了某些具有相关正优势的 token 的似然，策略比率将大于 1；如果是另一种情况，则小于 1。常见做法是在更新 $\pi_{\theta_{\text{old}}}$ 之前，使用策略梯度算法每批次采取 1-4 个梯度步骤。

### 理解 PPO 目标

总体而言，PPO 目标可以通过目标与策略比率的关系图中的两条线来可视化，如 @fig:ppo-obj 所示。
PPO 目标通过改变采样动作的概率来最大化。
数值上，目标通过巧妙使用最小化操作来控制正优势和负优势两种情况，使更新最多被推离策略比率 1 的 epsilon 距离。

在信任区域内，PPO 的操作与其他策略梯度算法相同。
这是有意设计的！信任区域是用于限制 PPO 及其同类算法的最大步长以保持更新稳定性的概念。PPO 算法的核心——裁剪和最小/最大函数——定义了这个区域。目标在其外部变平坦。

"信任区域"的概念来自数值优化文献 [@nocedal2006numerical]，但在深度 RL 中通过信任区域策略优化（TRPO）算法普及，该算法被接受为 PPO 的前身 [@schulman2015trust]。
信任区域是应用完整策略梯度步骤的区域，因为这些更新没有被 PPO 目标的最大/最小操作"裁剪"。

![仮设优势下 PPO 目标的不同区域的可视化。"信任区域"将被描述为策略比率 $\rho$ 在 $1\pm\varepsilon$ 范围内的区域。](images/ppo-viz-4x.png){#fig:ppo-obj}

策略比率和优势可以在几种不同的配置中一起出现。我们将情况分为两组：正优势和负优势。

#### 正优势（$A_t > 0$）

这意味着根据价值函数，采取的动作是有益的，我们希望增加未来采取该动作的似然。现在，让我们看看策略比率 $\rho(\theta)$ 的不同情况：

1. $\rho(\theta) < 1 - \varepsilon$：

    - **解释**：动作在新策略下比旧策略下更不可能
    - **未裁剪项**：$\rho(\theta) A_t$
    - **裁剪项**：$(1 - \varepsilon) A_t$
    - **目标**：$\rho(\theta) A_t$
    - **梯度**：$\nabla_\theta \rho(\theta) A_t \neq 0$
    - **发生什么**：正常策略梯度更新——增加动作的似然

2. $1 - \varepsilon \leq \rho(\theta) \leq 1 + \varepsilon$：

    - **解释**：动作在新策略下几乎与旧策略下同样可能
    - **未裁剪项**：$\rho(\theta) A_t$
    - **裁剪项**：$\rho(\theta) A_t$
    - **目标**：$\rho(\theta) A_t$
    - **梯度**：$\nabla_\theta \rho(\theta) A_t \neq 0$
    - **发生什么**：正常策略梯度更新——增加动作的似然

3. $1 + \varepsilon < \rho(\theta)$：

    - **解释**：动作在新策略下比旧策略下更可能
    - **未裁剪项**：$\rho(\theta) A_t$
    - **裁剪项**：$(1 + \varepsilon) A_t$
    - **目标**：$(1 + \varepsilon) A_t$
    - **梯度**：$\nabla_\theta (1 + \varepsilon) A_t = 0$
    - **发生什么**：无更新——动作在新策略下已经更可能

总结，当优势为正（$A_t>0$）时，我们希望提升动作的概率。因此：

- 我们仅在 $\pi_{\text{new}}(a) \leq (1+\varepsilon) \pi_{\text{old}}(a)$ 的情况下执行梯度步骤。直观上，既然优势是正的，我们希望提升动作的概率，但不要提升太多以至于使其显著更可能。
- 关键是，当 $\pi_{\text{new}}(a) > (1+\varepsilon) \pi_{\text{old}}(a)$ 时，我们不执行任何更新，裁剪目标的梯度为 $0$。直观上，动作在新策略下已经更表达了，所以我们不想过度强化它。

#### 负优势（$A_t < 0$）

这意味着根据价值函数，采取的动作是有害的，我们希望减少未来采取该动作的似然。现在，让我们看看策略比率 $\rho(\theta)$ 的不同情况：

1. $\rho(\theta) < 1 - \varepsilon$：

    - **解释**：动作在新策略下比旧策略下更不可能
    - **未裁剪项**：$\rho(\theta) A_t$
    - **裁剪项**：$(1 - \varepsilon) A_t$
    - **目标**：$(1 - \varepsilon) A_t$
    - **梯度**：$\nabla_\theta (1 - \varepsilon) A_t = 0$
    - **发生什么**：无更新——动作在新策略下已经更不可能

2. $1 - \varepsilon \leq \rho(\theta) \leq 1 + \varepsilon$：

    - **解释**：动作在新策略下几乎与旧策略下同样可能
    - **未裁剪项**：$\rho(\theta) A_t$
    - **裁剪项**：$\rho(\theta) A_t$
    - **目标**：$\rho(\theta) A_t$
    - **梯度**：$\nabla_\theta \rho(\theta) A_t \neq 0$
    - **发生什么**：正常策略梯度更新——减少动作的似然

3. $1 + \varepsilon < \rho(\theta)$：

    - **解释**：动作在新策略下比旧策略下更可能
    - **未裁剪项**：$\rho(\theta) A_t$
    - **裁剪项**：$(1 + \varepsilon) A_t$
    - **目标**：$\rho(\theta) A_t$
    - **梯度**：$\nabla_\theta \rho(\theta) A_t \neq 0$
    - **发生什么**：正常策略梯度更新——减少动作的似然

总结，当优势为负（$A_t < 0$）时，我们希望减少动作的概率。因此：

- 我们仅在 $\pi_{\text{new}}(a) \geq (1-\varepsilon) \pi_{\text{old}}(a)$ 的情况下执行梯度步骤。直观上，既然优势是负的，我们希望减少动作的概率，并按优势比例这样做。
- 关键是，当 $\pi_{\text{new}}(a) < (1-\varepsilon) \pi_{\text{old}}(a)$ 时，我们不执行任何更新，裁剪目标的梯度为 $0$。直观上，动作在新策略下已经不太可能了，所以我们不想过度抑制它。

记住，信任区域内的 PPO 与标准形式的策略梯度大致相同，这一点至关重要。

### 价值函数与 PPO

PPO 中的价值函数是模型的额外副本，用于预测每个 token 的价值。
传统 RL 中 token（或状态）的价值是预测从该时刻起的未来回报，通常带折扣。
PPO 中的这个价值被用作学习的基线，代表了用于 REINFORCE 的简单蒙特卡洛版本的演进（REINFORCE 不需要学习到的价值网络）。
这凸显了 PPO 是如何在优化形式、基线等多个方面对 REINFORCE 和普通策略梯度的演进。
在实践中，使用 PPO 和用于语言模型的其他算法时，这是在预测扣除 KL 惩罚后每个 token 的回报（传统上每 token 损失包括来自奖励的 KL，如上所述）。

有几种不同的方法（或目标）用于学习价值函数。
广义优势估计（GAE）被认为是现代系统中的最先进和经典实现，但它通过计算多步的价值预测误差而带有更多的复杂性——参见本章后面关于 GAE 的部分。
价值函数也可以使用来自用于更新策略的 rollout 的蒙特卡洛估计来学习。
PPO 有两个损失——一个用于学习价值函数，另一个用于使用该价值函数来更新策略。

![价值函数训练使用在线策略 rollout 来计算目标。模型在每个 token 处预测 $V_t$，通过 MSE 对目标回报 $\hat{V}_t$ 进行训练。优势 $A_t = \hat{V}_t - V_t$ 然后加权策略梯度更新。](images/value_fn_training.png){#fig:value_fn_training}

下面展示了一个价值网络损失的简单示例实现。

```python
# 基本 PPO 评论家目标和损失（无 GAE）
#
# B：批次大小
# L：补全长度
# 输入：
#   rewards：(B, L) 后 KL 每 token 奖励；EOS 行包含结果
#   done_mask：(B, L) 在终端 token 处为 1.0（EOS 或如果惩罚的截断），否则 0.0
#   completion_mask：(B, L) 在回答 token 上为 1.0 以监督（忽略提示）
#   values：(B, L) 当前评论家预测 V_theta(s_t)
#       因为价值网络是运行中的更新
#   old_values：(B, L) rollout 时的评论家预测 V_{theta_old}(s_t)
#   gamma：折扣因子，浮点数（通常 LM RLHF 为 1.0）
#   epsilon_v：浮点数值裁剪范围（例如 0.2），类似于 PPO 损失更新本身，可选

B, L = rewards.shape

# 1) 每个 token 的蒙特卡洛回报（在终端处重置）
# 如果启用，应用折扣
returns = torch.zeros_like(rewards)
running = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
for t in reversed(range(L)):
    running = rewards[:, t] + gamma * (1.0 - done_mask[:, t]) * running
    returns[:, t] = running

targets = returns  # y_t = G_t（后 KL）

# 2) PPO 风格价值裁剪（可选）
v_pred = values
v_old  = old_values
v_clip = torch.clamp(v_pred, v_old - epsilon_v, v_old + epsilon_v)

vf_unclipped = 0.5 * (v_pred - targets) ** 2
vf_clipped   = 0.5 * (v_clip - targets) ** 2
vf_loss_tok  = torch.max(vf_unclipped, vf_clipped)

# 3) 掩码到回答 token 并聚合
denom = completion_mask.sum(dim=1).clamp_min(1)
value_loss = ((vf_loss_tok * completion_mask).sum(dim=1) / denom).mean()

# 4) 策略损失的优势（无 GAE）：A_t = G_t - V(s_t)
advantages = (targets - v_pred).detach()

# 价值损失稍后应用，通常与 PG 损失一起，例如
# total_loss = policy_loss + vf_coef * value_loss
```

### 群组相对策略优化（GRPO）

群组相对策略优化（GRPO）在 DeepSeekMath [@shao2024deepseekmath] 中引入，并在其他 DeepSeek 工作中使用，例如 DeepSeek-V3 [@deepseekai2025deepseekv3technicalreport] 和 DeepSeek-R1 [@guo2025deepseek]。
GRPO 可以被视为受 PPO 启发的算法，具有非常相似的替代损失，但它避免了使用原始策略语言模型的另一个副本（或另一个检查点进行初始化）来学习价值函数。
这带来了两个假定的好处：

1. 避免从 LM 骨干学习价值函数的挑战，该领域尚未建立最佳实践。
2. 通过不需要在内存中保留额外的模型权重集来节省内存（从需要当前策略、参考策略和价值函数，减少到仅前两个副本）。

GRPO 通过简化价值估计并为回合中的每个 token 分配相同的值（即在提示的补全中，每个 token 被分配相同的值，而不是标准价值函数中的折扣奖励），通过估计优势或基线来实现这一点。
该估计通过从相同的初始状态/提示（$s$）收集多个补全（$a_i$）和奖励（$r_i$），即蒙特卡洛估计来完成。

正式陈述，GRPO 目标与上述 PPO 目标非常相似。
对于 GRPO，目标（或损失）在给定提示 $s$ 的一组补全 $\{a_1, a_2, ..., a_G\}$ 上累积。
这里，我们展示 GRPO 目标：

$$J(\theta) = \frac{1}{G}\sum_{i=1}^G \left(\min\left(\frac{\pi_\theta(a_i|s)}{\pi_{\theta_{\text{old}}}(a_i|s)}A_i, \text{clip} \left( \frac{\pi_\theta(a_i|s)}{\pi_{\theta_{\text{old}}}(a_i|s)}, 1-\varepsilon, 1+\varepsilon \right) A_i \right) - \beta \mathcal{D}_{\text{KL}}(\pi_\theta||\pi_{\text{ref}})\right).$$ {#eq:GRPO}

注意，相对于 PPO，GRPO 的标准实现在损失中包含了 KL 距离。
如上所述，我们可以将其展开为每 token 计算：

$$\begin{aligned}
J(\theta) = \frac{1}{G}\sum_{i=1}^G  \frac{1}{|a_i|} \sum_{t=1}^{|a_i|} \Bigg( &\min\!\left(\frac{\pi_\theta(a_{i,t}|s_{i})}{\pi_{\theta_{\text{old}}}(a_{i,t}|s_{i})}A_{i,t},\; \text{clip} \left( \frac{\pi_\theta(a_{i,t}|s_{i})}{\pi_{\theta_{\text{old}}}(a_{i,t}|s_{i})}, 1-\varepsilon, 1+\varepsilon \right) A_{i,t} \right) \\
&- \beta \mathcal{D}_{\text{KL}}\!\left(\pi_\theta(\cdot|s_{i})\|\pi_{\text{ref}}(\cdot|s_{i})\right) \Bigg)
\end{aligned}$$ {#eq:GRPO_token}

补全索引 $i$ 的优势计算：

$$A_i = \frac{r_i - \text{mean}({r_1, r_2, \cdots, r_G})}{\text{std}({r_1, r_2, \cdots, r_G})}.$$ {#eq:GRPO_ADV}

![GRPO 架构。优势相对于组均值和标准差进行归一化。KL 惩罚直接在损失中应用，而不是塑造奖励。](images/grpo_tikz.png){#fig:grpo-arch}

直观上，GRPO 更新是比较批次内对单个问题的多个答案。
模型学会变得更像标记为正确的答案，更不像其他答案。
这是一种非常简单的计算优势的方法，优势是衡量特定动作在给定状态下比平均值好多少的度量。
相对于 PPO、REINFORCE 和广泛使用奖励模型评分（相对于输出奖励）进行的 RLHF，GRPO 通常以每个提示远高的样本数运行，因为优势完全关于补全相对于其来自同一提示的同行的相对价值。
这里，当前策略对给定提示生成多个回答，并且组级 GRPO 优势估计被给予有价值的上下文。
PPO 和普通策略梯度算法设计为准确估计每个补全的奖励（事实上，在某些情况下，更多的补全对改进价值估计作用很小）。
GRPO 及其变体特别适合现代语言模型工具，其中对给定提示有多个补全非常自然（特别是当与例如机器人任务中来自固定环境状态的多个动作相比时）。

GRPO 的优势计算在其偏差中存在权衡。
通过标准差进行归一化奖励了一个批次中答案正确性变化较小的问题。
对于几乎全部正确或全部错误答案的问题，标准差会更低，优势会更高。
Liu 等人 2025 [@liu2025understanding] 建议在给定这种偏差的情况下去除标准差项，但这是以减少权重于具有少数正确答案的完全错误问题为代价的，这些问题可能被视为模型的有价值学习信号。
那些高方差提示可能正是最难的情况，其中只有少数采样的补全找到正确答案并提供强大的训练信号。

@eq:GRPO_ADV 是使用结果监督时 GRPO 的实现（标准奖励模型或单一可验证奖励），而使用过程监督时需要不同的实现。
在这种情况下，GRPO 将优势计算为后续推理步骤的归一化奖励之和。

最后，GRPO 的优势估计也可以在不使用 PPO 裁剪的情况下应用于更普通的策略梯度版本（例如 REINFORCE），但这不是经典形式。
作为这些算法如何交织在一起的例子，我们可以展示 GRPO 的一个变体 Dr. GRPO [@liu2025understanding] 的优势估计等价于 RLOO 估计（使用其他样本的平均奖励作为基线），最多相差一个常数缩放因子（由于实现细节中归一化优势，这通常不重要）。
Dr. GRPO 从 @eq:GRPO_ADV 中移除了标准差归一化项——注意这也将优势*放大*，等效于在答案分数有方差的样本上增加 GRPO 学习率。
这解决了对具有低奖励方差问题的偏差——即几乎全部答案都正确或错误——但如果从只有一个样本得到正确答案的问题中学习是重要的，这可能会带来潜在代价。
Dr. GRPO 对大小为 $G$ 的组内补全 $i$ 的优势定义为：

$$ \tilde{A}_i = r_i - \text{mean}({r_1, r_2, \cdots, r_G}) = r_i - \frac{1}{G}\sum_{j=1}^G r_j $$ {#eq:DrGRPO_ADV}

在相同符号中，我们可以回忆 RLOO 优势估计为：

$$ A_i^\text{RLOO} = r_i - \frac{1}{G-1}\sum_{j=1, i\neq j}^G r_j $$ {#eq:RLOO_ADV_AGAIN}

因此，如果我们将 Dr. GRPO 优势定义乘以 $\frac{G}{G-1}$，我们可以看到缩放等价：

$$
\begin{aligned}
\frac{G}{G-1} \tilde{A}_i &= \frac{G}{G-1} \left( r_i - \frac{1}{G}\sum_{j=1}^G r_j \right) \\
&= A_i^{\text{RLOO}}
\end{aligned}
$$ {#eq:RLOO_GRPO_EQUIV}

### 群组序列策略优化（GSPO）

当对从先前策略收集的一批数据采取多个梯度步骤时，需要重要性采样来纠正数据收集策略与当前正在优化的策略之间的分布不匹配。
标准的重要性采样恒等式允许我们使用来自另一个分布的样本来估计一个分布下的期望：

$$
\mathbb{E}_{p}[f(x)] = \mathbb{E}_{q}\left[f(x) \frac{p(x)}{q(x)}\right],
$$ {#eq:IS_identity}

其中 $p$ 是目标分布，$q$ 是采样分布，$\frac{p(x)}{q(x)}$ 是重要性权重。
在策略梯度方法中，$p = \pi_\theta$ 是我们想要优化的当前策略，$q = \pi_{\theta_{\text{old}}}$ 是生成训练数据的策略。
这允许我们重新加权在 $\pi_{\theta_{\text{old}}}$ 下收集的样本来估计 $\pi_\theta$ 的梯度，使得每批 rollout 可以进行多个梯度步骤。

这种分布不匹配在两种常见情况下出现：（1）在单个批次上采取多个梯度步骤，其中 $\pi_\theta$ 在每次更新后从 $\pi_{\theta_{\text{old}}}$ 漂移；（2）在异步训练系统中，推理后端（例如 vLLM）和训练后端（例如 FSDP）由于同步延迟可能具有不同的模型权重。

PPO 和 GRPO 在 token 级别应用重要性采样，并通过裁剪*替代目标*来稳定学习。
然而，这种方法有一个微妙的失败模式：当 token 的重要性比率移出裁剪范围 $[1-\varepsilon, 1+\varepsilon]$ 时，该 token 接收零梯度。
对于罕见但重要的 token——如模型最初分配低概率的关键推理步骤——这种"token 丢弃"可能阻止模型学习更可靠地产生它们。

群组序列策略优化（GSPO）[@zheng2025gspo] 通过在序列级别而非 token 级别计算重要性比率来扩展 GRPO。
该算法的实际动机——及其同类 CISPO，它修改了策略梯度算法的重要性采样计算方式——是每 token 重要性采样比率通常在数值上不稳定。
概念上的动机是，当奖励在序列级别分配时（如大多数 RLHF 和 RLVR 设置中），重要性采样校正应与该粒度匹配。

GSPO 通过计算每个回答的单一重要性权重来解决这个问题。

回忆完整回答的概率自回归地分解：

$$
\pi_\theta(a \mid s) = \prod_{t=1}^{|a|} \pi_\theta(a_t \mid s, a_{<t}).
$$ {#eq:response_factorization}

GSPO 使用几何平均定义长度归一化的序列级重要性比率（以避免长序列的数值问题）：

$$
\rho_i(\theta) = \left( \frac{\pi_\theta(a_i \mid s)}{\pi_{\theta_{\text{old}}}(a_i \mid s)} \right)^{\frac{1}{|a_i|}}.
$$ {#eq:GSPO_ratio}

GSPO 目标镜像 GRPO，但使用此序列级比率：

$$
J_{\text{GSPO}}(\theta) = \mathbb{E}_{s \sim \mathcal{D},\, \{a_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot \mid s)} \left[ \frac{1}{G} \sum_{i=1}^G \min\left( \rho_i(\theta) A_i,\, \text{clip}(\rho_i(\theta), 1-\varepsilon, 1+\varepsilon) A_i \right) \right].
$$ {#eq:GSPO_objective}

GSPO 可以总结为"具有序列级重要性比率的 GRPO"——IS 校正粒度与奖励粒度匹配。

### 裁剪重要性采样策略优化（CISPO）

裁剪重要性采样策略优化（CISPO）[@minimax2025minimax_m1] 采取不同的方法：不是裁剪替代目标，CISPO 裁剪重要性权重本身，同时保留所有 token 的梯度。
目标在裁剪的重要性权重上使用停止梯度，回归到 REINFORCE 风格的公式，而不是 PPO 风格的双边裁剪：

$$
J_{\text{CISPO}}(\theta) = \mathbb{E} \left[ \frac{1}{\sum |a_i|} \sum_{i} \sum_{t} \text{sg}\left( \hat{\rho}_{i,t}(\theta) \right) A_{i,t} \log \pi_\theta(a_{i,t} \mid s, a_{i,<t}) \right],
$$ {#eq:CISPO_objective}

其中 $\text{sg}(\cdot)$ 表示停止梯度，裁剪的重要性比率为：

$$
\hat{\rho}_{i,t}(\theta) = \text{clip}\left( \rho_{i,t}(\theta),\, 1 - \varepsilon_{\text{low}},\, 1 + \varepsilon_{\text{high}} \right).
$$ {#eq:CISPO_ratio}

与 PPO/GRPO 的关键区别微妙但重要：裁剪权重（而非目标）意味着每个 token 仍然接收与其优势成比例的梯度信号——权重只是限制了该信号被重要性比率放大或抑制的程度。

### 比较算法

本章中的每个算法共享相同的核心梯度形状（@eq:policy_gradient_intuition），但在如何估计优势和控制优化方面有所不同：

- **REINFORCE**：最简单的策略梯度实现，使用奖励的蒙特卡洛估计和基于状态的基线来减少方差。
- **RLOO**：每个提示多个样本的 REINFORCE，每个样本的基线是其他的平均奖励（留一）以减少梯度方差。
- **PPO**：添加学习到的价值函数和裁剪的策略比率以获得更准确和稳定的梯度更新。
- **GRPO**：PPO 的简化变体，对每个提示分组多个补全并在组内归一化奖励以计算优势，消除了对价值函数的需求。
- **CISPO**：一种 REINFORCE 风格算法，裁剪重要性采样权重（而不是 PPO/GRPO 中的目标），使用停止梯度以保持稳定性，因此每个 token 接收梯度信号。
- **GSPO**：类似于 GRPO，但按补全长度归一化策略比率，防止长度偏差。
- **DPO**：不是 RL 算法，而是一种完全绕过单独奖励模型的方法，直接从偏好对优化来解决相同的偏好优化问题（见第 8 章）。

| 方法 | IS 粒度 | 裁剪风格 | 优势 |
| :----- | :-----------: | :------------------: | :-------------------: |
| **REINFORCE** | 无 | 无 | 蒙特卡洛基线 |
| **RLOO** | 无 | 无 | 留一 |
| **PPO** | Token | 目标（双边） | 学习到的价值函数 |
| **GRPO** | Token | 目标（双边） | 群组相对 |
| **GSPO** | 序列 | 目标（双边） | 群组相对 |
| **CISPO** | Token | 权重（停止梯度） | 群组相对 |
表：比较策略梯度算法。{#tbl:pg_compare}

## 实现

与最初开发这些算法的原始深度 RL 文献相比，为优化语言模型或其他大型 AI 模型实现 RL 需要许多小的实现细节。
在本节中，我们重点介绍区分流行算法实现的一些关键因素。

还有许多其他小细节进入此训练。
例如，使用语言模型进行 RLHF 时，一个关键步骤是生成文本，然后由奖励模型评分。
在正常情况下，模型应生成序列结束（EOS）token 表示其完成生成，但常见做法是对生成长度设置硬上限以有效利用基础设施。
RLHF 的一种失败模式是模型经常在其答案中被截断，将奖励模型的评分驱动到分布外和不可预测的分数。
解决方法是在 `eos_token` 上*仅*运行奖励模型评分，否则对过长生成分配惩罚。

流行的开源 RLHF 工具在算法之间的实现细节上存在很大差异。
一些未在此处涵盖的决策包括：

- **价值网络初始化**：PPO 和其他类似算法使用的内部学习价值网络可以从相同架构的不同模型或随机选择的权重开始。这可能会对性能产生重大影响。InstructGPT [@ouyang2022training] 中建立的标准（并在 Tülu 3 的 RLVR 工作中重新使用 [@lambert2024t]）是从 RLHF 期间使用的奖励模型初始化价值网络。
- **奖励归一化、奖励白化和/或优势白化**：归一化将所有来自 RM（或环境）的值限制在 0 和 1 之间，有助于学习稳定性。[白化](https://en.wikipedia.org/wiki/Whitening_transformation) 更进一步，通过将奖励或优势估计转换为具有零均值和单位方差，提供更强的稳定性提升。
- **不同的 KL 估计器**：使用复杂的语言模型时，精确计算模型之间的 KL 散度可能很复杂，因此使用多种近似来代替精确计算 [@schulman2016klapprox]。
- **KL 控制器**：PPO 和相关算法的原始实现具有动态控制器，目标特定的 KL 并根据最近测量更改惩罚。大多数现代 RLHF 实现使用静态 KL 惩罚。

### 策略梯度基础

使用优势估计梯度的策略梯度的简单实现，为 PPO 和 GRPO 等高级算法做准备：
```python
pg_loss = -advantages * ratio
```
此处的比率是新策略模型概率相对于参考模型的（每 token）概率比率（通常从对数概率差计算）。

### 损失聚合权衡

使用语言模型实现任何策略梯度算法时的问题是：如何将每个 token 的损失聚合成最终的标量损失？
给定样本 $i$ 在 token $t$ 处的每 token 损失 $\ell_{i,t}$，具有补全长度 $|a_i|$ 和批次大小 $B$，有三种主要策略：

**策略 1：每序列归一化**（标准 GRPO；也在一些 PPO 实现中使用）

$$L = \frac{1}{B} \sum_{i=1}^{B} \frac{1}{|a_i|} \sum_{t=1}^{|a_i|} \ell_{i,t}$$ {#eq:loss_per_sequence}

**策略 2：每 token 归一化**（DAPO [@yu2025dapo]）

$$L = \frac{\sum_{i=1}^{B} \sum_{t=1}^{|a_i|} \ell_{i,t}}{\sum_{i=1}^{B} |a_i|}$$ {#eq:loss_per_token}

**策略 3：固定长度归一化**（Dr. GRPO [@liu2025understanding]）

$$L = \frac{1}{B} \sum_{i=1}^{B} \frac{1}{L_{\max}} \sum_{t=1}^{|a_i|} \ell_{i,t}$$ {#eq:loss_fixed_length}

直观上，每序列归一化（策略 1）似乎是最好的，因为我们关心的是*结果*，而不是单个 token。
然而，这引入了基于序列长度的微妙偏差。
在实践中，最佳策略取决于具体的训练设置。在 RLHF 中，通常偏好具有最佳数值稳定性或最小损失方差的方法。

### 相关：MDP 与 Bandit 框架

损失聚合的选择与如何构建 RL 问题有关。
**MDP（token 级别）**视角将每个 token $a_t$ 视为具有状态 $s_t$ 为运行前缀的动作，使用学习到的价值函数 $V(s_t)$ 计算 token 级优势（例如 GAE）。
**Bandit（序列级别）**视角将整个补全视为具有一个标量奖励 $R$ 的单一动作，计算序列级优势 $A_{\text{seq}}$ 并将其广播到所有 token。
折扣因子 $\gamma$ 在几乎所有 RLHF 实现中设置为 1.0，因为奖励信号对整个回答进行评分，而不是单个 token。

### 异步 RL 系统

策略梯度算法的默认实现是**在线策略**执行，其中智能体（语言模型）采取的动作（生成）在更新模型之前被评分。
策略梯度的理论推导依赖于所有动作完全在线策略，其中模型始终与最新试验/rollout 的结果保持最新。
在实践中，保持精确的在线策略执行会显著减慢训练 [@noukhovitch2024asynchronous]。

常见的解决方案是在单独的 GPU 节点上持续运行推理和训练，使用设计用于高效运行两者的软件。
流行的开源 RL 工具使用分布式进程管理库（如 Ray）在策略梯度学习循环和推理循环之间传递信息，使用高效的推理引擎（如 vLLM）。
在这些设置中，专用于采取 RL 步骤的 GPU 称为"学习器"，专用于从语言模型采样的 GPU 称为"行动者"。
使训练更异步时面临的主要挑战是保持训练稳定和维护学习信号。

### 截断重要性采样

截断重要性采样（TIS）是用于在现代异步 RL 框架中稳定训练的关键工具。
重要性采样是一种校正，重新加权从一种分布抽取的样本以估计另一种分布下的期望。
截断重要性采样 [@ionides2008truncated] 使用 $\min(\rho, C)$（对于某个常数 $C$）限制这些权重，在策略梯度中用小的偏差换取有界方差。

在 LLM RL 系统中，TIS 作为策略梯度损失上的每 token 校正权重应用。TIS 用于学习器-采样器比率已被主要开源 RL 框架（VeRL、TRL、OpenRLHF、SkyRL、OAT 和 Open Instruct，使用 $C = 2$）采用，并且对于长推理轨迹（第 7 章）变得越来越重要。

### 广义优势估计（GAE）

广义优势估计（GAE）是计算策略梯度算法优势的替代方法 [@schulman2015high]，更好地平衡了偏差-方差权衡。
GAE 计算多步优势估计的指数加权平均，其中 $\lambda$ 超参数控制偏差-方差权衡——范围从单步 TD（$\lambda=0$）到完整轨迹回报（$\lambda=1$）；$\lambda=0.95$ 是 LLM 微调的常见默认值。

### 双重正则化

我们在本章中看到了两种类型的正则化。一种内建于像 PPO 这样的算法中，具有步长约束，另一种是相对于优化起点的 KL 散度距离惩罚。
在使用语言模型的实践中，像 PPO 和 GRPO 这样的算法通常每批只运行一个梯度步骤，这意味着 PPO 原生正则化从未被应用（因为裁剪仅在策略大幅变化时在批次内发生）而 KL 距离惩罚占主导地位。

## 建议的实验

`code/policy_gradients/` 中的配套实现专为小型、可观察的 RL 运行设计。

1. **使用 GRPO 运行单词反转任务。**

   ```bash
   cd code/
   uv run python -m policy_gradients.train --config policy_gradients/configs/grpo.yaml
   ```

2. **比较群组相对和单样本估计器。**

   ```bash
   uv run python -m policy_gradients.train --config policy_gradients/configs/reinforce.yaml
   uv run python -m policy_gradients.train --config policy_gradients/configs/rloo.yaml
   uv run python -m policy_gradients.train --config policy_gradients/configs/grpo.yaml
   ```

3. **扫描对比参数。**
   复制配置并改变 `num_rollouts`、`temperature`、`data.size` 和 `format_weight`。

4. **从玩具奖励转向数学。**
   对于 GSM8K 风格的实验，在添加新的在线 RL 环境之前，从 `code/reward_models/train_orm.py` 和 `code/rejection_sampling/` 示例开始。
