<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "塑造模型性格与产品"
prev-url: "17-product"
page-title: "附录 A：定义"
search-title: "附录 A：定义"
meta-description: "RLHF、强化学习、语言模型和后训练术语的定义与背景。"
next-chapter: "超越\"只是风格\""
next-url: "appendix-b-style"
---

# 定义

本附录包含 RLHF 过程中经常使用的所有定义、符号和操作，以及对语言模型的快速概述，这是本书的指导应用。

## 语言建模概述

大多数现代语言模型被训练以自回归方式学习 token（单词、子词或字符）序列的联合概率分布。
自回归简单意味着每个下一个预测依赖于序列中先前的实体。
给定一个 token 序列 $x = (x_1, x_2, \ldots, x_T)$，模型将整个序列的概率分解为条件分布的乘积：

$$P_{\theta}(x) = \prod_{t=1}^{T} P_{\theta}(x_{t} \mid x_{1}, \ldots, x_{t-1}).$$ {#eq:llming}

为了拟合一个准确预测此的模型，目标通常是最大化当前模型预测的训练数据似然。
为此，我们可以最小化负对数似然（NLL）损失：

$$\mathcal{L}_{\text{LM}}(\theta)=-\,\mathbb{E}_{x \sim \mathcal{D}}\left[\sum_{t=1}^{T}\log P_{\theta}\left(x_t \mid x_{<t}\right)\right]. $$ {#eq:nll}

在实践中，使用相对于每个下一个 token 预测的交叉熵损失，通过比较序列中的真实 token 与模型的预测来计算。

语言模型有多种架构，在知识、速度和其他性能特征方面有不同的权衡。
现代 LM，包括 ChatGPT、Claude、Gemini 等，最常使用**仅解码器 Transformer** [@Vaswani2017AttentionIA]。
Transformer 的核心创新是大量利用**自注意力** [@Bahdanau2014NeuralMT] 机制，使模型能够直接关注上下文中的概念并学习复杂的映射。
在本书中，特别是在第 5 章讨论奖励模型时，我们将讨论为 Transformer 添加新头或修改语言建模（LM）头。
LM 头是一个最终的线性投影层，将模型的内部嵌入空间映射到分词器空间（也称为词汇表）。
我们将在本书中看到，语言模型的不同"头"可以应用于微调模型以适应不同的目的——在 RLHF 中，这通常在训练奖励模型时完成，这在第 5 章中突出显示。

## 机器学习

- **Kullback-Leibler（KL）散度（$\mathcal{D}_{\text{KL}}(P || Q)$）**，也称为 KL 散度，是衡量两个概率分布之间差异的度量。
对于在相同概率空间 $\mathcal{X}$ 上定义的离散概率分布 $P$ 和 $Q$，从 $Q$ 到 $P$ 的 KL 距离定义为：

$$ \mathcal{D}_{\text{KL}}(P || Q) = \sum_{x \in \mathcal{X}} P(x) \log \left(\frac{P(x)}{Q(x)}\right) $$ {#eq:def_kl}

## 自然语言处理

- **选定补全（$y_c$）**：被选择或偏好于其他替代项的补全，通常记为 $y_{chosen}$。

- **补全（$y$）**：语言模型对提示生成的输出文本。通常补全记为 $y\mid x$。奖励和其他值通常计算为 $r(y\mid x)$ 或 $P(y\mid x)$。

- **策略（$\pi$）**：可能补全上的概率分布，由 $\theta$ 参数化：$\pi_\theta(y\mid x)$。

- **偏好关系（$\succ$）**：表示一个补全优于另一个的符号，例如 $y_{chosen} \succ y_{rejected}$。例如，奖励模型预测偏好关系的概率 $P(y_c \succ y_r \mid x)$。

- **提示（$x$）**：给语言模型的输入文本，用于生成回答或补全。

- **被拒绝补全（$y_r$）**：在成对设置中不受青睐的补全。

## 强化学习

- **动作（$a$）**：智能体在环境中做出的决策或移动，通常表示为 $a \in A$，其中 $A$ 是可能动作的集合。

- **优势函数（$A$）**：优势函数 $A(s,a)$ 量化了在状态 $s$ 中采取动作 $a$ 相对于平均动作的相对益处。定义为 $A(s,a) = Q(s,a) - V(s)$。

- **折扣因子（$\gamma$）**：标量 $0 \le \gamma < 1$，在回报中对未来奖励进行指数加权递减，权衡即时性与长期收益，并保证无限 horizon 求和收敛。有时不使用折扣，等效于 $\gamma=1$。

- **期望奖励优化**：RL 的主要目标，涉及最大化期望累积奖励：

  $$\max_{\theta} \mathbb{E}_{s \sim \rho_\pi, a \sim \pi_\theta}\left[\sum_{t=0}^{\infty} \gamma^t r_t\right]$$ {#eq:expect_reward_opt}

- **有限 Horizon 奖励（$J(\pi_\theta)$）**：策略 $\pi_\theta$ 的期望有限 horizon 折扣回报定义为：

  $$J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \gamma^t r_t \right]$$ {#eq:finite_horizon_return}

- **在线策略**：在 RLHF 中，特别是在 RL 与直接对齐算法的辩论中，**在线策略**数据的讨论很常见。在 RL 文献中，在线策略意味着数据由智能体的*确切*当前形式生成，但在一般的偏好微调文献中，在线策略被扩展到意味着来自该版本模型的生成。

- **策略（$\pi$）**，在 RLHF 中也称为**策略模型**：在 RL 中，策略是智能体遵循的策略或规则，用于决定在给定状态下采取哪个动作：$\pi(a\mid s)$。

- **Q 函数（$Q$）**：一个估计从在给定状态采取特定动作开始的期望累积奖励的函数：$Q(s,a) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s, a_0 = a\right]$。

- **奖励（$r$）**：一个标量值，指示动作或状态的可取性，通常记为 $r$。

- **状态（$s$）**：环境的当前配置或情况，通常记为 $s \in S$，其中 $S$ 是状态空间。

- **轨迹（$\tau$）**：轨迹 $\tau$ 是智能体经历的状态、动作和奖励的序列：$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T, a_T, r_T)$。

- **价值函数（$V$）**：一个估计从给定状态开始的期望累积奖励的函数：$V(s) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s\right]$。

## RLHF 专用

- **参考模型（$\pi_{\text{ref}}$）**：在 RLHF 中使用的一组保存的参数，其输出用于正则化优化。

## 扩展词汇表

- **思维链（CoT）**：思维链是语言模型的一种特定行为，它们被引导到一种将问题分解为逐步形式的行为。原始版本是通过提示"让我们逐步思考" [@wei2022chain]。

- **蒸馏**：蒸馏是训练 AI 模型的一套通用实践，其中模型在更强模型的输出上进行训练。这是一种已知能制造强大、较小模型的合成数据类型。

- **上下文学习（ICL）**：上下文指的是语言模型上下文窗口内的任何信息。通常，这是添加到提示中的信息。最简单的上下文学习形式是在提示之前添加类似形式的示例。

- **（师生）知识蒸馏**：从特定教师到学生模型的知识蒸馏是上述蒸馏的一种特定类型，也是该术语的起源。它是一种特定的深度学习方法，其中神经网络损失被修改为从教师模型在多个潜在 token/logit 上的对数概率中学习，而不是直接从选定的输出中学习 [@hinton2015distilling]。

- **合成数据**：这是 AI 模型的任何训练数据，是来自另一个 AI 系统的输出。这可以是从模型开放式提示生成的文本到模型重写现有内容的任何内容。
