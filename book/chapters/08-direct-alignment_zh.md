<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "推理与推理时扩展"
prev-url: "07-reasoning"
page-title: 直接对齐算法
search-title: "第 8 章：直接对齐算法"
meta-description: "直接对齐算法，如 DPO，无需显式奖励模型或 RL 循环即可优化偏好目标。"
next-chapter: "拒绝采样"
next-url: "09-rejection-sampling"
---

# 直接对齐算法

直接对齐算法（Direct Alignment Algorithms, DAAs）允许人们更新模型以解决相同的 RLHF 目标，而无需训练中间奖励模型或使用强化学习优化器。
DAA 解决的是我们一直在研究的相同的偏好学习问题（实际上使用相同的数据！），以使语言模型更对齐、更智能、更易用。
无需奖励模型和在线优化使 DAA 实现起来简单得多，减少了训练期间花费的计算，并使实验更容易。
本章详述了推导这些算法所完成的复杂数学，然后展示了有时繁琐的推导所带来的简单实现。

最突出的 DAA，也是催化了整个对齐语言模型的学术运动的，是直接偏好优化（Direct Preference Optimization, DPO）[@rafailov2024direct]。
DPO 的核心是使用梯度上升来解决相同的约束 RLHF 目标（见第 3 章）：

$$ \max_{\pi} \mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)} \left[r_\theta(x, y)\right] - \beta \mathcal{D}_{\text{KL}}\left(\pi(y|x) \| \pi_{\text{ref}}(y|x)\right)$$ {#eq:review_rlhf}

自 2023 年 5 月发布以来，在社区弄清楚使用 DPO 的正确数据和超参数（特别是令人惊讶的低学习率）的短暂延迟之后，许多流行模型使用了 DPO 或其变体，从 Zephyr-$\beta$ 在 2023 年 10 月启动它 [@tunstall2023zephyr]，到 Llama 3 Instruct [@dubey2024llama]、Tülu 2 [@ivison2023camels] 和 3 [@lambert2024t]、Nemotron 4 340B [@adler2024nemotron] 等。
技术上说，序列似然校准（SLiC-HF）是第一个发布的现代直接对齐算法 [@zhao2023slic]，但由于多种因素（解开研究方法采用的原因总是一项棘手的任务）而没有流行起来。

DPO 和 DAA 最有影响力的部分是降低了语言模型后训练实验的准入门槛——它使用更少的计算，更容易从零实现，也更容易在玩具和生产示例上运行。

*在本章中，我们使用 $x$ 表示提示，$y$ 表示补全。这种表示法在语言模型文献中很常见，其中方法操作完整的提示-补全对而非单个 token。*

## 直接偏好优化

这里我们解释 DPO 工作原理的直觉，并完整重新推导核心方程。

### DPO 的工作原理

DPO 表面上直接优化策略以解决 RLHF 目标。
其损失函数（我们将在下面的推导中重新讨论）比较学习到的策略对选中和被拒绝补全的概率相对于参考模型偏移了多少。
从 Bradley-Terry 奖励模型推导的损失函数如下：

$$ \mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_c, y_r) \sim \mathcal{D}}\left[ \log \sigma\left( \beta \log \frac{\pi_{\theta}(y_c \mid x)}{\pi_{\text{ref}}(y_c \mid x)} - \beta \log \frac{\pi_{\theta}(y_r \mid x)}{\pi_{\text{ref}}(y_r \mid x)} \right) \right] $$ {#eq:dpo_core}

在 sigmoid 内部，第一项 $\beta \log \frac{\pi_{\theta}(y_c | x)}{\pi_{\text{ref}}(y_c | x)}$ 衡量策略相对于参考模型增加了多少*选中*补全的概率，第二项对*被拒绝*补全做同样的衡量。当选中的偏移幅超过被拒绝的偏移时——即当策略学会偏好正确的回答时——损失下降。

整章中，$\beta$ 是平衡奖励优化与最终模型和初始参考之间 KL 散度的超参数（即平衡过度优化，正确使用 DPO 时的关键超参数）。
这依赖于用于 DPO 训练的隐式奖励，替代使用外部奖励模型，它是对数概率比：

$$r(x, y) = \beta  \log \frac{\pi_r(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$ {#eq:dpo_reward}

其中 $\pi_r(y \mid x)$ 是我们正在求解的精确最优奖励策略。
这来自相对于最优策略推导 Bradley-Terry 奖励（如第 5 章 Bradley-Terry 模型部分中的 @eq:dpo_opt_policy 所示）。
本质上，正如 DPO 论文所述，这种重新参数化给了我们"人类偏好数据关于最优策略而非奖励模型的概率"——意味着我们可以完全绕过学习显式奖励模型。

让我们考虑优化器必须减少的 @eq:dpo_core 中所示的损失。
这里，当选中的回答的对数比大于被拒绝的回答的对数比（由参考模型归一化）时，损失会更低。
在实践中，这是模型在数据中呈现的 token 序列上的对数概率之和。
因此，DPO 在增加选中和被拒绝回答之间的相对对数概率差距。

使用 @eq:dpo_reward 中的奖励，我们可以写出损失的梯度以进一步解释正在发生的事情：

$$\nabla_{\theta}\mathcal{L}_{\text{DPO}}(\pi_{\theta}; \pi_{\text{ref}}) = -\beta \mathbb{E}_{(x, y_c, y_r)\sim \mathcal{D}}\left[ w \cdot \left(\nabla_{\theta}\log \pi(y_c \mid x) - \nabla_{\theta}\log \pi(y_r \mid x)\right) \right]$$ {#eq:dpo_gradient}

其中 $w = \sigma\!\left(r_{\theta}(x, y_r) - r_{\theta}(x, y_c)\right)$。

这里，梯度通过以下方式解决上述目标：

- sigmoid 函数中的第一项 $\sigma(\cdot)$ 创建一个从 0 到 1 的参数更新权重，当奖励估计错误时该权重更高。当被拒绝的样本比选中的更被偏好时，权重更新应该更大！
- 其次，内括号中的项 $[\cdot]$ 增加选中回答 $y_c$ 的似然，减少被拒绝回答 $y_r$ 的似然。
- 这些项由 $\beta$ 加权，它控制更新如何在正确排序补全和 KL 散度之间平衡。

核心直觉是 DPO 在拟合一个隐式奖励模型，其对应的最优策略可以以封闭形式提取（@eq:dpo_opt_policy，多亏梯度下降和我们的 ML 工具）。
因为 DPO 损失是直接可微的，计算精确梯度是直接的，而不需要通过训练奖励模型和采样补全来评分来估计。
经常被误解的是，DPO 在其核心是在学习一个奖励模型，因此论文的副标题是 *Your Language Model is Secretly a Reward Model*（你的语言模型秘密地是一个奖励模型）。
很容易将其与 DPO 目标直接训练策略混淆，因此研究下面的推导有助于完全理解。

通过隐式奖励模型学习，DPO 正在生成给定数据集中的数据和目标中特定 KL 约束 $\beta$ 下 RLHF 目标的最优解。
这里，DPO 针对特定的 KL 散度求解精确策略，因为生成不像策略梯度算法中那样是在线的——这是与偏好微调的 RL 方法的核心区别。
在很多方面，这使得 $\beta$ 值相对于在线 RL 方法更容易使用 DPO 调节，但关键是，最优值直觉上取决于被训练的模型和训练它的数据。

在由许多成对补全 $y_{chosen} \succ y_{rejected}$ 组成的每个偏好数据批次上，DPO 直接向最优解采取梯度步骤。
它比策略梯度方法简单得多。

![当 DPO 首次发布时，它在研究社区引发了关于如何最好地进行 RLHF 和偏好学习的激烈辩论。这个 meme 很好地捕捉了这种情绪，辩论经常让人感觉是被强加的和过度的，但许多人——无论是刚入门的还是在顶级实验室的——都从 DPO 中获得了巨大的收益。DPO 简洁性 meme，来源 Tom Goldstein。](images/dpo_meme.jpeg){#fig:dpo-meme}

### DPO 推导

DPO 推导分为两个主要部分。
首先，作者展示了最优解决本书中贯穿使用的 RLHF 目标的策略形式。
接下来，他们展示了如何从成对偏好数据（即 Bradley-Terry 模型）达到该解。

#### 推导最优 RLHF 解

首先，我们应该再次考虑 RLHF 优化目标，这里表示我们希望最大化这个量：

$$ \max_{\pi} \mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)} \left[r_\theta(x, y)\right] - \beta \mathcal{D}_{\text{KL}}\left(\pi(y|x) \| \pi_{\text{ref}}(y|x)\right)$$ {#eq:rlhf_opt_eq_repeat}

这里，双重期望仅适用于采样以计算期望奖励，因为 KL 项仍然是一个解析表达式。
首先，让我们展开 KL 散度的定义。回忆 $\mathcal{D}_{\text{KL}}(\pi \| \pi_{\text{ref}}) = \mathbb{E}_{y \sim \pi}\left[\log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\right]$，其中和中的 $\pi(y|x)$ 加权成为采样分布。
由于两个项现在共享相同的对 $y \sim \pi(y|x)$ 的期望，我们可以将它们组合：

$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[r(x,y)-\beta\log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\right] $$ {#eq:dpo_deriv_1}

接下来，将负号从括号中的差中提出。为此，将其分成两项：

$$ = \max_{\pi}\left(\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[r(x,y)\right] - \beta\,\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[\log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\right]\right) $$ {#eq:dpo_deriv_2}

然后，乘以 $-1$ 将最大化转换为最小化：

$$ = \min_{\pi}\left(-\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[r(x,y)\right] + \beta\,\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[\log\frac{\pi(y|x)}{\pi_{\mathrm{ref}}(y|x)}\right]\right) $$ {#eq:dpo_deriv_3}

除以 $\beta$ 并重新组合：

$$ = \min_{\pi}\left(\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[ \log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - \frac{1}{\beta}r(x,y) \right]\right) $$ {#eq:dpo_deriv_4}

接下来，我们必须引入一个配分函数 $Z(x)$：

$$ Z(x) = \sum_y \pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right) $$ {#eq:dpo_partition}

配分函数作为未归一化密度 $\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)$ 的归一化因子，从而使其成为每个固定 $x$ 上 $y$ 的有效概率函数。随着我们继续推导，对此的确切需求将变得清晰。

将此代入，我们得到中间变换：

$$ \min_{\pi}\mathbb{E}_{x\sim\mathcal{D}}\mathbb{E}_{y\sim\pi(y|x)}\left[\log\frac{\pi(y|x)}{\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)} - \log Z(x)\right] $$ {#eq:dpo_deriv_5}

要看到这是如何得到的，考虑 @eq:dpo_deriv_4 括号内优化的内部部分：

$$ \log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - \frac{1}{\beta}r(x,y) $$ {#eq:dpo_deriv_6}

然后，对两边加上 $\log Z(x) - \log Z(x)$：

$$ = \log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - \frac{1}{\beta}r(x,y) + \log Z(x) - \log Z(x) $$ {#eq:dpo_deriv_7}

然后，我们分组各项：

$$ = \left( \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} + \log Z(x) \right) - \log Z(x) - \frac{1}{\beta}r(x,y) $$ {#eq:dpo_deriv_8}

使用 $\log(x) + \log(y) = \log(x\cdot y)$（并将 $Z$ 移到分母），我们得到：

$$ = \log \frac{\pi(y|x)}{\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)}- \log Z(x) - \frac{1}{\beta}r(x,y) $$ {#eq:dpo_deriv_9}

接下来，我们将 $\frac{1}{\beta}r(x,y)$ 展开为 $\log \exp \frac{1}{\beta}r(x,y)$ 并做同样操作以获得 @eq:dpo_deriv_5，这里我们稍微重写一下：

$$ \min_{\pi}\mathbb{E}_{x\sim\mathcal{D}} \left[ \mathbb{E}_{y\sim\pi(y|x)}\left[\log\frac{\pi(y|x)}{\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)} \right] - \log Z(x)\right] $$ {#eq:dpo_deriv_10}

有了这个优化形式，我们需要实际求解最优策略 $\pi^*$。
由于我们引入了配分函数 $Z(x)$，从而使项 $\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)$ 成为 $y$ 上的有效概率分布，我们可以认识到内部期望实际上是一个正宗的 KL 散度！

$$ \min_{\pi}\mathbb{E}_{x\sim\mathcal{D}}\left[\mathcal{D}_{\text{KL}} \left(\pi(y|x) \middle\| \frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right) \right) - \log Z(x)\right] $$ {#eq:dpo_deriv_11}

由于项 $\log Z(x)$ 不依赖于 $\pi$（我们正在优化的策略），我们可以忽略它。这给我们留下了我们正在学习的策略与关联配分函数、$\beta$、奖励和参考策略的形式之间的 KL 散度。
Gibbs 不等式告诉我们这在距离为 0 时最小化，仅当两个量相等时！
因此，我们得到最优策略：

$$ \pi^*(y|x) = \pi(y|x) = \frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right) $$ {#eq:dpo_opt_policy}

#### 推导 BT 模型的 DPO 目标

首先，回忆第 5 章奖励建模和第 11 章偏好数据中，人类偏好的 Bradley-Terry 模型形式为：

$$p^*(y_1 \succ y_2 \mid x) = \frac{\exp\left(r^*(x,y_1)\right)}{\exp\left(r^*(x,y_1)\right) + \exp\left(r^*(x, y_2)\right)} $$ {#eq:bradley_terry_dpo}

通过操作 @eq:dpo_opt_policy，我们可以解出最优奖励。首先，对两边取对数：

$$\log \pi^*(y|x) = \log \left( \frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r^*(x,y)\right) \right)$$ {#eq:dpo_reward_deriv1}

使用 $\log(abc) = \log a + \log b + \log c$ 展开右边：

$$\log \pi^*(y|x) = -\log Z(x) + \log \pi_{\text{ref}}(y|x) + \frac{1}{\beta}r^*(x,y)$$ {#eq:dpo_reward_deriv2}

重新排列以解出 $r^*(x,y)$：

$$\frac{1}{\beta}r^*(x,y) = \log \pi^*(y|x) - \log \pi_{\text{ref}}(y|x) + \log Z(x)$$ {#eq:dpo_reward_deriv3}

两边乘以 $\beta$：

$$r^*(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$ {#eq:dpo_reward_full}

然后我们可以将奖励代入 @eq:bradley_terry_dpo 中所示的 Bradley-Terry 方程以获得：

$$p^*(y_1 \succ y_2 \mid x) = \frac{\exp\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)} + \beta \log Z(x)\right)}
{\exp\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)} + \beta \log Z(x)\right) + \exp\left(\beta \log \frac{\pi^*(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)} + \beta \log Z(x)\right)} $$ {#eq:dpo_loss_deriv0}

通过将指数表达式从 $e^{a+b}$ 分解为 $e^a e^b$，然后消去项 $e^{\beta \log Z(x)}$，这简化为：

$$p^*(y_1 \succ y_2 \mid x) = \frac{\exp\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)}\right)}
{\exp\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)}\right) + \exp\left(\beta \log \frac{\pi^*(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)}\right)} $$ {#eq:dpo_loss_deriv1}

然后，分子分母同乘以 $\exp\left(-\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)}\right)$ 以获得：

$$p^*(y_1 \succ y_2 \mid x) = \frac{1}{1 + \exp\left(\beta \log \frac{\pi^*(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)} - \beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)}\right)} $$ {#eq:dpo_loss_deriv2}

最后，使用 sigmoid 函数的定义 $\sigma(x) = \frac{1}{1+e^{-x}}$，我们得到：

$$p^*(y_1 \succ y_2 \mid x) = \sigma\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)} - \beta \log \frac{\pi^*(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)}\right) $$ {#eq:dpo_loss_deriv3}

这是在 Bradley-Terry 模型下，给定最优策略 $\pi^*$ 的偏好数据似然。回忆第 5 章奖励建模中，我们将 Bradley-Terry 目标推导为最大化似然，或等效地最小化负对数似然，这给了我们损失：
$$
\begin{aligned}
\mathcal{L}_{\text{DPO}}(\pi_{\theta}; \pi_{\text{ref}}) &= -\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\left[ \log p(y_c \succ y_r \mid x)  \right] \\
&= -\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\left[ \log \sigma\left(\beta \log \frac{\pi_{\theta}(y_c|x)}{\pi_{\text{ref}}(y_c|x)} - \beta \log \frac{\pi_{\theta}(y_r|x)}{\pi_{\text{ref}}(y_r|x)}\right)\right]
\end{aligned}
$${#eq:dpo_loss_deriv4}

这是 DPO 的损失函数，以 @eq:dpo_core 中所示的形式。
DPO 论文有一个针对 Plackett-Luce 模型下目标的额外推导，在实践中很少使用 [@rafailov2024direct]。

#### 推导 BT DPO 梯度

我们使用 @eq:dpo_gradient 中所示的 DPO 梯度来解释模型如何学习的直觉。
要推导此梯度，我们必须对模型参数取 @eq:dpo_loss_deriv4 的梯度。

$$\nabla_{\theta}\mathcal{L}_{\text{DPO}}(\pi_{\theta}; \pi_{\text{ref}}) = -\nabla_{\theta}\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\left[ \log \sigma\left(\beta \log \frac{\pi_{\theta}(y_c|x)}{\pi_{\text{ref}}(y_c|x)} - \beta \log \frac{\pi_{\theta}(y_r|x)}{\pi_{\text{ref}}(y_r|x)}\right)\right] $$ {#eq:dpo_grad_0}

首先，这可以被重写。
我们知道 sigmoid 函数的导数 $\frac{d}{dx} \sigma(x) = \sigma(x)(1-\sigma(x))$，对数的导数 $\frac{d}{dx} \log x = \frac{1}{x}$，以及 sigmoid 的性质 $\sigma(-x)=1-\sigma(x)$，因此我们可以重新格式化上述方程。

首先，令 $u=\beta \log \frac{\pi_{\theta}(y_c|x)}{\pi_{\text{ref}}(y_c|x)} - \beta \log \frac{\pi_{\theta}(y_r|x)}{\pi_{\text{ref}}(y_r|x)}$（sigmoid 内部的表达式）。
然后，我们有

$$\nabla_{\theta}\mathcal{L}_{\text{DPO}}(\pi_{\theta};\pi_{\text{ref}}) = -\mathbb{E}_{(x, y_c, y_r)\sim \mathcal{D}}\left[\frac{\sigma'(u)}{\sigma(u)}\nabla_{\theta}u\right] $$ {#eq:dpo_grad_2}

展开此式并使用上述 sigmoid 和对数的表达式，得到前面介绍的梯度：

$$ -\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\left[\beta\sigma\left(\beta\log\frac{\pi_{\theta}(y_r|x)}{\pi_{\text{ref}}(y_r|x)} - \beta\log\frac{\pi_{\theta}(y_c|x)}{\pi_{\text{ref}}(y_c|x)}\right)\left[\nabla_{\theta}\log\pi(y_c|x)-\nabla_{\theta}\log\pi(y_r|x)\right]\right] $$ {#eq:dpo_grad_3}

## 数值问题、弱点和替代方案

DPO 算法的许多变体已被提出以解决 DPO 的弱点。
例如，如果没有奖励模型可以评分生成的 rollout，DPO 将每个偏好数据对视为具有同等权重。
实际上，如第 11 章偏好数据所见，有许多方法可以用比二元更丰富的标签捕获偏好数据。
多个算法已被提出以重新平衡优化，而不是平等对待每个对。

- **REgression to RElative REward Based RL (REBEL)** 添加来自奖励模型的信号，作为选中和被拒绝回答之间的边际，而非仅成对偏好数据，以更准确地解决 RLHF 问题 [@gao2024rebel]。
- **保守 DPO (cDPO) 和 Identity 偏好优化 (IPO)** 通过假设偏好数据中存在噪声来解决过拟合。cDPO 假设 N% 的数据被错误标记 [@rafailov2024direct]，IPO 改变优化以软化偏好概率，而不是直接从标签优化 [@azar2024general]。实际上，IPO 将偏好概率改为非线性函数，脱离了 Bradley-Terry 假设，使用 $\Psi(q) = \log\left(\frac{q}{1-q}\right)$。
- **带偏移的 DPO (ODPO)** "要求被偏好和不被偏好回答的似然之间的差异大于一个偏移值" [@amini2024direct] —— 不是平等对待每个数据对，但这可能以更困难的标注环境为代价。

DPO 的一些变体试图通过对损失进行小改动来改善学习信号，或通过减少内存使用使应用更高效。

- **优势比策略优化 (ORPO)** 直接更新策略模型，通过拉向选中回答，类似于指令微调损失，并对选中回答施加小惩罚 [@hong2024reference]。这种损失函数的更改消除了对参考模型的需求，简化了设置。看待 ORPO 的最佳方式是将其视为受 DPO 启发，而非 DPO 的衍生物。
- **简单偏好优化 (SimPO)** 通过对对数概率取平均而不是求和或添加长度归一化，对 DPO 优化进行了微小更改，以改善性能 [@meng2025simpo]。

![DPO 中偏好位移的草图。](images/dpo_displacement.png){#fig:dpo_issue .center}

DPO 中*明显*的核心问题之一是，优化仅推动增加选中和被拒绝回答概率之间的差距。
数值上，模型减少了选中和被拒绝回答两者的概率，但*被拒绝回答减少的程度更大*，如 @fig:dpo_issue 所示。
直觉上，不清楚这如何泛化，但已有工作假定它增加了未处理行为的概率——即语言模型可以生成但不在后训练数据集分布中的 token [@razin2024unintentional] [@ren2024learning]。
简单方法——如 Cal-DPO [@xiao2024cal]（调整优化过程）和 AlphaPO [@gupta2025alphapo]（修改奖励形状）——减轻了这种**偏好位移**。
在实践中，这的确切影响并不广为人知，但指出了在线方法可能优于普通 DPO 的潜在原因。

假设的类 DPO 方法性能上限低于在线（基于 RL 的）RLHF 方法的另一个主要原因是，训练信号来自先前或其他模型的补全。
DPO 的在线变体通过在训练时生成新的补全并纳入偏好信号来减轻这些限制。**在线 DPO** [@guo2024direct] 从当前模型采样生成，而**判别器引导的 DPO**（D2PO）[@singhal2024d2po] 使用奖励模型重新标注以实时创建新的偏好数据，还有更多的变体存在。

还有许多其他 DAA 变体，如直接纳什优化（DNO）[@rosset2024direct] 或二元分类器优化（BCO）[@jung2024binary]，但算法的选择远不如初始模型和使用的数据重要 [@lambert2024t] [@zhao2024rainbowpo] [@gorbatovski2025differences]。

## 实现细节

像 DPO 这样的 DAA 的实现与策略梯度优化器非常不同。
DPO 损失，取自原始实现，大致可以总结如下 [@rafailov2024direct]：

```python
# 策略和冻结参考模型的对数概率差距
pi_logratios = policy_chosen_logps - policy_rejected_logps
ref_logratios = reference_chosen_logps - reference_rejected_logps

# 对数比率之差：当策略将概率
# 移向选中补全时为正
logits = pi_logratios - ref_logratios

# DPO 损失：负对数 sigmoid 驱动策略
# 扩大选中和被拒绝之间的差距
losses = -F.logsigmoid(beta * logits)

# 隐式奖励（已分离 -- 仅用于日志记录）
chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
```

这可以在标准语言模型训练栈中使用，因为这些信息已经在模型的前向传播中被整理（加上参考模型）。

在大多数方面，DAA 更简单且是生活质量的改善，但它们也提供了一组不同的考量。

1. **KL 散度是静态的**：在 DPO 和其他算法中，KL 散度由 $\beta$ 参数显式设置，该参数平衡优化与距离惩罚。这是因为 DPO 向给定数据下的 RLHF 目标的*最优*解采取梯度步骤——它精确地步进到由 $\beta$ 项设置的解。另一方面，基于 RL 的优化器基于批次和最近的数据采取步骤。
2. **缓存对数概率**：DPO 的简单实现同时为策略模型和参考模型做前向传播，以方便损失函数的使用。然而，这将使用的内存加倍，导致 GPU 使用增加。为了避免这一点，可以先计算参考模型在训练数据集上的对数概率，然后在计算损失和更新每个批次参数时重复使用这些缓存的参考对数概率，将峰值内存使用减少 50%。

## 使用合成偏好数据的 DAA

如今，使用 DAA 进行偏好微调的大多数流行数据集都是合成偏好，其中前沿模型将其他模型的输出评为胜者或败者。
突出的例子包括 UltraFeedback（此类别的第一个）[@cui2023ultrafeedback]、Tülu 3（使用扩展的 UltraFeedback 方法构建）[@lambert2024t]、SmolLM 3 的数据 [@bakouch2025smollm3]，或随 Olmo 3 发布的 Dolci Pref 数据集 [@teamolmo2025olmo3]。

构建这些数据集的最佳实践仍在演变。
Tülu 3 和 2024 年 11 月左右发布的数据集证明，合成的成对偏好数据需要在某种意义上是"在线策略"的，即一些补全是从你正在微调的模型中生成的（同时混合在更大的模型池中）。
数据的这种在线策略性质确保了 DAA 将优化模型生成的正确 token 空间——因为损失函数是对比的，且不如指令微调直接。
后来，随着 2025 年 Olmo 3 和 SmolLM 3 的发布，其他工作支持了一种不同的理论，称为 Delta Learning，该理论认为选中和被拒绝补全之间的差异比具体使用哪些模型进行补全更重要 [@geng2025the]。
例如，在这两个引用的模型中，选中的回答来自 Qwen 3 32B，被拒绝的回答来自 Qwen 3 0.6B——两位作者同时且独立地开发了这种配对。

总的来说，在合成偏好数据上使用 DAA 训练模型是大多数实践者应该开始的地方，因为实现简单，且相对于基于强化学习方法的偏好微调具有强大性能。
使用大量合成偏好数据时存在其他次要问题，例如模型评判补全时的偏差。
鉴于前沿模型如 GPT-4 已知具有长度偏差 [@dubois2024length] 和对与自己匹配的输出的偏好 [@panickssery2024llm]（更多信息见第 12 章），数据集中"选中"部分的文本稍微更可能是来自 OpenAI 模型或另一个在风格上类似它的强模型。

总结本节，我们将涵盖这些方法如何改变被训练模型生成的直觉。
在高层次上，大多数 DAA 优化以增加"选中"和"被拒绝"补全概率之间的差距（一些不太流行的算法设计为稍微改变这些动态，但核心保持不变）。
如本章前面讨论的（见 @fig:dpo_issue），这通常意味着两个概率都下降，但被拒绝的回答下降程度更大。
序列中的每个 token 接收不同的梯度（大小和方向），基于它对整体偏好差距的贡献程度，允许优化器识别哪些 token 对结果最重要。

## DAA 对比 RL：在线与离线数据

广义上说，争论归结为一个问题：我们是否需要强化学习的内部机制——价值函数、策略梯度等等——来用 RLHF 对齐语言模型？
这像大多数以这种方式表述的问题一样，是过于简单的。
当然，两种方法都很健全，但重要的是说明根本差异和性能流形在哪里。

多份报告得出结论，基于策略梯度和 RL 的方法优于 DPO 及其变体。
论证采取不同形式，从使用不同算法但控制数据训练模型 [@ivison2024unpacking] [@xu2024dpo]，或研究在线策略数据在 RL 优化循环中的角色 [@tajwar2024preference]。
在所有这些情况下，DPO 算法都略逊一筹。

即使有这种性能差距，DAA 由于其简单性仍然在领先模型中被广泛使用。
DAA 提供了一个受控的环境，可以快速迭代训练数据和其他配置，并且鉴于数据通常比算法更重要，使用 DPO 是可以的。

随着主要以 RL 训练的推理模型的出现，进一步的投资将回归到使用 RL 进行偏好微调，从长期来看这将提高 RL 基础设施的鲁棒性，并巩固 DAA 和 RL 在从人类反馈优化方面的差距。

## 建议的实验

`code/direct_alignment/` 中的配套代码在偏好数据上训练 DPO 和几个相关损失。
这是开始偏好微调实验最容易的地方，因为设置是离线的：不需要奖励模型服务器或 rollout 循环。

1. **在 UltraFeedback 上训练小型 DPO 运行。**

   ```bash
   cd code/
   uv run python -m direct_alignment.train --loss dpo --max_samples 1000
   ```

   观察 `loss`、`accuracy`、`margins`、`chosen_rewards` 和 `rejected_rewards`。
   主要的健全性检查是隐式奖励边际应该向期望的方向移动，而模型的采样生成不会崩溃。

2. **比较 DPO、IPO 和长度归一化的 DPO。**

   ```bash
   cd code/
   uv run python -m direct_alignment.train --config direct_alignment/configs/dpo.yaml
   uv run python -m direct_alignment.train --config direct_alignment/configs/ipo.yaml
   uv run python -m direct_alignment.train --config direct_alignment/configs/dpo_norm.yaml
   ```

   比较边际规模和学习率敏感性。
   IPO 的损失不在与 DPO 相同的数值尺度上，因此通过 `accuracy` 和边际行为来解读它，而不是仅看原始损失。

3. **仔细尝试无参考模型变体。**
   从其配置运行 SimPO 或 ORPO，然后检查训练期间记录的生成样本。
   这些损失对对数概率缩放和学习率更敏感，这使它们成为有用的调试练习。

   ```bash
   cd code/
   uv run python -m direct_alignment.train --config direct_alignment/configs/simpo.yaml
   uv run python -m direct_alignment.train --config direct_alignment/configs/orpo.yaml
   ```

4. **在改变损失之前改变数据。**
   保持损失固定，改变 `--max_samples`、`--max_length` 或偏好数据集。
   如果结果的变化比在不同类 DPO 目标之间切换更大，这是偏好微调核心主题的经验提醒：数据通常比小的算法差异占主导地位。
