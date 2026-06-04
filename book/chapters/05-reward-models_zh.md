<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "指令微调"
prev-url: "04-instruction-tuning"
page-title: 奖励建模
search-title: "第 5 章：奖励建模"
meta-description: "如何从偏好数据训练奖励模型，并将其用作 RLHF 后训练流程中的学习目标。"
next-chapter: "强化学习"
next-url: "06-policy-gradients"
lectures:
  - video: "https://www.youtube.com/watch?v=4gIwiSPmQkU&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=3"
    label: "第 2 讲：IFT、奖励建模、拒绝采样（第 4、5 和 9 章）"
---

# 奖励建模

奖励模型是现代 RLHF 方法的核心，它是学习复杂人类偏好的地方。
它们使我们的模型能够从难以明确指定的信号中学习。
它们将数据中的复杂特征压缩为可用于下游训练的表示——这是一种再次展示现代深度学习复杂能力的魔法。
这些模型充当核心优化的代理目标，正如后续章节所研究的那样。
如 @fig:rm-role-in-rlhf 所示，奖励模型扮演着类似于标准 RL 环境的角色，为智能体提供学习信号，但与固定环境不同，我们可以从人类偏好中学习它。

奖励模型历史上在强化学习研究中被广泛用作环境奖励的代理 [@sutton2018reinforcement]。
现代形式的奖励模型被提出作为研究价值对齐问题的工具 [@leike2018scalable]。
这些模型通常接受某种输入并输出单个标量奖励值。
此奖励可以有多种形式——在传统 RL 问题中，它试图近似问题的精确环境奖励，但在 RLHF 中我们将看到奖励模型实际上输出某个输入"高质量"的概率（即成对偏好关系中选中的答案）。
RLHF 的奖励建模实践与逆强化学习密切相关，逆强化学习的问题是在给定行为轨迹的情况下近似智能体的奖励函数 [@ng2000algorithms]，也与深度强化学习的其他领域相关。
高层次的问题陈述是相同的，但实现和关注领域完全不同，因此它们通常被视为完全独立的研究领域。

最常见的奖励模型，通常称为 Bradley-Terry 奖励模型，也是本章的主要焦点，它预测一段文本接近训练比较中"被偏好"文本的概率。
在本节后面，我们还将这些模型与结果奖励模型（ORM）、过程奖励模型（PRM）和其他类型的奖励模型进行比较。

*在本章中，我们使用 $x$ 表示提示，$y$ 表示补全。这种表示法在语言模型文献中很常见，其中方法操作完整的提示-补全对而非单个 token。*

![RLHF 中的奖励模型扮演着在标准 RL 中返回奖励的环境组件的角色。关键区别在于，在 RLHF 中，我们可以从人类偏好中控制和学习这个奖励函数，而不是让它由环境固定。](images/rlhf-overview.png){#fig:rm-role-in-rlhf}

## 训练 Bradley-Terry 奖励模型

奖励模型的经典实现源自 Bradley-Terry 偏好模型 [@BradleyTerry]。
对于如何训练 RLHF 的标准奖励模型，有两种流行的表达方式——它们在数学上是等价的。
首先，偏好的 Bradley-Terry 模型定义在项目 $i$ 和 $j$ 的成对比较中，评判者偏好 $i$ 超过 $j$ 的概率：

$$P(i > j) = \frac{p_i}{p_i + p_j}.$$ {#eq:bradterry}

Bradley-Terry 模型假设每个项目具有潜在强度 $p_i > 0$，并且观察到的偏好是这些潜在强度的有噪声反映。
通常使用无界分数重新参数化 Bradley-Terry 模型，其中 $p_i = e^{r_i}$，这导致以下形式：

$$P(i > j) = \frac{e^{r_i}}{e^{r_i} + e^{r_j}} = \sigma(r_i-r_j).$$ {#eq:bradterry_unbounded}

只有分数的差异重要：对每个 $r_k$ 加上相同的常数 $c$，$P(i > j)$ 保持不变。
这些形式不是自然法则，而是人类偏好的有用近似，在 RLHF 中通常效果很好。

为了训练奖励模型，我们必须制定一个满足上述关系的损失函数。
在实践中，这是通过将语言模型转换为输出标量分数的模型来完成的，通常通过一个小型线性头从模型的最终隐藏状态产生单个奖励值。
给定提示 $x$ 和两个采样的补全 $y_1$ 和 $y_2$，我们使用奖励模型 $r_\theta$ 对两者进行评分，并将条件分数写为 $r_\theta(y_i \mid x)$。

奖励模型赋予 $y_1$ 优于 $y_2$ 的概率变为：

$$P(y_1 > y_2 \mid x) = \frac{\exp\left(r_\theta(y_1 \mid x)\right)}{\exp\left(r_\theta(y_1 \mid x)\right) + \exp\left(r_\theta(y_2 \mid x)\right)}.$$ {#eq:bradterryrm}

我们将被偏好的补全记为 $y_c$（选中），将被拒绝的补全记为 $y_r$。

由此产生的损失鼓励奖励模型为人类偏好的补全分配比被拒绝的补全更高的分数，使用 sigmoid 将分数差转换为概率。
@eq:bradterryrm 中的偏好似然是起点。我们首先将该似然重写为 sigmoid 形式，仅在最后一步将其转换为用于训练奖励模型的等价负对数似然损失：

$$
\begin{aligned}
\theta^* = \arg\max_\theta P(y_c > y_r \mid x) &= \arg\max_\theta \frac{\exp\left(r_\theta(y_c \mid x)\right)}{\exp\left(r_\theta(y_c \mid x)\right) + \exp\left(r_\theta(y_r \mid x)\right)} \\
&= \arg\max_\theta \frac{\exp\left(r_\theta(y_c \mid x)\right)}{\exp\left(r_\theta(y_c \mid x)\right)\left(1 + \frac{\exp\left(r_\theta(y_r \mid x)\right)}{\exp\left(r_\theta(y_c \mid x)\right)}\right)} \\
&= \arg\max_\theta \frac{1}{1 + \frac{\exp\left(r_\theta(y_r \mid x)\right)}{\exp\left(r_\theta(y_c \mid x)\right)}} \\ 
&= \arg\max_\theta \frac{1}{1 + \exp\left(-(r_\theta(y_c \mid x) - r_\theta(y_r \mid x))\right)} \\
&= \arg\max_\theta \sigma \left( r_\theta(y_c \mid x) - r_\theta(y_r \mid x) \right) \\
&= \arg\min_\theta - \log \left( \sigma \left(r_\theta(y_c \mid x) - r_\theta(y_r \mid x)\right) \right)
\end{aligned}
$$ {#eq:bradterryrm_deriv}

第一种形式是上面推导出的对数 sigmoid 表达式，如在 [@ouyang2022training] 和其他工作中使用的：
$$\mathcal{L}(\theta) = - \log \left( \sigma \left( r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x) \right) \right)$$ {#eq:rewardmodeling1}

第二种是用 softplus 函数 $\log(1+e^x)$ 表达的数学等价形式，如在 [@askell2021general] 和其他工作中使用的：
$$\mathcal{L}(\theta) = \log \left( 1 + e^{r_{\theta}(y_r \mid x) - r_{\theta}(y_c \mid x)} \right)$$ {#eq:rewardmodeling2}

令 $\Delta = r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x)$ 并使用 $\sigma(\Delta) = \frac{1}{1 + e^{-\Delta}}$，可得 $-\log\sigma(\Delta) = \log(1 + e^{-\Delta}) = \log\left(1 + e^{r_{\theta}(y_r \mid x) - r_{\theta}(y_c \mid x)}\right)$，从而证明它们是等价的。
两者都出现在 RLHF 文献中。

![训练偏好奖励模型需要选中和被拒绝补全的配对。模型从序列级表示（通常是序列结束（EOS）token 的隐藏状态）为每个补全计算标量分数，对比损失仅取决于两者之间的分数差。](images/pref_rm_training.png){#fig:pref_rm_training}

## 默认奖励模型架构

奖励模型最常见的实现方式是通过类似于 Transformers 的 `AutoModelForSequenceClassification` 的抽象，它在语言模型上附加一个小型线性头，并在训练或推理时对提示-补全对产生标量奖励分数。
在推理时，模型输出*文本被选中的相对可能性*，作为来自模型的单个 logit。

存在其他实现选项，例如直接从最终嵌入中取线性层，但在开源工具中不太常见。

## 实现示例

实现奖励建模损失相当简单。
更多的实现挑战在于设置单独的数据加载器和推理流水线。
假设有正确的数据加载器，包含带有补全的已分词、选中和被拒绝的提示，损失实现如下：
```python
import torch.nn as nn
# inputs_chosen / inputs_rejected 包含提示 token x 以及奖励模型联合评分的
# 相应补全 token（y_c 或 y_r）。
rewards_chosen = model(**inputs_chosen)
rewards_rejected = model(**inputs_rejected)

loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
```

从更宏观的角度看，这通常位于因果语言模型（从左到右生成 token 的模型，每个 token 以所有前面的 token 为条件）内部，该模型添加了一个额外的头（使用上述损失学习），从最终隐藏状态过渡到输入的分数。
代码接受标准的 transformer 输入——`input_ids`（分词文本）和 `attention_mask`（标记真实 token 与填充）——并在最后一个真实 token 处提取隐藏状态（模型对输入的内部表示），然后通过线性层产生标量奖励。
该模型将具有如下结构：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BradleyTerryRewardModel(nn.Module):
    """
    用于 Bradley-Terry 偏好学习的标准标量奖励模型。

    用法（成对 BT 损失）：
        rewards_chosen = model(**inputs_chosen)    # (batch,)
        rewards_rejected = model(**inputs_rejected)  # (batch,)
        loss = -F.logsigmoid(rewards_chosen - rewards_rejected).mean()
    """
    def __init__(self, base_lm):
        super().__init__()
        self.lm = base_lm  # e.g., AutoModelForCausalLM
        self.head = nn.Linear(self.lm.config.hidden_size, 1)

    def _sequence_rep(self, hidden, attention_mask):
        """
        获取每个序列的单一向量进行评分。
        默认：最后一个非填充 token（EOS token）；如果没有掩码，则取最后一个 token。
        hidden: (batch, seq_len, hidden_size)
        attention_mask: (batch, seq_len)
        """

        # 每个序列中最后一个非填充 token 的索引
        # attention_mask 对真实 token 为 1，对填充为 0
        lengths = attention_mask.sum(dim=1) - 1  # (batch,)
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_idx, lengths]  # (batch, hidden_size)

    def forward(self, input_ids, attention_mask):
        """
        一个设计用于展示标准奖励模型推理结构的前向传播。
        要训练此模型，需要修改此函数以从选中和被拒绝的输入计算奖励，
        并应用上述损失。
        """
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # 最终隐藏状态：(batch, seq_len, hidden_size)
        hidden = outputs.hidden_states[-1]

        # 每个序列一个标量奖励：(batch,)
        seq_repr = self._sequence_rep(hidden, attention_mask)
        rewards = self.head(seq_repr).squeeze(-1)

        return rewards
```

在本节及后续内容中，奖励模型（以及大部分后训练）的大部分实现复杂性都围绕正确构造数据加载器和分布式学习系统。
注意，训练奖励模型时，最常见的做法是只训练 1 个 epoch 以避免过拟合。

## 奖励模型变体

奖励建模是 RLHF 中相对研究不足的领域。
传统的奖励建模损失在许多流行工作中被修改，但这些修改尚未固化为单一的最佳实践。

### 偏好边际损失

当标注者以 Likert 量表（一种带有指示偏好程度的有序类别的评分量表，例如 1–5）提供分数或排名时，关系量的大小可以在训练中使用。
最常见的做法是沿偏好方向将数据二值化，将相对评分或排名强度的混合信息简化为仅选中和被拒绝的补全。
额外的信息，如偏好的程度，已被用于改进模型训练，但尚未收敛为标准实践。
Llama 2 提出使用两个数据点之间的边际 $m(y_c, y_r)$ 来区分偏好程度：

$$\mathcal{L}(\theta) = - \log \left( \sigma \left( r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x) - m(y_c, y_r) \right) \right)$$ {#eq:rewardmodelingmargin}

例如，每个补全通常被赋予 1 到 5 的质量排名。
在选中样本被赋予 5 分、被拒绝样本被赋予 2 分的情况下，边际 $m(y_c, y_r)= 5 - 2 = 3$。
可以探索计算边际的其他函数。

注意，在 Llama 3 中边际项被移除了，因为团队观察到扩展后改进递减。

### 平衡每个提示的多个比较

InstructGPT 研究了使用每个提示 $K = 4$ 到 $9$ 个补全进行排名的影响，从每个提示产生 $\binom{K}{2}$ 个成对比较 [@ouyang2022training]。
因为这些比较高度相关（它们共享相同的提示），将它们幼稚地混入数据集会导致奖励模型过拟合。
为了解决这个问题，他们按每个提示的每个比较加权损失更新——如果不重新加权，具有更多补全的提示将仅仅因为产生更多配对而贡献更多总损失。
在实践中，来自单个提示的所有 $\binom{K}{2}$ 比较通常包含在同一个训练批次中并一起平均，因此每个提示贡献一个分组更新，而不是出现在许多单独的批次中。
这减少了对单个提示的过拟合，并防止具有更多采样补全的提示主导损失。
损失函数变为：

$$\mathcal{L}(\theta) = - \frac{1}{\binom{K}{2}} \mathbb{E}_{(x, y_c, y_r)\sim D} \log \left( \sigma \left( r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x) \right) \right)$$ {#eq:rewardmodelinginstructgpt}


### K-Wise 损失函数

还有许多其他公式可以为 RLHF 创建合适的人类偏好模型。
其中一个例子，用于流行的早期 RLHF 模型 Starling 7B 和 34B [@zhu2024starling]，是基于 Plackett-Luce 模型的 K-wise 损失函数 [@liu2019learning]。

Zhu 等人 2023 [@zhu2023principled] 将设置形式化如下。
给定提示或状态 $s^i$，从 $P(a_0,\cdots,a_{K-1}|s^i)$ 采样 $K$ 个动作 $(a_0^i, a_1^i, \cdots, a_{K-1}^i)$。
然后，标注者按偏好对 $K$ 个动作进行排名，产生排列 $\sigma^i: [K] \mapsto [K]$，其中 $\sigma^i(0)$ 是最受欢迎的动作。这产生所有 $K$ 个项目的完整排名上的 Plackett-Luce 概率：

$$P(\sigma^i|s^i,a_0^i,a_1^i,\ldots,a_{K-1}^i) = \prod_{k=0}^{K-1} \frac{\exp(r_{\theta\star}(s^i,a_{\sigma^i(k)}^i))}{\sum_{j=k}^{K-1}\exp(r_{\theta\star}(s^i,a_{\sigma^i(j)}^i))}$$ {#eq:kwise_rm}

当 $K = 2$ 时，这退化为成对比较的 Bradley-Terry（BT）模型。
无论如何，一旦训练完成，这些模型在 RLHF 训练中的使用方式与其他奖励模型类似。


## 结果奖励模型

<!-- 非常感谢东北大学研究生 Hangliang Ren 对本节（以及 PRM）的帮助，参见 https://github.com/myhott163com/RLHF_ORM_PRM -->

语言模型和其他 AI 系统的大多数*偏好微调*都是使用上面讨论的 Bradley-Terry 模型完成的。
对于推理密集型任务，可以使用结果奖励模型（Outcome Reward Model, ORM）。
ORM 的训练数据以类似于标准偏好微调的方式构建。
这里，我们有一个问题陈述或提示 $x$ 和两个补全 $y_1$ 和 $y_2$。
这里使用的归纳偏置是，一个补全应该是问题的正确解决方案，另一个是错误的，从而产生 $(y_c,y_{ic})$。

所用模型的架构与标准奖励模型非常相似，在线性层附加到可以输出单个 logit 的模型之上（在 RM 的情况下）——使用 ORM 时，随后的训练目标略有不同 [@cobbe2021gsm8k]：

> [我们] 使用联合目标训练验证器，模型学习将模型补全标记为正确或错误，以及原始的语言建模目标。
> 在架构上，这意味着我们的验证器是语言模型，带有一个小型标量头，在
> 每个 token 的基础上输出预测。
> 我们将这个标量头实现为一个单一的偏置参数和一个单一的增益参数，它们作用于语言模型最终反嵌入层输出的 logit。

翻译一下，这被实现为一个语言建模头，可以为每个 token 预测两个类别（1 表示正确，0 表示错误），而不是传统 RM 的分类头，后者为整个序列输出一个 logit。
形式上，遵循 [@lyu2025exploring]，这是一个逐 token 的二元交叉熵损失：

$$\mathcal{L}_{\text{CE}}(\theta) = -\mathbb{E}_{(s,r)\sim \mathcal{D}}\left[r\log p_\theta(s) + (1-r)\log(1-p_\theta(s))\right]$$ {#eq:orm_loss}

其中 $r \in \{0,1\}$ 是二元标签，1 适用于对给定提示的正确回答，0 适用于错误回答，$p_\theta(s)$ 是与被训练模型的预测正确概率成比例的标量。
在代码中，这个结果标签被复制到每个补全 token 上，而提示 token 用 `-100` 掩码，使它们不贡献损失。

实现结果奖励模型（以及其他类型，正如我们将看到的 Process Reward Model）涉及基于补全是否为正确样本，在每个 token 上应用交叉熵损失。
这更接近语言建模损失，不需要标准 Bradley-Terry 奖励模型的结构化选中-被拒绝性质。
在下面简化的 ORM 训练设置中，我们没有采样新的 token 或在下个 token 预测上训练 LLM；我们将固定的提示-补全序列通过骨干网络并训练 ORM 头来预测正确性标签。

模型结构可以如下所示：

```python
import torch.nn as nn
import torch.nn.functional as F

class OutcomeRewardModel(nn.Module):
    def __init__(self, base_lm):
        super().__init__()
        self.lm = base_lm  # e.g., AutoModelForCausalLM
        self.head = nn.Linear(self.lm.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        input_ids 包含完整的提示+补全序列。
        labels 与 token 对齐：提示 token 为 -100，每个补全
         token 重复序列结果标签（1=正确, 0=错误）。
        如果 labels=None，这是一个仅推理的前向传播，损失返回 None。
        """
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # 最终隐藏状态：(batch, seq_len, hidden_size)
        hidden = outputs.hidden_states[-1]
        # 每个 token 一个标量 logit：(batch, seq_len)
        logits = self.head(hidden).squeeze(-1)

        # 仅推理的前向传播：不计算损失。
        if labels is None:
            return None, logits
        # 仅对补全 token 计算损失（标签 0 或 1）
        # 提示 token 的 labels = -100
        mask = labels != -100
        loss = None
        if mask.any():
            loss = F.binary_cross_entropy_with_logits(
                logits[mask], labels[mask].float()
            )
        else:
            loss = logits.sum() * 0
        return loss, logits
```

简化版损失如下：

```python
# 将完整的提示+补全序列送入一次；这里没有 token 采样发生。
# 假设模型已有：model.lm（骨干）+ model.head
hidden = model.lm(**inputs, output_hidden_states=True).hidden_states[-1]
logits_per_token = model.head(hidden).squeeze(-1)  # (batch, seq_len)
# 这在其他实现中有时会被压缩为 model.forward()

# 二元标签：1=正确, 0=错误（提示 token 用 -100 掩码）
mask = labels != -100
loss = F.binary_cross_entropy_with_logits(
    logits_per_token[mask], labels[mask].float()
)
```

这里重要的直觉是，ORM 将在序列中的每个 token 处输出正确性概率（仅由最终答案判断——ORM 训练过程不会捕获推理错误）。
这可能是一个嘈杂的过程，因为更新和损失根据结果和注意力映射在每个 token 上传播。

![在推理时，结果奖励模型在补全 token 上输出每个 token 的正确性概率。提示 token 被忽略不计分，补全概率可以聚合为回答级分数，用于验证、过滤或重排序。](images/orm_inference.png){#fig:orm_inference}

![训练结果奖励模型使用来自验证器或数据集的离线标签（例如，正确补全全部为 1）。每个补全 token 用二元交叉熵针对结果标签进行训练，每个 token 的概率被聚合为最终分数，用于验证、过滤或重排序。](images/orm_training){#fig:orm_training}

这些模型继续被使用，但在开源 RLHF 工具中支持较少。
例如，相同类型的 ORM 在开创性工作 *Let's Verify Step by Step* 中被使用 [@lightman2023let]，但没有损失中的语言建模预测部分。
然后，最终损失是每个 token 上的交叉熵损失，预测最终答案是否正确。

由于缺乏支持，术语结果奖励模型（ORM）以多种方式被使用。
一些文献，例如 [@lyu2025exploring]，继续使用 Cobbe 等人 2021 的原始定义；其他文献将其更广泛地用于任何被训练来预测补全是否正确的验证器。


## 过程奖励模型

过程奖励模型（Process Reward Models, PRMs），最初称为过程监督奖励模型，是经过训练的奖励模型，在思维链推理过程中的每个*步骤*输出分数。
这些不同于仅在 EOS token 处输出分数的标准 RM 或每个 token 输出分数的 ORM。
过程奖励模型需要在每个推理步骤结束时进行监督，然后以类似的方式训练，其中步骤中的 token 被训练到其相关目标——PRM 中的目标是步骤，ORM 中的目标是整个回答。

遵循 [@lightman2023let]，二元标记的 PRM 通常使用每步交叉熵损失进行优化：

$$\mathcal{L}_{\text{PRM}}(\theta) = - \mathbb{E}_{(x, s) \sim \mathcal{D}} \left[ \sum_{i=1}^{K} y_{s_i} \log r_\theta(s_i \mid x, s_{< i}) + (1 - y_{s_i}) \log \left(1 - r_\theta(s_i \mid x, s_{< i})\right) \right] $$ {#eq:prm_loss}

其中 $s$ 是具有 $K$ 个标注步骤的采样思维链，$y_{s_i} \in \{0,1\}$ 表示第 $i$ 步是否正确，$r_\theta(s_i \mid x, s_{< i})$ 是 PRM 预测的步骤 $s_i$ 有效的概率，以原始提示 $x$ 和所有先前步骤 $s_{< i}$ 为条件。

以下是如何在训练器中打包每个步骤标签的示例，来自 HuggingFace 的 TRL（Transformer 强化学习）[@vonwerra2022trl]：

```python
# 获取分隔 token 的 ID 并将其添加到补全中
separator_ids = tokenizer.encode(step_separator, add_special_tokens=False)
completions_ids = [completion + separator_ids for completion in completions_ids]

# 创建标签
labels = [[-100] * (len(completion) - 1) + [label] for completion, label in zip(completions_ids, labels)]
```

传统上，PRM 使用语言建模头进行训练，该头仅在推理步骤结束时输出 token，例如在对应于双换行或其他特殊 token 的 token 处。
这些预测通常是 -1 表示错误，0 表示中性，1 表示正确。
这些标签不一定与模型是否走在正确路径上有关，而是与步骤是否正确有关。

![过程奖励模型仅在步骤边界（例如换行 token）提供监督。每个步骤接收一个 3 类标签：正确（+1）、中性（0）或错误（-1）。所有其他 token 在训练期间被掩码。](images/prm_training_inference.png){#fig:prm_training_inference}

PRM 的示例构造如下所示。

```python
import torch.nn as nn
import torch.nn.functional as F

class ProcessRewardModel(nn.Module):
    def __init__(self, base_lm, num_classes=3):
        super().__init__()
        self.lm = base_lm  # e.g., AutoModelForCausalLM
        self.head = nn.Linear(self.lm.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        输入是分词的提示和补全，其中"推理步骤"的结束由指定的分隔 token 表示，
        如换行或其他特殊标记，而不是批次填充。
        labels 将是标签列表，True、False 和 Neutral（3 个标签），
        将由模型预测。
        如果 labels=None，这是一个仅推理的前向传播，损失返回 None。
        """
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # 最终隐藏状态：(batch, seq_len, hidden_size)
        hidden = outputs.hidden_states[-1]
        # 每个 token 一个 logit 向量：(batch, seq_len, num_classes)
        logits = self.head(hidden)

        # 仅推理的前向传播：不计算损失。
        if labels is None:
            return None, logits
        # 仅在步骤边界计算损失（其中 labels != -100）
        # 标签映射：-1 -> 0, 0 -> 1, 1 -> 2（类别索引）
        mask = labels != -100
        loss = None
        if mask.any():
            loss = F.cross_entropy(
                logits[mask], labels[mask]
            )
        else:
            loss = logits.sum() * 0
        return loss, logits
```

核心损失函数看起来与结果奖励模型非常相似，标签在不同的间隔应用。
```python
# 假设模型输出每个 token 的 3 类 logit
hidden = model.lm(**inputs, output_hidden_states=True).hidden_states[-1]
logits = model.head(hidden)  # (batch, seq_len, 3)

# 仅在步骤边界的 3 类标签：0=-1, 1=0, 2=1（其他用 -100 掩码）
mask = labels != -100
loss = F.cross_entropy(logits[mask], labels[mask])
```

## 比较奖励模型类型（与价值函数）

所涵盖的各种奖励模型类型展示了在 RLHF 和其他后训练方法中"质量"可以被衡量的各种方式。
下面是模型预测什么以及如何训练的总结。

::: {.table-wrap}
| 模型类别 | 预测什么 | 如何训练 | LM 结构 |
|------------|------------------|---------------------|--------------|
| **奖励模型（RM）** | 序列级质量分数 $r_\theta(x, y)$ | 补全之间成对（或 N-wise）比较的对比损失 | EOS/最后 token 隐藏状态上的线性头 |
| **结果奖励模型（ORM）** | 每个 token 答案正确的概率 | 标记的结果对（例如，可验证领域的成功/失败） | 每 token 二元交叉熵头；标签重复结果标签 |
| **过程奖励模型（PRM）** | 推理步骤结束时中间步骤的奖励或分数 | 使用中间反馈或逐步标注进行训练（对推理步骤中的每个 token 训练） | 预测步骤正确性的每 token 头（-1, 0, 1） |
| **价值函数** | 给定当前状态的期望回报 | 通过回归训练到序列中的每个点 | 具有每 token 输出的标量回归头 |
表：奖励模型类型的比较。 {#tbl:rm_compare}
:::

关于此表中的区别有几点说明，因为模型类型之间的边界并不总是清晰的：

- 在偏好微调和推理训练中，价值函数通常具有折扣因子 1，这使得价值函数更接近结果奖励模型，但具有不同的训练损失。
- 过程奖励模型可以通过从中间状态进行 rollout 并收集结果数据来监督。这混合了多种思想，但如果*损失*使用每个推理步骤的标签，最好将其称为 PRM。

**如果你用正确/错误对训练 Bradley-Terry 成对模型呢？**
关于结果奖励模型的大部分困惑来自一小部分文献，这些文献在源自答案正确性的成对数据上训练奖励模型。
在这个领域中，你将选中的回答设置为问题的正确答案，将拒绝的回答设置为*同一问题的*错误答案。
这在技术上不是 ORM，仍然直接用对比的序列级损失训练。
这在技术上仍然是 Bradley-Terry 模型，属于我们涵盖的第一类模型。

**ORM 对比价值函数。**
ORM 和价值函数可能看起来相似，因为两者都使用相同的头架构产生每 token 输出，但它们在*预测什么*和*目标来自哪里*方面有所不同：

- **ORM** 预测即时的、token 局部的量：$p(\text{correct}_t)$ 或 $r_t$。目标来自*离线标签*（验证器或数据集将 token/序列标记为正确或错误）。
- **价值函数**预测期望的*剩余*回报：$V(s_t) = \mathbb{E}\left[\sum_{k \geq t} \gamma^{k-t} r_k \mid s_t\right]$。目标通常*根据当前策略 $\pi_\theta$ 的在线策略 rollout 计算*，并随着策略的变化而变化（技术上，价值函数也可以是离线的，但这在语言建模工作中尚未建立）。

如果你定义密集的 token 奖励 $r_t = \mathbb{1}[\text{token is correct}]$ 并使用 $\gamma = 1$，则 ORM 在学习 $r_t$（或 $p(r_t = 1)$），而价值头在学习剩余和 $\sum_{k \geq t} r_k$。
它们可以共享相同的基础模型和头维度，但*语义和监督流水线*不同：ORM 从固定标签离线训练，而价值函数在线训练并用于为策略梯度计算优势 $A_t = \hat{R}_t - V_t$。

### 不同奖励模型类型的推理

模型在推理时（训练完成后）以不同方式处理数据，以处理 RM 用于的一系列任务。

**Bradley-Terry RM（偏好模型）：**

- *输入：* 提示 $x$ + 候选补全 $y$
- *输出：* 通过 EOS/最后 token 隐藏状态的线性层得到单个标量 $r_\theta(x, y)$
- *用法：* 对 $k$ 个补全重排序，选 top-1（best-of-N 采样）；或为 RLHF 提供终端奖励
- *聚合：* 使用标量输出，不需要聚合

**结果 RM：**

- *输入：* 提示 $x$ + 补全 $y$
- *输出：* 补全 token 上每个 token 的概率 $p_t \approx P(\text{在 token } t \text{ 正确})$
- *用法：* 对完成的候选评分；通过均值、最小值（尾部风险）或乘积 $\prod_t p_t$（等价地，对数概率之和 $\sum_t \log p_t$）聚合
- *聚合选择：* 平均正确性、最小 $p_t$、最后 $m$ 个 token 的平均值，或阈值标记（如果有任何 $p_t < \tau$）

**过程 RM：**

- *输入：* 提示 $x$ + 带步骤边界的推理跟踪
- *输出：* 步骤边界的分数（例如，正确/中性/错误的类别 logit）
- *用法：* 对完成的思维链评分；或通过剪枝低分分支来引导搜索/解码
- *聚合：* 跨步骤（而非 token）——平均步骤分数、最小值（快速失败）或倾向后期步骤的加权和

**价值函数：**

- *输入：* 提示 $x$ + 当前前缀 $y_{\leq t}$（一个状态）
- *输出：* 补全中每个 token 位置的 $V_t$（从状态 $t$ 开始的期望剩余回报）
- *用法：* 在 RL 训练期间计算每 token 优势 $A_t = \hat{R}_t - V_t$；每个步骤的值作为基线
- *聚合：* 通常取最后一个生成 token 的 $V$；解释与"正确性概率"不同

总之，理解不同模型的方式是：

- **RM：** "整个答案有多好？" → 标量值
- **ORM：** "哪些部分看起来正确？" → 每 token 正确性
- **PRM：** "推理步骤是否合理？" → 每步骤分数
- **价值函数：** "从这里开始还剩下多少奖励？" → RL 优势的基线

## 生成式奖励建模（又称 LLM-as-a-judge）

随着偏好数据成本的增加，一个大型研究领域出现了，即使用现有的语言模型作为人类偏好或评估设置中的评判者 [@zheng2023judging]。
核心理念是向语言模型提示判断指令、一个提示和两个补全（就像对人类标注者所做的那样）。
以下是一个示例提示，来自聊天评估 MT-Bench 的开创性工作之一 [@zheng2023judging]：

```text
[System]
请充当公正的评判者，评估下面显示的两个 AI 助手对用户问题的回答质量。
你应该选择更好地遵循用户指令并回答用户问题的助手。
你的评估应考虑诸如回答的帮助性、相关性、准确性、深度、创造性和详细程度等因素。
通过比较两个回答开始你的评估，并提供简短的解释。
避免任何位置偏见，确保回答的呈现顺序不会影响你的决定。
不要让回答的长度影响你的评估。
不要偏向助手的某些名称。
尽可能客观。
在提供解释之后，严格遵循以下格式输出你的最终裁决："[[A]]" 如果助手 A 更好，"[[B]]" 如果助手 B 更好，"[[C]]" 表示平手。
[用户问题]
{question}
[助手 A 回答的开始]
{answer_a}
[助手 A 回答的结束]
[助手 B 回答的开始]
{answer_b}
[助手 B 回答的结束]
```

鉴于 LLM-as-a-judge 在评估中的有效性——这催生了许多其他评估，如 AlpacaEval [@dubois2024length]、Arena-Hard [@li2024crowdsourced] 和 WildBench [@lin2024wildbench]——许多人在创建和使用偏好数据时开始使用 LLM-as-a-judge 而不是奖励模型。

围绕如何使用所谓的"生成式奖励模型" [@mahan2024generative]
[@zhang2024generative] [@ankner2024critique]（包括专门训练为有效评判者的模型 [@kim2023prometheus]），已经催生了一个完整的研究领域，但在 RM 评估上它们往往落后于现有的奖励模型，这表明奖励建模是当前 RLHF 的重要技术。

提高 LLM-as-a-judge 工作流鲁棒性的一个常见技巧是使用采样温度为 0 以降低评分的方差。

## 延伸阅读

奖励建模的学术文献在 2024 年确立。
奖励建模早期的大部分进展集中在建立基准和识别行为模式。
第一个 RM 基准 RewardBench 为测试奖励模型提供了通用基础设施 [@lambert2024rewardbench]。
此后，RM 评估已扩展到与通用后训练模型可用的评估类型相似，其中一些评估测试在具有已知正确答案的领域上的预测准确性 [@lambert2024rewardbench]，或那些更类似于用 LLM-as-a-judge 进行"感觉"或与其他基准的相关性的评估 [@wen2024rethinking]。

新基准的例子包括：

- **纯文本（通用聊天/偏好）：** RMB [@zhou2024rmb]、RewardBench2 [@malik2025rewardbench]、偏好代理评估 [@frick2024evaluate] 或 RM-Bench [@liu2024rm]。
- **专门纯文本（数学等）：** 多语言奖励基准（M-RewardBench）[@gureja2024m]、用于检索增强生成的 RAG-RewardBench（RAG）[@jin2024rag]、用于拼写错误的 ReWordBench [@wu2025rewordbench]、RewardMATH [@kim2024evaluating] 或 AceMath-RewardBench [@liu2024acemath]。
- **过程 RM：** PRM Bench [@song2025prmbench] 或 ProcessBench [@zheng2024processbench] 以及视觉基准 VisualProcessBench [@wang2025visualprm] 或 ViLBench [@tu2025vilbench]。
- **智能体 RM：** Agent-RewardBench [@men2025agentrewardbench] 或 CUARewardBench [@lin2025cuarewardbench]。
- **多模态：** MJ-Bench [@chen2024mj]、Multimodal RewardBench [@yasunaga2025multimodal]、VL RewardBench [@li2024vlrewardbench] 或 VLRMBench [@ruan2025vlrmbench]。

要理解*训练*奖励模型的进展，可以参考新的奖励模型训练方法，包括方面条件模型 [@wang2024interpretable]、高质量人类数据集 [@wang2024helpsteer2] [@wang2024helpsteer2p]、缩放实验 [@adler2024nemotron]、广泛实验 [@touvron2023llama] 或去偏数据 [@park2024offsetbias]。

## 建议的实验

配套代码仓库在 `code/reward_models/` 中包含小型奖励模型训练脚本。
这些旨在作为学习练习，而非调优的参考方案。
从干净的 `code/` 环境开始，使用 `uv sync`，然后一次运行一个实验。

1. **在 UltraFeedback 上训练 Bradley-Terry 偏好奖励模型。**
   运行：

   ```bash
   cd code/
   uv run python -m reward_models.train_preference_rm --samples 2000 --epochs 1
   ```

   观察选中和被拒绝回答之间的奖励边际是否在演示和 W&B 日志中增长。
   然后改变 `--samples`、`--lr` 和 `--model-id` 来查看信号何时变得嘈杂或不稳定。

2. **比较结果监督和过程监督。**
   运行 GSM8K 结果奖励模型和 PRM800K 过程奖励模型：

   ```bash
   cd code/
   uv run python -m reward_models.train_orm --samples 400 --epochs 2
   uv run python -m reward_models.train_prm --samples 500 --epochs 2
   ```

   比较每种模型在训练后可以评分的内容：ORM 应该区分正确和错误的最终答案，而 PRM 应该跨中间推理步骤分配分数。
   这是序列级、结果级和过程级监督之间区别的实践版本。

3. **添加一个小型留出奖励模型评估。**
   一个有用的贡献是为 `reward_models/` 添加一个 50 到 200 个示例的评估，报告准确率或偏好对排序，无需完整的训练运行。
   将评估保持足够小，以便在调优超参数时使用。
