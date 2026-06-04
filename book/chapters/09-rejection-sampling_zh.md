<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "直接对齐算法"
prev-url: "08-direct-alignment"
page-title: 拒绝采样
search-title: "第 9 章：拒绝采样"
meta-description: "使用奖励或偏好信号改进后训练语言模型的拒绝采样和 best-of-n 方法。"
next-chapter: "偏好的本质"
next-url: "10-preferences"
lectures:
  - video: "https://www.youtube.com/watch?v=4gIwiSPmQkU&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=3"
    label: "第 2 讲：IFT、奖励建模、拒绝采样（第 4、5 和 9 章）"
---

# 拒绝采样

拒绝采样（Rejection Sampling, RS）是偏好微调中使用最广泛但文档化最少的方法之一。
许多突出的 RLHF 论文将其作为训练流程的核心组成部分，然而对其为何如此有效却没有经典的实现或解释。
RS 可以在训练流程中的多个点应用——在指令微调之后、在基于 RL 的优化之后，甚至在 RLVR 之后——使其成为一个多功能但难以定位的工具。
结合其文档化不足的性质，这就是它出现在核心优化方法末尾的原因。

拒绝采样通过筛选新的候选补全、基于训练好的奖励模型过滤它们，然后仅在顶级补全上（与指令微调相同的损失函数）微调原始模型来运作。

这个名称源自计算统计学 [@gilks1992adaptive]，其中人们希望从复杂分布中采样，但没有直接的方法来做到这一点。
为了缓解这一点，人们从更容易建模的分布中采样，并使用启发式方法检查样本是否可接受。
对于语言模型，目标分布是对提示的高质量补全，过滤器是一个奖励模型，采样分布是当前模型。

WebGPT [@nakano2021webgpt]、Anthropic 的 Helpful and Harmless 智能体 [@bai2022training]、OpenAI 关于过程奖励模型的流行论文 [@lightman2023let]、Llama 2 Chat 模型 [@touvron2023llama] 和其他开创性工作都使用此基线；更近的工作直接形式化了它（例如 RAFT [@dong2023raft] 用于将其应用于多模态对齐，以及统计拒绝采样优化（RSO）[@liu2023statistical] 提供了拒绝采样如何与其他偏好学习目标相关联的原则性概述）。

*在本章中，我们使用 $x$ 表示提示，$y$ 表示补全。这种表示法在语言模型文献中很常见，其中方法操作完整的提示-补全对，而不是单个 token。*

## 训练过程，逐步说明

拒绝采样整体遵循几个阶段。

0. **提示和奖励模型选择**：首先，您必须选择要训练的提示，相对于其他训练阶段。最简单的方法是重复使用第一个 SFT/IFT 阶段的每个提示，但这可能导致一些过拟合。在进行拒绝采样之前，您还必须训练好一个奖励模型（更多信息见第 5 章）。
1. **从起始检查点生成补全**：接下来，必须使用想要优化的模型为所选提示生成补全。这可能涉及调整许多设置，如采样温度、top-p、最大序列长度、每个提示的补全数等。
2. **使用奖励模型选择顶级补全**：所有补全由奖励模型排名。此阶段可能还包括去重，对每个提示只保留一个补全，尽管许多这样的设计选择归结为经验性的消融研究。
3. **在顶级补全上进行 SFT**：完成拒绝采样后，对起始检查点在所选补全上进行指令微调。

拒绝采样过程的视觉概览包含在下方 @fig:rs-overview 中。

![拒绝采样概览。](images/rejection-sampling.png){#fig:rs-overview}

关于使用哪些提示、如何选择奖励模型、如何排序拒绝采样等的实际细节在文献中没有很好的文档记录。
本章提供了方法的概述，将进一步的实验留给读者。

### 生成补全

要生成每个提示的多个候选补全集，我们将 $M$ 个提示定义为一个向量：

$$X = [x_1, x_2, ..., x_M]$$ {#eq:rs_prompt_vector}

这些提示可以来自许多来源，但最常见的是来自指令训练集。

对于每个提示 $x_i$，我们生成 $N$ 个补全。我们可以将其表示为一个矩阵：

$$Y = \begin{bmatrix}
y_{1,1} & y_{1,2} & \cdots & y_{1,N} \\
y_{2,1} & y_{2,2} & \cdots & y_{2,N} \\
\vdots & \vdots & \ddots & \vdots \\
y_{M,1} & y_{M,2} & \cdots & y_{M,N}
\end{bmatrix}$$ {#eq:rs_completion_matrix}

其中 $y_{i,j}$ 表示第 $i$ 个提示的第 $j$ 个补全。
每一行 $i$ 对应一个单一的提示 $x_i$ 并包含其 $N$ 个候选补全；每一列 $j$ 对应所有提示中的第 $j$ 个采样补全。

### 评分补全

现在，我们将所有这些提示-补全对通过奖励模型，以获得奖励矩阵。
我们将奖励表示为一个矩阵 $R$：

$$R = \begin{bmatrix}
r_{1,1} & r_{1,2} & \cdots & r_{1,N} \\
r_{2,1} & r_{2,2} & \cdots & r_{2,N} \\
\vdots & \vdots & \ddots & \vdots \\
r_{M,1} & r_{M,2} & \cdots & r_{M,N}
\end{bmatrix}$$ {#eq:rs_reward_matrix}

每个奖励 $r_{i,j}$ 通过将补全 $y_{i,j}$ 及其对应的提示 $x_i$ 通过奖励模型 $\mathcal{R}$ 来计算：

$$r_{i,j} = \mathcal{R}(y_{i,j} \mid x_i)$$ {#eq:rs_reward_computation}

有多种方法可以选择用于训练的顶级补全。

为了形式化基于奖励矩阵选择最佳补全的过程，我们可以定义一个在奖励矩阵 $R$ 上操作的选择函数 $S$。

#### 每个提示的顶级

第一个潜在的选择函数取每个提示的最大奖励。

$$S(R) = \left[\arg\max_{j} r_{1,j}, \arg\max_{j} r_{2,j}, ..., \arg\max_{j} r_{M,j}\right]$$ {#eq:rs_selection_per_prompt}

此函数 $S$ 返回一个索引向量，其中每个索引对应于 $R$ 中每行具有最大奖励的列。
然后我们可以使用这些索引来选择我们选中的补全：

$$Y_{chosen} = [y_{1,S(R)_1}, y_{2,S(R)_2}, ..., y_{M,S(R)_M}]$$ {#eq:rs_chosen_completions}

#### 总体顶级对

或者，我们可以从整个集合中选择顶级的 $K$ 个提示-补全对。
首先，让我们将奖励矩阵 $R$ 展平为单个向量：

$$R_{flat} = [r_{1,1}, r_{1,2}, ..., r_{1,N}, r_{2,1}, r_{2,2}, ..., r_{2,N}, ..., r_{M,1}, r_{M,2}, ..., r_{M,N}]$$ {#eq:rs_flattened_rewards}

此 $R_{flat}$ 向量长度为 $M \times N$，其中 $M$ 是提示数，$N$ 是每个提示的补全数。

现在，我们可以定义一个选择函数 $S_K$，选择 $R_{flat}$ 中 K 个最高值的索引：

$$S_K(R_{flat}) = \text{argsort}(R_{flat})[-K:]$$ {#eq:rs_topk_selection}

其中 $\text{argsort}$ 返回按升序排列数组的索引，我们取最后 $K$ 个索引以获得 $K$ 个最高值。

为了获取我们选择的补全，需要将这些展平索引映射回原始补全矩阵 $Y$。
要恢复相应的提示-补全对，可以通过 $i = \lfloor k / N \rfloor + 1$ 和 $j = (k \bmod N) + 1$ 将零索引展平索引 $k$ 映射到 $(i,j)$。

#### 选择示例

考虑以下情况，我们有五个提示和四个补全。
我们将展示两种基于奖励选择补全的方法。

$$R = \begin{bmatrix}
0.7 & 0.3 & 0.5 & 0.2 \\
0.4 & 0.8 & 0.6 & 0.5 \\
0.9 & 0.3 & 0.4 & 0.7 \\
0.2 & 0.5 & 0.8 & 0.6 \\
0.5 & 0.4 & 0.3 & 0.6
\end{bmatrix}$$ {#eq:rs_example_matrix}

首先，**每个提示**。直观上，我们可以如下高亮奖励矩阵：

$$R = \begin{bmatrix}
\textbf{0.7} & 0.3 & 0.5 & 0.2 \\
0.4 & \textbf{0.8} & 0.6 & 0.5 \\
\textbf{0.9} & 0.3 & 0.4 & 0.7 \\
0.2 & 0.5 & \textbf{0.8} & 0.6 \\
0.5 & 0.4 & 0.3 & \textbf{0.6}
\end{bmatrix}$$ {#eq:rs_example_per_prompt}

使用 argmax 方法，为每个提示选择最佳补全：

$$S(R) = \left[\arg\max_{j} r_{i,j} \text{ for } i \in [1,5]\right]$$ {#eq:rs_example_selection_formula}

$$S(R) = [1, 2, 1, 3, 4]$$ {#eq:rs_example_selection_result}

这意味着我们将选择：

- 提示 1：补全 1（奖励 0.7）
- 提示 2：补全 2（奖励 0.8）
- 提示 3：补全 1（奖励 0.9）
- 提示 4：补全 3（奖励 0.8）
- 提示 5：补全 4（奖励 0.6）

现在，**总体最优**。
让我们高亮顶级的五个总体补全对。

$$R = \begin{bmatrix}
\textbf{0.7} & 0.3 & 0.5 & 0.2 \\
0.4 & \textbf{0.8} & 0.6 & 0.5 \\
\textbf{0.9} & 0.3 & 0.4 & \textbf{0.7} \\
0.2 & 0.5 & \textbf{0.8} & 0.6 \\
0.5 & 0.4 & 0.3 & 0.6
\end{bmatrix}$$ {#eq:rs_example_top_overall}

首先，我们展平奖励矩阵：

$$R_{flat} = [0.7, 0.3, 0.5, 0.2, 0.4, 0.8, 0.6, 0.5, 0.9, 0.3, 0.4, 0.7, 0.2, 0.5, 0.8, 0.6, 0.5, 0.4, 0.3, 0.6]$$ {#eq:rs_example_flattened}

现在，我们选择五个最高值的索引：
$$S_5(R_{flat}) = [8, 5, 14, 0, 11]$$ {#eq:rs_example_topk_result}

将这些映射回原始矩阵：

- 索引 8 → 提示 3，补全 1（奖励 0.9）
- 索引 5 → 提示 2，补全 2（奖励 0.8）
- 索引 14 → 提示 4，补全 3（奖励 0.8）
- 索引 0 → 提示 1，补全 1（奖励 0.7）
- 索引 11 → 提示 3，补全 4（奖励 0.7）

#### 实现示例

以下是一个展示如何实现选择方法的代码片段。

```python
import numpy as np

x = np.random.randint(10, size=10)
print(f"{x=}")
sorted_indices = np.argsort(x)
x_sorted = x[sorted_indices]
print(f"{x_sorted=}")

# 恢复原始数组的第一种方法
i_rev = np.zeros(10, dtype=int)
i_rev[sorted_indices] = np.arange(10)
np.allclose(x, x_sorted[i_rev])

# 恢复原始数组的第二种方法
np.allclose(x, x_sorted[np.argsort(sorted_indices)])
```

### 微调

使用所选补全，然后在模型的当前版本上执行标准指令微调。
更多细节可以在[指令微调章节](https://rlhfbook.com/c/04-instruction-tuning)中找到。

## 实现细节

执行此训练的核心超参数非常直观：

- **采样参数**：拒绝采样直接依赖于从模型接收的补全。拒绝采样的常见设置包括高于零的温度，例如在 0.7 到 1.0 之间，并对诸如 top-p 或 top-k 采样等参数进行其他修改。
- **每个提示的补全数**：拒绝采样的成功实现对每个提示使用了 10 到 30 或更多的补全。使用太少的补全将使训练变得有偏和/或嘈杂。
- **指令微调细节**：关于拒绝采样期间的指令微调，没有公布明确的训练细节。很可能使用与模型初始指令微调阶段略有不同的设置。
- **异构模型生成**：一些拒绝采样的实现包括来自多个模型的生成，而不仅仅是即将训练的当前模型。如何做到这一点的最佳实践尚未建立。
- **奖励模型训练**：使用的奖励模型将极大地影响最终结果。有关奖励模型训练的更多资源，请参阅[相关章节](https://rlhfbook.com/c/05-reward-models)。

在进行批量奖励模型推理时，可以按长度对分词的补全进行排序，使批次具有相似的长度。
这消除了对太多填充 token 运行推理的必要性，并将以少量实现复杂性换取吞吐量的提高。

## 相关：Best-of-N 采样

Best-of-N（BoN）是拒绝采样的近亲，遵循相同生成和评分的流程，但你**不对**所选补全微调模型。
相反，BoN 在推理时计算对静态提示（或提示集）的最佳可能补全，相关技术通常用于聊天模型的"Pro"级别，花费额外计算来获取你的查询答案。

Best-of-N 采样通常被作为相对于 RLHF 训练方法的基线包括在内。
重要的是要记住，BoN *不修改*底层模型，而是一种采样技术。
因此，将 BoN 采样与在线训练方法（如 PPO）进行比较在某些上下文中仍然有效。
例如，你仍然可以在运行 BoN 采样时测量相对于任何其他策略的 KL 距离。

这里，我们将展示当在一个提示上使用简单的 BoN 采样时，上述两种选择标准是等价的。

令 $R$ 是我们具有 $N$ 个补全的单个提示的奖励向量：

$$R = [r_1, r_2, ..., r_N]$$ {#eq:rewards_vector}

其中 $r_j$ 表示第 j 个补全的奖励。

使用 argmax 方法，我们为提示选择最佳补全：

$$S(R) = \arg\max_{j \in [1,N]} r_j$$ {#eq:selection_function}

使用 $K=1$ 的 top-K 方法退化为相同的方法，这是常见做法。

## 建议的实验

`code/rejection_sampling/` 中的配套实现运行完整的 GSM8K 拒绝采样流水线：生成 rollout、使用奖励模型评分、选择训练子集、微调并评估精确匹配准确率。
四个配置安排为匹配的处理/控制对，因此读者可以询问奖励模型是否真的有帮助。

1. **一次性构建 rollout 缓存。**

   ```bash
   cd code/
   uv run python -m rejection_sampling.preprocess \
       --config rejection_sampling/configs/top_per_prompt.yaml
   ```

   这为共享的 GSM8K 切片生成并评分补全。
   后续的训练配置只要生成和评分设置保持不变就可以重用缓存。

2. **比较奖励选择与随机控制。**

   ```bash
   cd code/
   uv run python -m rejection_sampling.train \
       --config rejection_sampling/configs/top_per_prompt.yaml
   uv run python -m rejection_sampling.train \
       --config rejection_sampling/configs/random_per_prompt.yaml
   uv run python -m rejection_sampling.train \
       --config rejection_sampling/configs/top_k_overall.yaml
   uv run python -m rejection_sampling.train \
       --config rejection_sampling/configs/random_k_overall.yaml
   ```

   以成对方式阅读结果：`top_per_prompt` 对比 `random_per_prompt`，以及 `top_k_overall` 对比 `random_k_overall`。
   如果奖励选择的运行没有击败其随机基线，那么奖励模型或采样的补全未在该切片上提供有用的信号。

3. **改变奖励模型获得的选择空间。**
   复制一个配置并更改 `num_completions_per_prompt`、`temperature`、`top_p` 和 `selection.top_k`。
   更多的补全可以改善可用的最佳样本，但只有在奖励模型能够区分好答案和坏答案时才行。

4. **尝试较小的策略模型。**
   将 `model_name` 设置为较小的兼容指令模型，减少 `max_train_samples`，并重新运行相同的成对比较。
   这使实验成本更低，并突出显示拒绝采样是在挽救弱生成还是在从已经很好的生成中选择。
