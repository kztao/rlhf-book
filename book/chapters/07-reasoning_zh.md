<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "强化学习"
prev-url: "06-policy-gradients"
page-title: 推理与推理时扩展
search-title: "第 7 章：推理与推理时扩展"
meta-description: "后训练中的推理训练和推理时扩展，包括 RLVR 和思考模型。"
next-chapter: "直接对齐算法"
next-url: "08-direct-alignment"
---

# 推理与推理时扩展

推理模型和推理时扩展在 2024 年底、贯穿 2025 年并延伸至未来，使语言模型性能实现了巨大飞跃。
推理时扩展是通过在生成期间使用更多计算来提高模型性能的能力，例如生成更长的推理链或采样多个回答。
被训练为在回答之前进行大量思考的语言模型非常好地利用了此特性。
这些模型使用大量可验证奖励的强化学习（RLVR）进行训练 [@lambert2024t]，仍然使用大量的 RLHF。
在本章中，我们回顾引领 AI 社区转变对 RL 在语言模型中潜力认识的路径，回顾 RLVR 的基础知识，强调关键工作，并指出将在未来几年定义该领域的未来辩论。

## RLVR 的角色

首先，在 2016 年神经信息处理系统（NeurIPS）会议上，Yann LeCun 首次介绍了他现在著名的蛋糕比喻，说明现代机器学习系统中的学习发生在哪里：

> 如果智能是一个蛋糕，蛋糕的主体是无监督学习，蛋糕上的糖霜是监督学习，而蛋糕上的樱桃是强化学习（RL）。

这个类比在现代语言模型和后训练栈的最新变化中已经基本完整。
RLHF 是这一点的前驱，而用于推理模型（主要在数学、代码和科学主题上）的 RL 则是对此的确认。
在这个类比中：

- 在大量互联网数据上的自监督学习构成了蛋糕的大部分（尤其是在以 FLOPs 花费的计算量来看），
- 后训练的开始——用于指令的监督微调（SFT）——将模型调优到更窄的分布，以及
- 最后，"纯"强化学习（RL）是顶部的樱桃。用于创建新的"推理"或"思考"模型的规模化强化学习就是这个完成部分（在 RLHF 的帮助下，RLHF 不算经典 RL，正如我们将解释的）。

这一小部分推理训练随着**思考模型**的出现而出现，这些模型结合使用本书讨论的后训练技术来对齐偏好，同时在可验证领域上进行 RL 训练，以大幅提升推理、编程和数学问题解决等能力。

这些模型的训练方法——可验证奖励的强化学习（RLVR）[@lambert2024t]——与 RLHF 非常相似，但它使奖励模型变得可选，取而代之的是一个评分函数，当答案正确时返回正奖励，否则返回 0。

例如，考虑为 RLHF 评分与为 RLVR 评分有多么不同。
在 RLHF 中，奖励模型必须评估主观质量：

> **提示**：解释经济学中机会成本的概念。
>
> **回答**：机会成本是你在做决策时放弃的次优替代方案的价值。例如，如果你花一小时学习而不是工作，机会成本就是你本可以赚到的工资……

对这个回答评分需要判断清晰度、准确性、完整性和帮助性——所有这些质量都需要学到的偏好，并且缺乏确定的正确答案。

相比之下，RLVR 使用返回确定分数的验证函数。
对于数学：

> **提示**：小于 20 的所有质数之和是多少？
>
> **回答**：小于 20 的质数是 2, 3, 5, 7, 11, 13, 17 和 19。
> 相加：2 + 3 = 5，然后 5 + 5 = 10，然后 10 + 7 = 17，然后 17 + 11 = 28，然后 28 + 13 = 41，然后 41 + 17 = 58，最后 58 + 19 = 77。
> 答案是 $\boxed{77}$。
>
> **验证**：`extracted_answer == 77` → 奖励 = 1

`\boxed{}` 表示法是从数学排版中采用的习惯，使答案提取变得简单——一个简单的正则表达式就可以从回答中提取最终答案，无论模型是如何得出它的。
请注意，存在其他答案提取方法，例如仅使用短语"The answer is: "如上所示，特殊 token 如 `<answer>`，或分隔符如 `####`。

对于代码生成，验证通常采取单元测试的形式：

> **提示**：编写一个 Python 函数 `fib(n)`，返回第 n 个斐波那契数，其中 fib(0) = 0 且 fib(1) = 1。
>
> **回答**：
> def fib(n):
>     if n < 2:
>         return n
>     return fib(n - 1) + fib(n - 2)
>
> **验证（单元测试）**：
>
> assert fib(0) == 0   # 基本情况
> assert fib(1) == 1   # 基本情况
> assert fib(10) == 55 # 较大值
> （所有测试通过 → 奖励 = 1）

单元测试是代码的自然验证函数：它们根据已知的输入-输出对执行模型的解决方案。
一种常见的评分形式是执行简单的门控：如果所有断言通过，奖励为 1；如果任何断言失败，奖励为 0。
其他设置使用与通过测试数量成比例的部分分数。
对于这两个例子，不需要学习的奖励模型，并且大多数设置中没有使用（因为模型在这些领域对过度优化是鲁棒的），但可以使用奖励的线性组合。

RLVR 背后的思想对 RL 文献来说并不新鲜，基于答案是否正确来采取梯度更新的核心思想几乎是强化学习的教科书定义。
将其应用于语言模型时的创新主要在于如何在保持被微调模型的强大通用能力的同时应用它。在此之中，语言建模文献中有许多相关的思想，其中模型从关于答案正确性的反馈中学习。

最初，在我参与并提出术语"可验证奖励的 RL（RLVR）"的工作中 [@lambert2024t]，该方法被命名为"带真实答案奖励的 RL（RLGT）"。
然而 RLVR 与仅从真实答案中学习有微妙的不同。
在数学等领域中，单一的真实答案可用于验证解决方案，正如我们上面看到的。
在其他领域中，如代码生成或精确的指令遵循，答案可以通过检查函数（例如单元测试）进行验证，即使存在多个正确解决方案而非仅一个真实答案。
RLVR 进展的核心是拥有这些可验证问题的多样性和深度，即使确切的解决方案事先未知。

![RLVR 在 RL 反馈循环中的形式。使用验证函数而非奖励模型。](images/rlvr-system.png){#fig:rlvr}

首先成功部署此类训练的模型是 OpenAI 的 o1 [@openai2024o1] 和开源权重的模型 DeepSeek R1 [@guo2025deepseek]。
此后不久，整个 AI 行业优先考虑了这个训练过程和模型风格。
这里的核心变化更多的是训练阶段的重新分配和不同行为优先级的调整，而非这种 RL 设置是全新的。
推理模型带来了一个规模化 RL 训练成为预期的时代。

至于这些模型表现出的行为类型，考虑以下 DeepSeek V3 0325 与其推理模型 DeepSeek R1 在查询 `Write me a short poem about a goldfish` 上的例子。
DeepSeek V3 非常直截了当：

> **回答**：Goldfish Dreams
>
> In a bowl of glass so clear and round,
> A tiny goldfish swims without a sound.
> With scales that gleam like sunlight's kiss,
> He drifts along in watery bliss.
>
> A flick of fin, a flash so bright,
> Chasing bubbles in the light.
> Does he dream of oceans wide?
> Or is his little world enough---his pride?

与此同时，DeepSeek R1 的体验不同。
它首先生成大量的思考 token 来精心制作回答，然后才响应用户。
答案如下所示：

> **回答**：`<thinking>` Okay, the user wants a short poem about a goldfish. Let me start by brainstorming some goldfish-related imagery. Goldfish are often in bowls, right? So maybe start with the setting. Words like "gilded bowl" or "crystal globe" could work.
>
> Next, think about the movement of the goldfish. They swim smoothly, so words like "dart", "glide", or "swirl" come to mind. Maybe describe their fins as "silken" to emphasize delicacy.
>
> Then, consider the environment around the goldfish. Sunlight through water creates shimmering effects. Words like "sunbeams dance" or "ripples spin" might capture that. Including elements like castles or pebbles adds a playful touch.
>
> Themes? Maybe the contrast between the goldfish's small world and its perception of freedom. The bowl as a universe, tiny but complete. Ending with a sense of mystery or silent observation could give depth. Need to keep it concise, maybe four stanzas with a rhyme scheme. Let me check the rhythm and flow. Avoid clichés but keep it accessible. Alright, let's piece it together. `</thinking>`
>
> Gilded Dreams in Liquid Glass
>
> A gilded bowl, a crystal globe,
> Where tiny tides in silence flow---
> A flash of orange, a whispered robe,
> You dart through sunlit indigo.
>
> Your world is small, yet boundless too---
> Each pebble, castle, weed unfurled.
> The surface bends the sky to blue,
> And ripples spin your liquid world.
>
> You glide through myths we'll never know,
> A silent monarch, veiled in mist.
> Do dreams drift where the bubbles go?
> Or memories in amethyst?
>
> We watch you trace your endless sphere---
> A universe within a tear.

`<thinking>` 标签中的 token 是模型的推理。
对于更复杂的问题，推理阶段可以在产生答案之前花费数千个 token。
因此，长上下文语言模型是高级推理行为的先决条件，但这不是本章的重点。

关于*这种训练如何工作*的核心直觉是，对于给定模型，我们重复以下循环：

1. 为多个问题采样多个答案，
2. 向正确的答案采取梯度步骤，以及
3. 重复，重新访问相同的数据。

值得注意的是，这种极其简单的方法（当使用仔细的数据分布和稳定的训练基础设施进行时）通过反复访问相同的问题来帮助模型学习。
更值得注意的是，在这些训练问题上的改进泛化到了模型从未见过的问题和（某些）领域！

这种简单的方法允许模型在行为空间上进行轻度搜索，RL 算法增加与正确答案相关的行为的似然。

## 新推理模型的起源

这里我们详述导致 2025 年推理模型爆发的高层趋势。

### 为什么 RL 现在有效？

尽管有许多"RL 还不有效"的观点 [@irpan2018deep] 和详细说明 RL 深度可复现性问题的论文 [@henderson2018deep]，该领域克服了这些问题并找到了高影响力的应用。
本书涵盖了一些，如 ChatGPT 的 RLHF 和 DeepSeek R1 的 RLVR，但还有许多其他应用，包括改进芯片设计 [@mirhoseini2020chip]、掌握视频游戏 [@schrittwieser2020mastering]、自动驾驶 [@cusumano2025robust] 等等。
语言模型上 RL 聚焦训练的起飞表明在研究领域的许多基本问题上取得了进展，包括：

- **RL 的稳定性可以解决**：在其存在的整个历史中，RL 采用的限制因素一直是稳定性。这表现在两个方面。首先，学习本身可能是变化无常的，并不总是有效。其次，训练本身被知道比标准语言模型训练更脆弱，更容易发生损失尖峰、崩溃等。无数的新模型发布正在使用这种在预训练基础模型之上的带可验证奖励的 RL 训练风格，并且已经发生了大量的学术采纳。RL 的技术门槛处于历史最低点。

- **开源版本已经"存在"**：已经存在许多使用 RLVR 及相关技术训练语言模型的工具。
例子包括 TRL [@vonwerra2022trl]、Open Instruct [@lambert2024t]、veRL [@sheng2024hybridflow] 和 OpenRLHF [@hu2024openrlhf]，其中许多都建立在 RLHF 和后训练早期的优化之上。工具的可访问性正在推动大量且加速的研究。

多个资源指出，用于推理的 RL 训练仅在 2024 年左右推出的领先模型上才变得可行，这表明在推理训练成为可能之前，模型需要具备一定水平的底层能力。

### RL 训练与推理时扩展

使用强化学习训练来激发推理行为和可验证领域上的性能，与推理时扩展的思想密切相关。
推理时扩展（也称为测试时扩展）是在推理时使用更多计算能力以在下游任务上表现更好的通用方法类别。
推理时扩展的方法在 DeepSeek R1 和 OpenAI 的 o1 发布之前就已经被研究，这两者都大规模推广了在 RL 训练上的投资。
例子包括价值引导采样 [@liu2023don] 或带答案提取的重复随机采样 [@brown2024large]。
除此之外，推理时扩展可以用于改进更多超出思维链推理的 AI 训练方法以解决问题，例如使用深入考虑选项的奖励模型 [@ankner2024critique] [@liu2025inference]。

RL 训练是使用推理时缩放法则的捷径，但从长远来看，我们将有更多方法来激发我们需要的推理时权衡以获得最佳性能。
大量使用 RL 训练模型通常使它们能够在每个回答中生成更多 token，其方式与改进的下游性能强相关（尽管序列长度增加是默认行为，但也存在明确研究如何在不依赖此推理时扩展的情况下提高性能的研究）。
这是从早期 RLHF 系统中看到的长序列偏差的重大转变 [@singhal2023long]，在早期 RLHF 系统中，人类偏好训练有一个副作用，即增加平均回答长度以获得偏好排名上的边际收益。

除了核心的 RL 训练模型外，还有许多正在探索的方法来继续推动推理和推理时计算的极限。
这些在很大程度上超出本书的范围，因为它们的快速发展性质，但它们包括通过指令微调将推理行为从更大的 RL 训练模型蒸馏到更小的模型 [@muennighoff2025s1]、组合更多的推理调用 [@chen2024more] 等。
这里重要的是下游性能与生成 token 数量增加之间的相关性——否则只是浪费能量。

### RLVR 的未来（超越推理）

在许多领域，这些新的 RLVR 方法由于专注于性能而非行为，更加符合开发者的目标。
标准微调 API 通常使用参数高效微调方法，如 LoRA（低秩适应，一种仅训练小型附加矩阵而非所有权重的方法，也称为参数高效微调 PEFT），结合指令的监督微调。
开发者传入提示和补全，模型通过更新参数以匹配补全来进行调优，这增加了数据中的特征在模型生成中出现的普遍性。

RLVR 专注于匹配答案。
给定查询和正确答案，RLVR 帮助模型学习产生正确答案。
而标准指令微调对数据进行 1 或 2 个 epoch 的损失更新，RLVR 得名于对相同的少数数据点进行数百或数千个 epoch 的训练，以给模型时间来学习新行为。
这可以被视为将基础模型版本中表现稀少的正面行为强化为 RLVR 之后的鲁棒行为。

**语言模型的 RL 训练范围继续增长**：从 o1 和 R1 中获得的最重要的基础科学启示是，我们有更多方法来训练语言模型以获得潜在有价值的行为。
向研究人员和工程师开放的门越多，我们就应该对 AI 的总体轨迹越乐观。


## 理解推理训练方法

对推理的投资促使模型如何被训练以遵循人类指令的技艺发生了重大演变。
这些方案仍然使用前面章节讨论的常见组件（如第 3 章中概述 DeepSeek R1 方案时讨论的），包括指令微调、基于人类反馈的强化学习和可验证奖励的强化学习（RLVR）。
核心变化是使用了更多的 RLVR，并以不同的顺序应用其他训练技术——对于推理模型来说，传统的核心训练步骤要么是大规模 RL 运行，要么是对另一个经过大量 RLVR 训练的模型的*输出*进行大规模指令微调（称为蒸馏）。

### OpenAI o1 或 DeepSeek R1 之前的推理研究

在推理模型起飞之前，大量努力致力于理解如何训练语言模型在可验证领域上变得更好。
以下这些工作与后面工作如 DeepSeek R1 中使用的方法的主要区别在于，它们的方法论没有扩展到相同的水平，或者它们产生的模型在牺牲整体性能以换取更高的数学或编程能力。
底层的想法和动机被包含在内，以描绘推理模型如何在领域中出现的更广阔图景。

训练语言模型在可验证领域上的一些最早努力包括自学习推理器（STaR）系列工作 [@zelikman2022star] [@Zelikman2024QuietSTaRLM] 和 TRICE [@hoffman2023training]，两者都在 2022 年和 2023 年间使用真实答案奖励信号来鼓励模型中的思维链推理。
STaR 有效地近似了策略梯度算法，但在实践中以不同方式过滤样本，并使用交叉熵度量而不是对数概率，而 Quiet-STaR 通过让模型在尝试回答可验证问题之前生成 token（这有助于训练性能）来扩展非常相关的近期推理模型思想。
TRICE [@hoffman2023training] 也通过生成轨迹然后用自定义的马尔可夫链蒙特卡洛启发的期望最大化算法进行优化来改进推理。
VinePPO [@VinePPO] 紧随其后，使用了一种更接近现代推理模型的设置。
VinePPO 使用基于 PPO 的算法和数学问题正确性的二元奖励，在 GSM8K 和 MATH 上进行训练。
在 OpenAI o1 和 DeepSeek R1 之前的其他工作使用代码执行作为训练反馈信号 [@gehring2024rlefgroundingcodellms], [@xu2024dpo] 或用于定理证明的验证（在此称为来自验证器反馈的强化学习，RLVF）[@amit2024models]。
Tülu 3 通过使用简单的 PPO 训练器来奖励具有正确答案的补全来扩展这些方法——最重要的是，在广泛的评估套件上保持模型的整体性能。
Tülu 3 的二元奖励和现代推理训练技术可以与 STaR 的迭代方法或 Quiet-STaR 的对数似然奖励形成对比。

### 早期推理模型

DeepSeek R1 之后的基础推理研究报告摘要，其中一些附带开放数据和模型权重，显示在 @tbl:reasoning_list 中。

::: {.table-wrap}
| 日期 | 名称 | 简况 | 开放权重 | 开放数据 |
|-------------|----------------------------|-----------------------------------------------------------------------|--------------|-----------|
| 2025-01-22 | DeepSeek R1 [@guo2025deepseek] | 基于 RL 的 DeepSeek 升级，在数学和代码推理上取得大幅收益 | 是 | 否 |
| 2025-01-22 | Kimi 1.5 [@team2025kimi] | 在中英文数据上扩展 PPO/GRPO；AIME 数学成绩出色 | 否 | 否 |
| 2025-03-31 | Open-Reasoner-Zero [@hu2025openreasonerzero] | 基础模型 RL 的完全开源复现 | 是 | 是 |
| 2025-04-10 | Seed-Thinking 1.5 [@seed2025seed] | 字节跳动 RL 流水线，带动态 CoT 门控 | 是 | 否 |
| 2025-04-30 | Phi-4 Reasoning [@abdin2025phi4] | 14B 模型；精细 SFT→RL；擅长 STEM 推理 | 是 | 否 |
| 2025-05-02 | Llama-Nemotron [@bercovich2025llamanemotron] | 多尺寸"推理开关"模型 | 是 | 是 |
| 2025-05-12 | INTELLECT-2 [@primeintellectteam2025intellect2reasoningmodeltrained] | 首个公开文档化的全球去中心化 RL 训练运行 | 是 | 是 |
| 2025-05-12 | Xiaomi MiMo [@xia2025mimo] | 从预训练到后训练的端到端推理流水线 | 是 | 否 |
| 2025-05-14 | Qwen 3 [@yang2025qwen3] | 类似 R1 方案应用于新模型 | 是 | 否 |
| 2025-05-21 | Hunyuan-TurboS [@liu2025hunyuan] | Mamba-Transformer MoE，自适应长短 CoT | 否 | 否 |
| 2025-05-28 | Skywork OR-1 [@he2025skyworkor1] | 避免熵崩塌的 RL 方案；AIME 上超过 DeepSeek | 是 | 是 |
| 2025-06-04 | Xiaomi MiMo VL [@coreteam2025mimovltechnicalreport] | 将推理流水线端到端适配到多模态任务 | 是 | 否 |
| 2025-06-04 | OpenThoughts [@guha2025openthoughts] | 从 QwQ-32B 蒸馏的公开 120 万示例指令数据集 | 是 | 是 |
| 2025-06-10 | Magistral [@mistral2025magistral] | 在 Mistral 3 上的纯 RL；多语言 CoT；小型模型开源 | 是 | 否 |
| 2025-06-16 | MiniMax-M1 [@minimax2025minimax_m1] | 开源权重 456B MoE 混合/Lightning Attention 推理模型；1M 上下文；RL w/CISPO；发布 40K/80K 思考预算检查点 | 是 | 否 |
| 2025-07-10 | Kimi K2 [@kimiteam2025kimik2] | 1T MoE（32B 活跃）使用 MuonClip（QK-clip）保持稳定；15.5T token 预训练无损失尖峰；多阶段后训练含智能体数据合成 + 联合 RL；发布基础 + 后训练检查点 | 是 | 否 |
| 2025-07-28 | GLM-4.5 [@zeng2025glm45] | 开源权重 355B-A32B MoE "ARC" 模型，含思考/非思考模式；23T token 多阶段训练 + 后训练含专家迭代和 RL；发布 GLM-4.5 + GLM-4.5-Air（MIT） | 是 | 否 |
| 2025-08-20 | Nemotron Nano 2 [@nvidia2025nemotronnano2] | 混合 Mamba-Transformer 用于长"思考轨迹"；20T token FP8 预训练后压缩/蒸馏；明确发布多个检查点加"大部分"预/后训练数据集 | 是 | 是（大部分）|
| 2025-09-09 | K2-Think [@llm3602025k2think] | 参数高效数学推理系统：32B 开源权重模型带测试时缩放方案；定位为完全开源含训练数据/代码（根据发布材料）| 是 | 是 |
| 2025-09-23 | LongCat-Flash-Thinking [@mlcteam2025longcat] | 560B MoE 推理模型；报告明确说明了从长 CoT 冷启动到大规模 RL 的分阶段方案；开源发布 | 是 | 否 |
| 2025-10-21 | Ring-1T [@ringteam2025everystepevolves] | 万亿规模"思考模型"，聚焦 RL 缩放；报告框架化 1T 规模 RL 缩放的瓶颈/解决方案并发布开源模型 | 是 | 否 |
| 2025-11-20 | OLMo 3 Think [@teamolmo2025olmo3] | 完全开放的"模型流程"发布：报告整个生命周期（阶段、检查点和数据点），并将 OLMo 3 Think 32B 定位为旗舰开放思考模型 | 是 | 是 |
| 2025-12-02 | DeepSeek V3.2 [@deepseekai2025v32] | 开源权重 MoE 前沿推进，报告突出注意力效率变化、RL 框架升级和智能体/推理性能的数据合成 | 是 | 否 |
| 2025-12-05 | K2-V2 [@liu2025k2] | 70B 密集"360-open"从头训练模型；带 3 级努力的纯 SFT 后训练用于可控思考 | 是 | 是 |
| 2025-12-15 | Nemotron 3 Nano [@nvidia2025nemotron3nano] | 30B-A3B MoE 混合 Mamba-Transformer；25T token 预训练并包括 SFT + 大规模 RL；明确声明发布权重 + 方案/代码 + 大部分训练数据 | 是 | 是（大部分）|
| 2025-12-16 | MiMo-V2-Flash [@mimo2025flash] | 309B MoE（15B 活跃）为速度优化：混合 SWA/GA 注意力（5:1, 128 token 窗口）+ 轻量 MTP；27T token FP8 预训练；后训练使用 MOPD + 大规模智能体 RL 用于推理/编程 | 是 | 否 |
表：2025 年显著的推理模型技术报告摘要，2025 年是使用 RLHF 进行实质性推理时扩展的第一年。{#tbl:reasoning_list}
:::

### 训练推理模型的常见实践

在本节中，我们详述在训练推理模型时用于排序训练阶段和修改数据以最大化性能的常见方法。

请注意，这些论文可能使用了列出的技术但未提及，而其他论文提到了，因此这些示例是已知实现的子集，应作为参考使用，而不是关于最佳方案是什么的最终宣言。

- **离线难度过滤**：RLVR 的核心直觉是，模型只能从存在梯度的示例中学习。如果 RLVR 的起始模型可以 100% 或 0% 地解决一个问题，则不同补全之间将没有梯度（即，所有策略对策略梯度算法看起来都相同）。许多模型在开始大规模 RL 之前使用难度过滤，将训练问题限制在起始模型仅解决 20-80% 的问题上。这些数据通过采样 N（例如 16）个训练集中每个提示的补全，并验证正确百分比来收集。Seed-Thinking 1.5、Open Reasoner Zero、Phi 4、INTELLECT-2、MiMo RL、Skywork OR-1 等都使用了这种形式。
- **每批次在线过滤**（或贯穿训练的难度课程）：作为离线过滤的补充，另一个主要问题是：在学习过程中问题应以什么顺序呈现给模型？为了解决这个问题，许多模型使用批次中间题的在线过滤、预构建的课程/数据调度器、将更难的问题保留到训练后期，或其他改进长期稳定性的想法。Kimi 1.5、Magistral、Llama-Nemotron、INTELLECT-2、MiMo-RL、Hunyuan-TurboS 等都使用了相关思想。
- **移除 KL 惩罚**：随着 RL 运行的时长（以任何度量：总 GPU 小时、FLOPs 或 RL 步数）相对于 RLHF 训练的增加，且奖励函数变得不太容易过度优化，许多模型移除了 KL 惩罚（该惩罚约束 RL 学习的策略与训练开始时的基础模型相似）。这允许模型在训练期间进一步探索。RAGEN [@wang2025ragenunderstandingselfevolutionllm]、Magistral、OpenReasonerZero、Skywork OR-1 等都使用了这一点。
- **放宽的策略梯度裁剪**：GRPO 算法的新变体，如 DAPO [@yu2025dapo]，提出了对 GRPO（或 PPO）中使用的双侧裁剪目标的修改，以启用更好的探索。裁剪也被证明在奖励不完美时可能导致虚假学习信号 [@shao2025spurious]。RAGEN、Magistral、INTELLECT-2 等都使用了这种不同梯度方向上不同范围的双侧裁剪。
- **离策略数据**（或完全异步更新）：随着使用 RL 解决任务所需的补全长度随着更难的问题急剧增加（特别是在回答长度的*方差*方面，经常有具有极长长度的异常值），RL 运行中的计算可能闲置。为了解决这个问题，训练正在转向异步更新或改变问题在批次中的排列方式以提高整体吞吐量。Seed-Thinking 1.5、INTELLECT-2 等都使用了部分到完全的异步（离策略）数据。
- **额外的格式奖励**：为了使推理过程可预测，许多模型添加了小奖励以确保模型遵循正确的格式，例如在回答之前使用 `<think>...</think>`。DeepSeek R1、OpenReasonerZero、Magistral、Skywork OR-1 等都使用了这一点。
- **语言一致性奖励**：类似于格式奖励，一些多语言推理模型使用语言一致性奖励来优先考虑在推理时不改变语言的模型（为了更好和更可预测的用户体验）。这些包括 DeepSeek R1、Magistral 等。
- **长度惩罚**：许多模型在 RL 训练期间使用不同形式的长度惩罚，以稳定学习过程或减轻在难题上的过度思考。一些例子包括 Kimi 1.5 逐步扩展目标长度以对抗过度思考（在难度课程中训练准确率高时）或 INTELLECT-2 贯穿始终运行小型长度惩罚。逐步扩展训练序列长度通过强制模型首先在更有限的思考预算内有效地进行领域推理，然后过渡到更长时间的训练（模型可以在更复杂的问题上高效使用这些行为），从而减轻过度思考。其他模型使用过长过滤和其他相关实现来提高吞吐量。
- **损失归一化**：围绕原始 GRPO 算法的每组归一化项可能引入的潜在长度或难度偏差，已有一些讨论（见策略梯度章节或 [@liu2025understanding]）。因此，一些模型，如 Magistral 或 MiMo，选择在批次级别而非组级别归一化损失或优势。
- **并行测试时计算扩展**：组合来自多个并行、独立采样的 rollout 的答案可以比使用单个 rollout 的答案带来显著改进。最朴素形式的并行测试时计算扩展，如 DeepSeek-R1、Phi-4 等中所做的，涉及使用多数 rollout 返回的答案作为最终答案。一种更高级的技术是使用训练好的评分模型从并行 rollout 的答案中选择最佳答案。截至 2026 年，此技术在开放、文档化的推理模型方案中尚未普及，但在 Claude 4 发布公告中被提及 [@anthropic2025claude4] 并在 DeepSeek-GRM 中使用 [@liu2025inference]。

补充这些常见技术，还有许多关于推理训练如何在不牺牲辅助能力的情况下创建有用模型的常见发现：

- **纯文本推理提升多模态性能**：Magistral、MiMo-VL 等发现，训练多模态模型然后在此多模态训练之后进行纯文本推理训练可以*提升*最终模型中的多模态性能。
- **通过系统提示可切换的推理**（或长度控制）：Llama-Nemotron、Nemotron Nano、Qwen 3、SmolLM 3 等使用特定的系统提示（可能与长度控制的 RL 训练结合 [@aggarwal2025l1]）为用户启用可切换的开/关思考长度。其他开源模型，如 OpenAI 的 GPT-OSS 和 LLM360 的 K2-V2 [@liu2025k2]，在系统提示中采用了低-中-高推理努力设置，但训练此类行为的方法文档化不够充分。

## 展望

推理模型领域的发展速度远超近期记忆中 AI 研究的任何领域，这里列出的一些常见实践将不可避免地被新技术取代。

正在进行多项系统性理解推理训练为何有效的努力。
OLMo 3 Think [@teamolmo2025olmo3] 代表了推理模型完整训练生命周期的最全面开放文档，提供了每个阶段的检查点和数据供研究社区研究，最终在 220 个 GPU 上进行了近 4 周的训练运行。
类似地，关于理解 RL 用于推理的缩放属性 [@khatri2025art] 的工作正在开始形式化计算、数据和性能之间的关系，这些关系以前只能由实践者直觉感知。

仍然清楚的是，强化学习已从蛋糕比喻中的"顶部樱桃"毕业为前沿模型训练的承重组件。
本章围绕 RLVR 思想的辅助技术——难度过滤、格式奖励等——不是最终答案，但它们代表了该领域当前对如何从语言模型中激发推理的最佳理解。
下一方法可能会看起来不同，但它们将建立在此处建立的基础上。
