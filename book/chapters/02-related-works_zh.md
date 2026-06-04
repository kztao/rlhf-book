<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "引言"
prev-url: "01-introduction"
page-title: RLHF 简史
search-title: "第 2 章：RLHF 简史"
meta-description: "RLHF、奖励建模、偏好学习和语言模型后训练背后的关键论文与历史里程碑。"
next-chapter: "训练概述"
next-url: "03-training-overview"
lectures:
  - video: "https://www.youtube.com/watch?v=o6l6tJQgUg4&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y&index=2"
    label: "第 1 讲：概述（第 1–3 章）"
---

# RLHF 简史

RLHF 及其相关方法非常新。
我们强调历史，是为了展示这些流程直到最近才被正式确立，以及有多少文档存在于学术文献中。
借此，我们想强调 RLHF 正在快速演进，因此本章为一本将表达某些方法的不确定性、并预期某些核心实践的细节可能发生变化 的书奠定基础。
此外，这里列出的论文和方法展示了为什么 RLHF 流程中的许多部分是这样设计的，因为一些开创性论文所针对的应用与现代语言模型完全不同。

在本章中，我们将详细介绍推动 RLHF 领域发展到今天的关键论文和项目。
这并非意在成为 RLHF 及相关领域的全面综述，而是一个起点，并讲述我们是如何走到今天的。
本章有意聚焦于近期的、推动了 ChatGPT 诞生的工作。
在 RL 文献中，关于从偏好中学习的进一步工作相当丰富 [@wirth2017survey]。
如需更详尽的列表，应使用正式的综述论文 [@kaufmann2023survey], [@casper2023open]。

![本章讨论的 RLHF 关键发展时间线，从早期的偏好强化学习到 RLHF 在大型语言模型中的应用。](images/rlhf_timeline.png){#fig:rlhf_timeline}

## 起源至 2018 年：基于偏好的强化学习

该领域随着深度强化学习的发展而最近流行起来，并已发展为对许多大型科技公司 LLM 应用的更广泛研究。
尽管如此，今天使用的许多技术与早期偏好强化学习文献中的核心技术密切相关。

最早具有类似现代 RLHF 方法的论文之一是 *TAMER*。
*TAMER: Training an Agent Manually via Evaluative Reinforcement* 提出了一种方法，其中人类迭代地为智能体的行为打分以学习奖励模型，然后用该模型来学习动作策略 [@knox2008tamer]。
其他同期或稍后的工作提出了一个演员-评论家算法 COACH，其中人类反馈（正面和负面）被用于调整优势函数 [@macglashan2017interactive]。

最主要的参考文献，Christiano 等人 2017 年的工作，是将 RLHF 应用于 Atari 游戏中智能体轨迹之间的偏好 [@christiano2017deep]。
这篇引入 RLHF 的工作紧随 DeepMind 在深度 Q 网络（DQN）上的开创性强化学习工作之后，该工作表明 RL 智能体可以从零开始学会解决流行的视频游戏。
这项工作表明，人类在轨迹之间做出选择在某些领域可能比直接与环境交互更有效。这使用了一些巧妙的条件设计，但仍然令人印象深刻。

![Christiano 等人（2017）的核心 RLHF 循环：奖励预测器从轨迹片段的比较中异步训练，智能体最大化预测奖励。](images/rlhf_schematic.png){#fig:rlhf_schematic width=66%}

这一方法被更直接的奖励建模 [@ibarz2018reward] 所扩展，而早期 RLHF 工作中对深度学习的采用，在一年后通过使用神经网络模型对 TAMER 的扩展而达到顶峰 [@warnell2018deep]。

随着奖励模型作为一个通用概念被提出，这一时代开始转型——奖励模型不再仅仅是解决 RL 问题的工具，而是被作为研究对齐的方法 [@leike2018scalable]。

## 2019 年至 2022 年：语言模型上的人类偏好强化学习

基于人类反馈的强化学习，早期也常被称为基于人类偏好的强化学习，很快被越来越关注扩展大语言模型的 AI 实验室所采用。
这项工作的大部分始于 2019 年的 GPT-2 和 2020 年的 GPT-3 之间。
2019 年最早的工作 *Fine-Tuning Language Models from Human Preferences* 与现代 RLHF 工作以及本书将涵盖的内容有许多惊人的相似之处 [@ziegler2019fine]。
许多标准术语，如学习奖励模型、KL 距离、反馈图示等，都在该论文中被正式定义，尽管最终模型的评估任务及其能力与今天人们所做的工作不同。
自此之后，RLHF 被应用于各种任务。
重要的例子包括通用摘要 [@stiennon2020learning]、书籍的递归摘要 [@wu2021recursively]、指令遵循（InstructGPT）[@ouyang2022training]、浏览器辅助问答（WebGPT）[@nakano2021webgpt]、用引文支持答案（GopherCite）[@menick2022teaching]，以及通用对话（Sparrow）[@glaese2022improving]。

除了应用之外，许多开创性论文定义了 RLHF 未来的关键领域，包括：

1. 奖励模型过度优化 [@gao2023scaling]：RL 优化器对在偏好数据上训练的模型过拟合的能力，
2. 语言模型作为对齐的通用研究领域 [@askell2021general]，以及
3. 红队测试 [@ganguli2022red] —— 评估语言模型安全性的过程。

工作继续致力于改进 RLHF 以应用于聊天模型。
Anthropic 继续广泛使用它来训练早期版本的 Claude [@bai2022training]，并且出现了早期的 RLHF 开源工具 [@ramamurthy2022reinforcement], [@havrilla-etal-2023-trlx], [@vonwerra2022trl]。

## 2023 年至今：ChatGPT 时代

ChatGPT 的发布非常明确地说明了 RLHF 在其训练中的作用 [@openai2022chatgpt]：

> 我们使用基于人类反馈的强化学习（RLHF）训练了这个模型，使用的方法与 InstructGPT 相同，但数据收集设置略有不同。

自那时起，RLHF 已被广泛应用于主流语言模型中。
众所周知，Anthropic 的 Constitutional AI for Claude [@bai2022constitutional]、Meta 的 Llama 2 [@touvron2023llama] 和 Llama 3 [@dubey2024llama]、Nvidia 的 Nemotron [@adler2024nemotron]、Ai2 的 Tülu 3 [@lambert2024t] 等都使用了 RLHF。

今天，RLHF 正在发展成为一个更广泛的偏好微调（PreFT）领域，包括新的应用，如用于中间推理步骤的过程奖励 [@lightman2023let]（在第 5 章中介绍）；受直接偏好优化（DPO）启发的直接对齐算法 [@rafailov2024direct]（在第 8 章中介绍）；从代码或数学的执行反馈中学习 [@kumar2024training], [@singh2023beyond] 以及其他受 OpenAI o1 启发的在线推理方法 [@openai2024o1]（在第 7 章中介绍）。
