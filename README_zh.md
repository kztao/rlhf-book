# RLHF Book

一本关于基于人类反馈的强化学习的全面指南（以及对后训练语言模型的广泛介绍）。

**[在线阅读](https://rlhfbook.com)** | **[在 Manning 订购印刷版](https://hubs.la/Q03Tc3cf0)** | **[加入 Discord 社区](https://discord.gg/yz5AwK4gBR)**

这本书是我尝试开源我在 ChatGPT 后语言模型起飞时期在开放模型前沿工作中获得的所有知识。
当我开始时，许多成熟的方法如拒绝采样没有经典参考文献。
另一方面，使模型更具个性的行业实践——通俗称为性格训练——没有开放研究。
对我来说很明显，记录、学习基础知识、仔细策划参考文献（在 AI 垃圾内容的时代）以及它们之间的一切将是人们的绝佳起点。

今天，我正在添加代码，并将其视为想要学习的人的家园。
你应该使用编程助手来提问。
你应该购买纸质书，因为现实世界很重要。
你应该阅读针对你的特定 AI 输出。

未来，我想为此构建更多的教育资源，如开源幻灯片和更多学习方式。
最终，由于衡量人类偏好是多么不可能，RLHF 将永远不会是一个已解决的问题。

谢谢你的阅读。
感谢你贡献任何反馈或参与社区。

-- Nathan Lambert, @natolambert

## 仓库结构

```
rlhf-book/
├── book/                   # 书籍源文件和构建文件
│   ├── chapters/           # Markdown 源文件（01-introduction.md 等）
│   ├── images/             # 章节中引用的图片
│   ├── assets/             # 品牌资源（封面、标志）
│   ├── templates/          # Pandoc 模板（HTML、PDF、EPUB）
│   ├── scripts/            # 构建工具
│   └── data/               # 库数据
├── code/                   # 参考实现
│   ├── instruction_tuning/ # 使用聊天模板对基础模型进行 SFT
│   ├── policy_gradients/   # PPO、REINFORCE、GRPO、RLOO
│   ├── reward_models/      # 偏好 RM、ORM、PRM 训练
│   ├── direct_alignment/   # DPO 及其变体
│   └── rejection_sampling/ # Best-of-N 拒绝采样
├── diagrams/               # 图表源文件
├── teach/                  # 教学材料（课程、幻灯片）
├── build/                  # 生成输出（git 忽略）
└── Makefile                # 构建系统
```

## 引用

要引用本书，请使用以下格式：

```bibtex
@book{rlhf2026lambert,
  author       = {Nathan Lambert},
  title        = {Reinforcement Learning from Human Feedback},
  year         = {2026},
  publisher    = {Online},
  url          = {https://rlhfbook.com},
}
```

## 许可证

- 代码：[MIT](LICENSE-CODE)
- 章节：[CC-BY-NC-SA-4.0](LICENSE-CHAPTERS)

## 贡献者

虽然我作为唯一的"作者"和此项目的创建者获得荣誉，但我非常幸运有许多来自早期读者的贡献。这些极大加速了编辑进度，并直接为本书添加了有意义的内容。我很乐意向实质性贡献者发送免费副本，并期望互联网的善意以意想不到的方式回馈他们。

查看所有[贡献者](https://github.com/natolambert/rlhf-book/graphs/contributors)。
