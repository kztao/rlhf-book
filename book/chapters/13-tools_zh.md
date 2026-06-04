<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "合成数据"
prev-url: "12-synthetic-data"
page-title: 工具使用与函数调用
search-title: "第 13 章：工具使用与函数调用"
meta-description: "工具使用和函数调用作为构建更强大语言模型产品和智能体的后训练目标。"
next-chapter: "过度优化"
next-url: "14-over-optimization"
---

# 工具使用与函数调用

语言模型使用工具是扩展其能力的自然方式，特别是对于需要外部工具包含信息的高精度任务，或对于需要与复杂网络系统交互的智能体。
工具使用是语言模型需要被训练才能拥有的技能，而 RLHF 和本书中介绍的所有其他方法都可以对此进行优化。
考虑来自用户的如下问题：

> **用户**：今天的总统是谁？

没有工具的语言模型由于预训练数据的知识截止点而难以回答这个问题，但通过一次搜索查询就可以轻松获取此信息。
考虑另一个例子：

> **用户**：将我的下载文件夹中所有的 arXiv 论文移动到我的 ~/research/ 目录中，文件名标明论文日期。

这是一个模型权重本身甚至无法尝试的任务——工具的使用使语言模型能够解决更广泛的任务。

在深入之前，区分通常可以互换使用的相关术语是有用的：

- **工具使用**：模型发出结构化请求（工具名称和参数）；编排器执行工具；结果附加到上下文中；模型继续生成。
- **函数调用**：工具使用中参数必须符合一组函数的声明模式（通常是 JSON Schema），从而实现可靠的解析和验证。
- **代码执行**：工具使用的一种特殊情况，其中"工具"是代码解释器（例如 Python）；结果作为工具输出返回。

## 工具使用概述

AI 模型通过输出特殊 token 来触发特定端点，从而使用任何外部工具。
这些可以是任何东西，从高度特定的工具，如返回特定地点天气的函数，到代码解释器或搜索引擎，作为复杂行为的基本构建块。
我们的第一个示例展示了语言模型需要更多最新信息来补充其基于过去数据训练的权重的固定性质，但也有像代码执行这样的工具，它让语言模型绕过其概率性、生成性的本质，返回精确的答案。
考虑打印 pi 的 50 位近似值的任务（而不是从记忆中背诵并冒着幻觉的风险）。
具有工具的语言模型可以执行以下操作：

```text
<code>
from decimal import Decimal, getcontext
getcontext().prec = 60

def compute_pi():
    # 计算 pi 的 Chudnovsky 算法
    C = 426880 * Decimal(10005).sqrt()
    K, M, X, L, S = 0, 1, 1, 13591409, Decimal(13591409)
    for i in range(1, 100):
        M = M * (K**3 - 16*K) // ((i)**3)
        K += 12
        L += 545140134
        X *= -262537412640768000
        S += Decimal(M * L) / X
    return C / S

print(str(compute_pi())[:52])
</code>

<output>
3.14159265358979323846264338327950288419716939937510
</output>
```

本章概述了工具使用在现代语言模型中的起源、其基础和格式化，以及在领先模型中有效利用工具的当前权衡。

"工具使用"这个术语的确切起源不清楚，但这个想法的起源远早于 RLHF 普及的后 ChatGPT 世界。
约 2015 年的早期例子试图构建早于现代语言模型的系统，如神经程序解释器（NPI）[@reed2015neural]，"一个递归和组合的神经网络，学习表示和执行程序。"
随着语言模型变得更受欢迎，许多子领域正在使用与外部能力的集成来提升性能。
为了获取权重之外的信息，许多人使用检索增强生成 [@lewis2020retrieval] 或网络浏览 [@nakano2021webgpt]。
此后不久，其他人在探索与程序 [@gao2023pal] 或工具 [@parisi2022talm] 集成的语言模型。

随着领域的成熟，这些模型除了底层语言建模的巨大改进外，还获得了更复杂的能力。
例如，ToolFormer 可以使用"计算器、问答系统、两个不同的搜索引擎、翻译系统和日历" [@schick2023toolformerlanguagemodelsteach]。
不久之后，Gorilla 被训练为使用 1645 个 API（来自 PyTorch Hub、TensorFlow Hub v2 和 Hugging Face），其评估 APIBench 成为流行的 Berkeley Function Calling Leaderboard 的基础 [@patil2023gorilla]。
自这些早期模型以来，被调用的动作的多样性大幅增长。

工具使用模型现在与常规语言模型交互深度交织在一起。
模型上下文协议（MCP）作为一种通用格式出现，用于将语言模型连接到外部数据源（或工具）[@anthropic_mcp_2024]。
随着更强的模型和更好的格式，工具使用语言模型在许多情况下被使用，包括流行应用程序中的生产力副驾驶，如 Microsoft Office 或 Google Workspace、科学领域 [@bran2023chemcrow]、医疗领域 [@li2024mmedagent]、编程代理 [@zhang2024codeagent] 如 Claude Code 或 Cursor、与数据库的集成，以及许多其他自主工作流。

评估工具使用模型涉及多个维度：工具名称和参数正确性的精确匹配指标、模式有效性以及在模拟环境中的端到端任务完成。
跨试验的可靠性也很重要——$\tau$-bench 引入了 pass^k 指标（与 pass@k 不同）来衡量代理是否一致地成功而非偶尔成功 [@yao2024taubench]。
ToolLLM 及其 ToolBench 数据集为在 16,000+ 个真实世界 API 上训练和评估工具使用提供了大规模框架 [@qin2023toollm]，而 Berkeley Function Calling Leaderboard（BFCL）仍然是比较模型在函数调用准确性方面的流行基准 [@patil2023gorilla]。

## 在生成中交织工具调用

函数调用的训练数据看起来很像其他后训练数据，但有一个补充：一个系统提示，告知模型有哪些可用的工具。
下面展示了一个格式化的示例数据点，带有系统提示和 JSON 格式的可用工具：
```xml
<system>
你是一个函数调用 AI 模型。你在 <functions></functions> XML 标签内获得函数签名。你可以调用一个或多个函数来协助用户查询。不要假设将哪些值插入到函数中。
</system>

<functions>
[
  {
    "name": "search_movies",
    "description": "按标题搜索电影并返回带有 ID 的匹配结果。",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "电影标题的搜索字符串。"
        }
      },
      "required": ["query"]
    }
  },
  {
    "name": "get_movie_details",
    "description": "获取电影的详细信息，包括演员阵容、片长和剧情简介。",
    "parameters": {
      "type": "object",
      "properties": {
        "movie_id": {
          "type": "string",
          "description": "电影的唯一标识符。"
        }
      },
      "required": ["movie_id"]
    }
  },
  {
    "name": "get_showtimes",
    "description": "获取给定位置和日期的电影放映时间。",
    "parameters": {
      "type": "object",
      "properties": {
        "movie_id": {
          "type": "string",
          "description": "电影的唯一标识符。"
        },
        "zip_code": {
          "type": "string",
          "description": "影院位置的邮政编码。"
        },
        "date": {
          "type": "string",
          "description": "放映日期，格式为 YYYY-MM-DD。"
        }
      },
      "required": ["movie_id", "zip_code"]
    }
  }
]
</functions>

<user>
...
</user>
```
尽管语言模型正在生成补全，但如果它遵循此示例，它将生成 token `search_movies("Star Wars")` 来搜索 Star Wars。
这通常在特殊格式化 token 内编码，然后插入到序列中的下一些 token 将包含工具输出。
有了这个，模型可以学习完成比许多简单的独立模型更具挑战性的任务。

工具使用的一种流行形式是代码执行，它允许模型获得复杂逻辑或数学问题的精确答案。
例如，语言模型执行中的代码执行可以在推理模型的思考 token 期间发生。
与函数调用一样，首先有用于执行的代码标签（由模型生成），然后是单独的输出标签。
```text
<|user|>
第 50 个斐波那契数是多少？（使用标准 F_0=0, F_1=1 索引。）</s>
<|assistant|>
<think>
好的，我将用一个简单的循环计算第 50 个斐波那契数，然后返回结果。

<code>
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

fib(50)
</code>

<output>
12586269025
</output>
</think>
<answer>
第 50 个斐波那契数是 12 586 269 025。
</answer>
```

底层发生的是语言模型将工具输入和输出与标准自回归生成的 token 交织在一起。
使这成为可能的编排循环看起来像这样：

```python
messages = [...]
while True:
    response = model(messages, tools=tools)
    if not response.tool_calls:
        return response.text

    for call in response.tool_calls:
        result = execute_tool(call.name, call.args)
        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
```

![工具使用将模型生成与外部执行交织：模型生成 token 直到发出工具调用（橙色），外部系统执行工具并将输出（紫色）注入序列，然后模型继续生成。模型可以在单次生成中发出多个工具调用。在训练期间，工具调用和输出 token 通常从损失中掩码。](images/tool_use_generation.png){#fig:tool-use-generation}

工具使用的训练是关于让模型以这种不同的 token 流可预测地行为——知道何时发出工具调用、如何正确格式化参数以及如何将结果纳入其回答。
开放模型必须被训练以与用户可能现成连接的各种工具一起工作。

## 多步工具推理

OpenAI 的 o3 模型代表了多步工具使用如何与语言模型集成的实质性进步。
这种行为与社区中更早的研究趋势相关。
例如，ReAct [@yao2023react] 展示了动作和推理如何交织到一个模型生成中：

> 在本文中，我们探索使用 LLM 以交织的方式生成推理轨迹和任务特定动作，允许两者之间更大的协同作用：推理轨迹帮助模型诱导、跟踪和更新动作计划以及处理异常，而动作允许它与外部来源如知识库或环境交互并收集额外信息。

随着工具使用能力的巩固和推理模型的起飞，多轮工具使用已成长为一个令人兴奋的研究领域 [@wang2025ragenunderstandingselfevolutionllm]。

## 模型上下文协议

模型上下文协议（MCP）是将语言模型连接到外部数据源和信息系统的开放标准 [@anthropic_mcp_2024]。
在数据层，MCP 使用 JSON-RPC 2.0，其原语具有发现和执行方法。
与每个外部系统需要特定的工具调用格式不同，MCP 使模型能够通过标准化协议访问丰富的上下文信息。

MCP 是在本章工具使用内容之上的简单添加——它是应用程序以可预测的 JSON schema 向语言模型传递上下文（数据 + 动作）的方式。
模型与之交互的 MCP 服务器具有核心原语：资源（只读数据块）、提示（模板化消息/工作流）和工具（模型可以调用的函数）。
由此，MCP 架构可以总结为：

- MCP 服务器包装特定的数据源或能力。
- MCP 客户端（例如 Claude Desktop、IDE 插件）聚合一个或多个服务器。
- 主机，例如 Claude 或 ChatGPT 应用程序，提供用户/LLM 界面；切换模型供应商或后端工具仅意味着在中间交换客户端。

MCP 使工具使用模型的开发者可以使用相同的基础设施将其服务器或客户端附加到不同的模型，同时模型具有可预测的格式，可以使用它来集成外部组件。
这些共同为真实世界领域中的工具使用模型创造了一个更可预测的开发环境。

MCP 服务器通过标准化的 JSON schema 向客户端暴露工具：
```json
{
  "name": "get_weather",
  "description": "获取某个位置的当前天气",
  "inputSchema": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "城市名称或坐标"
      }
    },
    "required": ["location"]
  }
}
```

实现此工具的最小 Python MCP 服务器：
```python
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("weather-server")

@server.list_tools()
async def list_tools():
    return [Tool(
        name="get_weather",
        description="获取当前天气",
        inputSchema={
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }
    )]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_weather":
        weather = fetch_weather(arguments["location"])
        return [TextContent(type="text", text=weather)]
```

## 实现细节

在实现工具使用模型时，有多个格式化和掩码决策：

- **Python 与 JSON 格式化**：在本章中，我们包括了将工具使用格式化为 JSON 数据结构和 Python 代码的示例。模型倾向于选择一种结构，而行业中不同提供商使用不同的格式。
- **掩码工具输出**：训练工具使用模型时的一个重要细节是，工具输出中的 token 从模型的训练损失中掩码。这确保模型不学习预测处理工具调用的系统的输出（因为结果不是由模型生成的 token）。
- **用于工具调用的多轮格式化**：在实现工具调用模型时，常见做法是向数据加载格式添加更多结构。后训练数据集的标准做法是用户和助手交替的消息列表（通常还有系统消息）。工具使用的总体结构相同，但模型的轮次被分割为由每个工具调用分隔的内容子部分。
- **分词和消息格式细节**：OpenAI 消息格式中的工具调用通常通过聊天模板进行分词（控制发送到模型的消息格式的代码），将结构化 JSON 表示转换为原始 token 流。此过程在不同模型架构中有所不同——一些使用特殊 token 来标示工具调用，而其他则在 token 流本身中维护结构化格式。[聊天模板游乐场](https://huggingface.co/spaces/huggingfacejs/chat-template-playground?modelId=Qwen/Qwen3-8B) 提供了一个交互式环境来探索不同模型如何将消息格式转换为 token 流。
- **推理 token 连续性**：随着推理模型的出现，它们在回答之前有单独的"推理"token 流，对于如何在循环中处理工具使用存在不同的实现。一些模型在一个轮次内的工具调用步骤之间保留推理 token，跨多个工具调用维护上下文。然而，这些 token 通常在轮次之间被擦除以最小化服务成本（但并不总是如此——这是一个设计决策）。
- **跨提供商的 API 格式化**（截至 2026 年 5 月）：不同提供商使用概念上相似但技术上不同的格式。OpenAI 的 Chat Completions API 使用带有唯一 ID 的 `tool_calls` 数组，而较新的 Responses API 将调用表示为 `function_call` 项目，并将结果返回为由 `call_id` 键控的 `function_call_output` 项目。Anthropic 使用 `input_schema` 定义工具，并将调用和结果表示为 `tool_use` 和 `tool_result` 内容块。Gemini 暴露函数调用模式，如 `AUTO`、`ANY`、`NONE`，以及在支持的 Gemini 和 Vertex AI 配置中的 `VALIDATED`。
- **模式一致性与约束解码**：生产系统通常使用约束解码或"strict mode"选项强制执行有效的 JSON 和正确的参数类型，减少因格式错误输出导致的重试。一些闭源模型提供商进行额外的后训练，专门使结构化 JSON 输出可靠，而对于开源模型，这在 vLLM 等系统中作为推理标志处理。
- **工具输出上下文消耗**：工具输出可以快速消耗模型的上下文窗口，尤其是搜索或检索工具返回许多结果时。系统必须决定如何截断、总结或分页工具输出，以保持上下文可管理，同时保留模型继续所需的信息。

将其与后训练联系起来：工具使用训练数据来自哪里，使用什么目标？
人类编写的工具跟踪成本高昂，因此大多数现代工具使用语料是合成的或自举的——Toolformer 风格的自标注 [@schick2023toolformerlanguagemodelsteach] 或像 ToolBench [@qin2023toollm] 中的大规模生成。
对于训练目标，对工具轨迹的监督微调（SFT）教授基本格式化和工具选择。
这引导了行为，通常足以建立技能的基础。
对轨迹的偏好优化（例如 DPO）可以改进关于何时调用工具与直接回答的决策。
对于具有多步工具使用的智能体任务，使用环境反馈（任务成功、约束满足）的 RL 成为自然目标——模型从其工具增强的动作是否实际解决了问题中学习。
