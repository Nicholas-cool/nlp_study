# 大语言模型笔记







## `Llama`

`Llama`（`Large Language Model Meta AI`）是由 `Meta`（前 `Facebook`）开发的大型语言模型系列，专为自然语言处理任务设计。`Llama` 主要用于支持生成文本、总结、翻译以及回答问题等应用。

- [`llama-models`](https://github.com/meta-llama/llama-models) ：基础模型的中央存储库，包括基本实用程序、模型卡、许可证和使用政策。
- [`PurpleLlama`](https://github.com/meta-llama/PurpleLlama) ：`Llama Stack` 的关键组成部分，重点关注安全风险和推理时间降低措施。
- [`llama-toolchain`](https://github.com/meta-llama/llama-toolchain) ：模型开发（推理/微调/安全屏蔽/合成数据生成）接口和规范实现。
- [`llama-agentic-system`](https://github.com/meta-llama/llama-agentic-system) ：`E2E` 独立式 `Llama Stack` 系统，以及有主见的底层接口，可创建代理应用程序。
- [`llama-recipes`](https://github.com/meta-llama/llama-recipes) ：社区驱动的脚本和集成。



## `LoRA`

`LoRA`（`Low Rank Adaption`）是一种参数高效的微调方法（`parameter efficient fine-tuning`，`PEFT`）。

【`HuggingFace` `peft` 库】https://github.com/huggingface/peft

