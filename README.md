[合集 - MCP(4)](https://github.com)

[1.[MCP][01]简介与概念09-14](https://github.com/XY-Heruo/p/19091500)[2.[MCP][02]快速入门MCP开发09-15](https://github.com/XY-Heruo/p/19092074):[闪电加速器](https://sanzhijia.com)[3.[MCP][03]使用FastMCP开发MCP应用09-15](https://github.com/XY-Heruo/p/19092229)

4.[MCP][04]Sampling示例09-15

收起

## 前言

在第一篇MCP文章中我们简单介绍了Sampling：

> 采样是工具与LLM交互以生成文本的机制。通过采样，工具可以请求LLM生成文本内容，例如生成诗歌、文章或其他文本内容。采样允许工具利用LLM的能力来创建内容，而不仅限于执行预定义的操作。

为什么我们要在MCP Server通过Sampling方式调用Client的LLM，而不是MCP Server直接调用LLM呢？这背后其实有一套巧妙的设计哲学：

* MCP 服务端更像是一个"指挥家"，它统筹整合各种资源和工具，将它们编排成一个完整的服务提供给客户端。当服务端在实现其功能时需要借助 LLM 的"智慧"，由服务端发起请求（服务端 -> 客户端 -> LLM）是最合理的安排。
* 根据 MCP 的设计理念，服务端专注于提供工具和资源服务，而不是直接与 LLM 交互。这就像是一个专业的中介，负责协调而不是亲自下场。因此，服务端会将请求发给客户端，由客户端这个"桥梁"再将请求转发到 LLM（服务端 -> 客户端 -> LLM）。

本文基于FastMCP演示下MCP Server和MCP Client如何实现Sampling，让你彻底搞懂这个有趣的机制。

## MCP Server

在MCP Server端，我们实现了一个情感分析工具，它会通过Sampling机制请求LLM帮助分析文本情感：

```
from fastmcp import Context, FastMCP
from mcp.types import SamplingMessage, TextContent

from pkg.log import logger

mcp = FastMCP("custom")

@mcp.tool()
async def analyze_sentiment(text: str, ctx: Context) -> dict:
    """Analyze the sentiment of a given text.
    
    Args:
        text (str): The text to analyze.

    Returns:
        dict: A dictionary containing the sentiment analysis result.
    """

    prompt = f"""Analyze the sentiment of the following text as positive, negative, or neutral. 
    Just output a single word - 'positive', 'negative', or 'neutral'.
    """

    logger.info(f"Analyzing sentiment for text: {text}, prompt: {prompt}")
    response = await ctx.sample(
        messages=[SamplingMessage(role="user", content=TextContent(type="text",text=text))],
        system_prompt=prompt
    )

    logger.info(f"response: {response}")

    sentiment = response.text.strip().lower()

    # Map to standard sentiment values
    if "positive" in sentiment:
        sentiment = "positive"
    elif "negative" in sentiment:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    return {"text": text, "sentiment": sentiment}


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="localhost", port=8001, show_banner=False)
```

这段代码的核心在于`ctx.sample()`调用。当工具需要LLM的"智慧"时，它不会直接调用LLM API，而是通过上下文中的sample方法发起一个采样请求。这就像你问朋友一个问题，朋友会去请教更专业的人，然后把答案告诉你。

## MCP Client

MCP Client端的实现更加有趣，它需要同时扮演"翻译官"和"调度员"的角色：

```
import asyncio
import json
import readline  # For enhanced input editing
import traceback
from typing import cast

from fastmcp import Client
from fastmcp.client.sampling import SamplingMessage, SamplingParams
from mcp.shared.context import RequestContext
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageFunctionToolCall

from pkg.config import cfg
from pkg.log import logger


class MCPHost:
    """MCP主机类，用于管理与MCP服务器的连接和交互"""
    
    def __init__(self, server_uri: str):
        """
        初始化MCP客户端
        
        Args:
            server_uri (str): MCP服务器的URI地址
        """
        # 初始化MCP客户端连接
        self.mcp_client: Client = Client(server_uri, sampling_handler=self.sampling_handler)
        # 初始化异步OpenAI客户端用于与LLM交互
        self.llm = AsyncOpenAI(
            base_url=cfg.llm_base_url,
            api_key=cfg.llm_api_key,
        )
        # 存储对话历史消息
        self.messages = []

    async def close(self):
        """关闭MCP客户端连接"""
        if self.mcp_client:
            await self.mcp_client.close()

    async def sampling_handler(self, messages: list[SamplingMessage], params: SamplingParams, ctx: RequestContext) -> str:
        """处理采样消息的回调函数"""
        conversation = []
        # Use the system prompt if provided
        system_prompt = params.systemPrompt or "You are a helpful assistant."
        conversation.append({"role": "system", "content": system_prompt})
        for message in messages:
            content = message.content.text if hasattr(message.content, 'text') else str(message.content)
            conversation.append({"role": message.role, "content": content})

        resp = await self.llm.chat.completions.create(
            model=cfg.llm_model,
            messages=conversation,
            temperature=0.3,
        )
        message = resp.choices[0].message
        return message.content if hasattr(message, "content") else ""

    async def process_query(self, query: str) -> str:
        """Process a user query by interacting with the MCP server and LLM.
        
        Args:
            query (str): The user query to process.

        Returns:
            str: The response from the MCP server.
        """
        # 将用户查询添加到消息历史中
        self.messages.append({
            "role": "user",
            "content": query,
        })

        # 使用异步上下文管理器确保MCP客户端连接正确建立和关闭
        async with self.mcp_client:
            # 从MCP服务器获取可用工具列表
            tools = await self.mcp_client.list_tools()
            # 构造LLM可以理解的工具格式
            available_tools = []

            # 将MCP工具转换为OpenAI格式
            for tool in tools:
                available_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    }
                })
            logger.info(f"Available tools: {[tool['function']['name'] for tool in available_tools]}")

            # 调用LLM，传入对话历史和可用工具
            resp = await self.llm.chat.completions.create(
                model=cfg.llm_model,
                messages=self.messages,
                tools=available_tools,
                temperature=0.3,
            )

            # 存储最终响应文本
            final_text = []
            # 获取LLM的首个响应消息
            message = resp.choices[0].message
            # 如果响应包含直接内容，则添加到结果中
            if hasattr(message, "content") and message.content:
                final_text.append(message.content)

            # 循环处理工具调用，直到没有更多工具调用为止
            while message.tool_calls:
                # 遍历所有工具调用
                for tool_call in message.tool_calls:
                    # 确保工具调用有函数信息
                    if not hasattr(tool_call, "function"):
                        continue

                    # 类型转换以获取函数调用详情
                    function_call = cast(ChatCompletionMessageFunctionToolCall, tool_call)
                    function = function_call.function
                    tool_name = function.name
                    # 解析函数参数
                    tool_args = json.loads(function.arguments)

                    # 检查MCP客户端是否已连接
                    if not self.mcp_client.is_connected():
                        raise RuntimeError("Session not initialized. Cannot call tool.")
                    
                    # 调用MCP服务器上的指定工具
                    result = await self.mcp_client.call_tool(tool_name, tool_args)

                    # 将助手的工具调用添加到消息历史中
                    self.messages.append({
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": function.name,
                                    "arguments": function.arguments
                                }
                            }
                        ]
                    })

                    # 将工具调用结果添加到消息历史中
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id":tool_call.id,
                        "content": str(result.content) if result.content else ""
                    })
                
                # 基于工具调用结果再次调用LLM
                final_resp = await self.llm.chat.completions.create(
                    model=cfg.llm_model,
                    messages=self.messages,
                    tools=available_tools,
                    temperature=0.3,
                )
                # 更新消息为最新的LLM响应
                message = final_resp.choices[0].message
                # 如果响应包含内容，则添加到最终结果中
                if message.content:
                    final_text.append(message.content)
            
            # 返回连接后的完整响应
            return "\n".join(final_text)

    async def chat_loop(self):
        """主聊天循环，处理用户输入并显示响应"""
        print("Welcome to the MCP chat! Type 'quit' to exit.")

        # 持续处理用户输入直到用户退出
        while True:
            try:
                # 获取用户输入
                query = input("You: ").strip()

                # 检查退出命令
                if query.lower() == "quit":
                    print("Exiting chat. Goodbye!")
                    break

                # 跳过空输入
                if not query:
                    continue

                # 处理用户查询并获取响应
                resp = await self.process_query(query)
                print(f"Assistant: {resp}")
            
            # 捕获并记录聊天循环中的任何异常
            except Exception as e:
                logger.error(f"Error in chat loop: {str(e)}")
                logger.error(traceback.format_exc())


async def main():
    """主函数，程序入口点"""
    # 创建MCP主机实例
    client = MCPHost(server_uri="http://localhost:8001/mcp")
    try:
        # 启动聊天循环
        await client.chat_loop()
    except Exception as e:
        # 记录主程序中的任何异常
        logger.error(f"Error in main: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        # 确保客户端连接被正确关闭
        await client.close()
    

if __name__ == "__main__":
    # 运行主程序
    asyncio.run(main())
```

Client的关键在于`sampling_handler`函数，当Server端发起采样请求时，Client会通过这个函数接收请求，并实际调用LLM完成文本生成。这就像一个称职的助理，当老板（Server）需要某些信息时，助理（Client）会去查询资料（LLM）并把结果汇报给老板。

Client运行输出

```
Welcome to the MCP chat! Type 'quit' to exit.
You: 分析下这句的情感倾向：问君能有几多愁，恰似一江春水向东流
Assistant: 这句诗"问君能有几多愁，恰似一江春水向东流"表达的情感倾向是**负面**的。它通过比喻的方式，将忧愁比作一江春水向东流，暗示了忧愁的绵长和无法排解，带有浓厚的哀愁与感伤情绪。
You: what can you do?
Assistant: 我可以帮助你进行情感分析，例如分析诗句、句子的情感倾向。如果你有其他需求，也可以告诉我，我会尽力提供帮助！
You: quit
Exiting chat. Goodbye!
```

从运行结果可以看出，当用户请求分析诗句情感时，整个流程是这样的：

1. 用户输入需要分析的诗句
2. Client调用Server端的analyze\_sentiment工具
3. 工具通过Sampling机制请求LLM分析情感
4. Client接收LLM的响应并返回给用户

## 小结

通过这个示例，我们可以看到MCP中Sampling机制的巧妙之处：

1. **职责分离**：MCP Server专注于业务逻辑和工具编排，不直接与LLM交互，保持了架构的清晰性。
2. **灵活性**：Client端可以自由选择不同的LLM提供商和模型，Server端无需关心具体实现细节。
3. **可扩展性**：可以轻松添加更多需要LLM能力的工具，而无需修改Client端的LLM调用逻辑。
4. **统一接口**：通过标准化的Sampling接口，不同组件之间可以无缝协作。

这种设计让MCP系统既保持了良好的模块化结构，又充分发挥了LLM的能力，真正做到了"各司其职，协同工作"。

## 参考

* [yuan - 一文读懂 MCP 的 Sampling（采样），赋予 MCP 服务端智能能力！](https://github.com)
