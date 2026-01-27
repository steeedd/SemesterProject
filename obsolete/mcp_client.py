
import re
import json
import asyncio
from typing import Optional
from contextlib import AsyncExitStack
import warnings

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

# Use there the model you wish (it must support tool calling)
MODEL_ID = "Qwen/Qwen3-1.7B"


# Load the model and tokenizer (quantized to 8bit)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
)

# Helper function becaus in transformers the tool calls should be a field of assistant messages.
def try_parse_tool_calls(content: str):
    """Try parse the tool calls."""
    tool_calls = []
    offset = 0
    for i, m in enumerate(re.finditer(r"<tool_call>\n(.+)?\n</tool_call>", content)):
        if i == 0:
            offset = m.start()
        try:
            func = json.loads(m.group(1))
            tool_calls.append({"type": "function", "function": func})
            if isinstance(func["arguments"], str):
                func["arguments"] = json.loads(func["arguments"])
        except json.JSONDecodeError as e:
            print(f"Failed to parse tool calls: the content is {m.group(1)} and {e}")
            pass
    if tool_calls:
        if offset > 0 and content[:offset].strip():
            c = content[:offset]
        else:
            c = ""
        return {"role": "assistant", "content": c, "tool_calls": tool_calls}
    return {"role": "assistant", "content": re.sub(r"<\|im_end\|>$", "", content)}


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.llm = model
        self.tokenizer = tokenizer

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        print("DEBUG: Before stdio_client")
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        print("DEBUG: After stdio_client")
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        print("TOOLS:", [t.name for t in response.tools])
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
      """Process a query using Claude and available tools"""
      messages = [
          {
              "role": "user",
              "content": query
          }
      ]

      response = await self.session.list_tools()
      available_tools = [{
          "name": tool.name,
          "description": tool.description,
          "input_schema": tool.inputSchema
      } for tool in response.tools]


      # Initial LLM Call
      text = self.tokenizer.apply_chat_template(messages, tools=available_tools, add_generation_prompt=True, tokenize=False)
      print("PROMPT SENT TO LLM:\n", text)
      inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
      outputs = self.llm.generate(**inputs, max_new_tokens=512)
      output_text = tokenizer.batch_decode(outputs)[0][len(text):]
      print("OUTPUT LLM:\n", output_text)


      # Processing response and handel tool calls with the LLM `output_text`
      final_text = []

      parsed_message = try_parse_tool_calls(output_text)
      messages.append(parsed_message)

      final_text.append(parsed_message["content"])

      if tool_calls := messages[-1].get("tool_calls", None):
          for tool_call in tool_calls:
              if fn_call := tool_call.get("function"):
                  fn_name: str = fn_call["name"]
                  fn_args: dict = fn_call["arguments"]

                  print(f"Calling tool: {fn_name} with args: {fn_args}")
                  final_text.append(f"Calling tool: {fn_name} with args: {fn_args}")
                  result = await self.session.call_tool(fn_name, fn_args)
                  #print(result)
                  fn_res = result.content
                  #print(f"Tool result: {fn_res}")

                  messages.append({
                      "role": "tool",
                      "name": fn_name,
                      "content": fn_res,
                  })

              # Get next response from Claude
              text = self.tokenizer.apply_chat_template(messages, tools=available_tools, add_generation_prompt=True, tokenize=False)
              inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
              outputs = self.llm.generate(**inputs, max_new_tokens=512)
              output_text = self.tokenizer.batch_decode(outputs)[0][len(text):]

              final_text.append(output_text)



      return "\n".join(final_text)

    async def chat_loop(self):
      """Run an interactive chat loop"""
      print("\nMCP Client Started!")
      print("Type your queries or 'quit' to exit.")

      while True:
          try:
              query = "What is the weather in San Francisco today?"

              if query.lower() == 'quit':
                  break

              response = await self.process_query(query)
              print("\n" + response)

          except Exception as e:
              print(f"\nError: {str(e)}")

    async def cleanup(self):
      """Clean up resources"""
      await self.exit_stack.aclose()


async def main():
    client = MCPClient()
    try:
        # Connetti al server MCP
        await client.connect_to_server("weather_mcp_server.py")


        # Dai il prompt una sola volta
        user_prompt = "What is the weather in San Francisco today?"
        response = await client.process_query(user_prompt)


        # Stampa la risposta finale
        print("\nLLM + Tool Calling Response:\n", response)

    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())
