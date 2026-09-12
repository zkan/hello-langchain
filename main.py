from langchain.agents import create_agent
from langchain.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


def main():
    agent = create_agent(
        model="ollama:qwen3.5:4b",
        tools=[get_weather],
        system_prompt="You are a helpful assistant",
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What's the weather in San Francisco?"
                }
            ]
        }
    )
    # print(result)
    print(result["messages"][-1].content_blocks)


if __name__ == "__main__":
    main()
