from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph .prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

@tool
def calculator(a:float, b:float) -> str:
    """Useful for performing calculations on numbers"""
    print("Tool called: calculator")
    return f"The result of {a} + {b} is {a + b}"

def main():
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    tools = [calculator]
    agent_executor = create_react_agent(model, tools)

    print("Welcome to the chatbot! Type 'exit' to end the conversation.")
    print("Chatbot: You can ask me to perform calculations or chat with me")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        print("\nChatbot: ", end="")
        for chunk in agent_executor.stream({"messages": [HumanMessage(content=user_input)]}):
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    print(message.content, end="")
        print()
    print("Thank you for using the chatbot!")

if __name__ == "__main__":
    main()