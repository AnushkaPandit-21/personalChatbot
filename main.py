from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

def create_model():
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        streaming=True,
    )
    return model

def main():
    model = create_model()
    messages = []

    print("Welcome to the chatbot! Type 'exit' to end the conversation.")
    print("Chatbot: I'm you personal Chatbot. Ask me anything.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # 1) Add the new user message to history
        messages.append(HumanMessage(content=user_input))

        print("\nChatbot: ", end="", flush=True)

        # 2) Stream the model's response token by token
        full_text = ""
        for chunk in model.stream(messages):
            # Each chunk is a partial message; chunk.content is the new text
            text = chunk.content or ""
            print(text, end="", flush=True)
            full_text += text

        # 3) After streaming finishes, store the assistant's reply in history
        messages.append(AIMessage(content=full_text))

    print("\nThank you for using the chatbot!")


if __name__ == "__main__":
    main()
