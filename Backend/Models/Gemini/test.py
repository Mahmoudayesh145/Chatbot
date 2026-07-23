from model import get_chat_session

def main():
    print("\n=====================================================")
    print("             Gemini API Chatbot Tester               ")
    print("=====================================================")
    print("Type your messages below to chat with Gemini!")
    print("It will remember your conversation history.")
    print("Type 'exit' or 'quit' to end the chat.")
    print("=====================================================\n")
    
    chat = get_chat_session()
    if not chat:
        print("Failed to start chat session. Check your API key!")
        return
        
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ["exit", "quit"]:
            print("\nEnding chat. Goodbye!")
            break
            
        if not user_input:
            continue
            
        print("Gemini is typing...")
        try:
            # send_message automatically updates the chat history
            response = chat.send_message(user_input)
            print(f"\nGemini: {response.text}")
        except Exception as e:
            print(f"\n[Error communicating with Gemini: {e}]")
            
        print("-" * 60)

if __name__ == "__main__":
    main()
