from model import generate_text, generate_story, write_email, brainstorm_ideas

def main():
    print("\n=====================================================")
    print("           DistilGPT-2 Interactive Tester            ")
    print("=====================================================")
    print("Available Tasks:")
    print("  1. Generic Text Generation")
    print("  2. Story Generation")
    print("  3. Email Writing")
    print("  4. Brainstorming")
    print("\nType 'exit' or 'quit' at any prompt to stop.")
    
    while True:
        print("\n" + "=" * 53)
        choice = input("\nSelect a task (1-4): ").strip()
        
        if choice.lower() in ["exit", "quit"]:
            break
            
        if choice == "1":
            prompt = input("Enter prompt: ").strip()
            if prompt.lower() in ["exit", "quit"]: break
            print("\nGenerating...")
            print("-" * 60)
            print(generate_text(prompt))
            print("-" * 60)
            
        elif choice == "2":
            prompt = input("Enter story topic: ").strip()
            if prompt.lower() in ["exit", "quit"]: break
            print("\nGenerating...")
            print("-" * 60)
            print(generate_story(prompt))
            print("-" * 60)
            
        elif choice == "3":
            topic = input("Enter email topic: ").strip()
            if topic.lower() in ["exit", "quit"]: break
            recipient = input("Enter recipient name (default: friend): ").strip()
            if recipient.lower() in ["exit", "quit"]: break
            if not recipient:
                recipient = "friend"
            print("\nGenerating...")
            print("-" * 60)
            print(write_email(topic, recipient=recipient))
            print("-" * 60)
            
        elif choice == "4":
            topic = input("Enter brainstorming topic: ").strip()
            if topic.lower() in ["exit", "quit"]: break
            print("\nGenerating...")
            print("-" * 60)
            print(brainstorm_ideas(topic))
            print("-" * 60)
            
        else:
            print("Invalid choice. Please select 1, 2, 3, or 4.")

if __name__ == "__main__":
    # We delay loading message until user actually selects something, 
    # but since transformers loads on import in model.py, it will load on startup.
    print("Model loaded successfully!")
    main()
