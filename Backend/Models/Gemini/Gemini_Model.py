import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API with the loaded key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key or api_key == "YOUR_API_KEY_HERE":
    print("WARNING: GEMINI_API_KEY is not set or is still the default value.")
    print("Please open the .env file and paste your actual Gemini API key.")

genai.configure(api_key=api_key)

# We'll use the recommended standard model for general text tasks
model = genai.GenerativeModel("gemini-2.5-flash")

def get_chat_session():
    """Starts and returns a new Gemini chat session that remembers history."""
    try:
        # Start a chat with an empty history
        chat = model.start_chat(history=[])
        return chat
    except Exception as e:
        print(f"Error starting chat: {e}")
        return None
