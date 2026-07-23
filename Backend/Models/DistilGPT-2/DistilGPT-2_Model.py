from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model for DistilGPT-2
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_text(prompt, max_length=100, temperature=0.7):
    """Generic text generation"""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=1,
        top_k=50,
        top_p=0.95
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_story(prompt, max_length=200):
    """Generate a story based on a prompt"""
    story_prompt = f"Write a short story about {prompt}. Once upon a time, "
    return generate_text(story_prompt, max_length=max_length, temperature=0.8)

def write_email(topic, recipient="friend", sender="me"):
    """Write an email based on a topic"""
    email_prompt = f"Write an email to {recipient} about {topic}. \n\nSubject: {topic}\n\nDear {recipient},\n"
    return generate_text(email_prompt, max_length=150, temperature=0.7)

def brainstorm_ideas(topic, num_ideas=3):
    """Brainstorm ideas on a topic"""
    brainstorm_prompt = f"Brainstorm some creative ideas for {topic}:\n1."
    return generate_text(brainstorm_prompt, max_length=150, temperature=0.8)


