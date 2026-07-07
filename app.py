from flask import Flask, request, jsonify, render_template
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel
import time

app = Flask(__name__, static_folder='static', template_folder='templates')

# ===== CONFIG =====
MODEL_PATH = "./chatbot-gpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ===== LOAD MODEL =====
print("⚡ Loading model...")
try:
    # Base model
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    base_model.config.pad_token_id = base_model.config.eos_token_id
    
    # Fine-tuned model
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model = model.to(device)
    model.eval()
    
    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Model loaded on {device.upper()}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    tokenizer = None

# ===== GENERATION FUNCTION =====
def generate_response(user_input, max_length=100):
    """Generate bot response with your format"""
    if model is None or tokenizer is None:
        return "⚠️ Model not available. Please check server logs."
    
    try:
        model.eval()
        
        # Format prompt with Fatima/GBT-2 format
        prompt_text = f"Fatima: {user_input}\nGBT-2:"
        
        # Tokenize
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        
        # Generate with your parameters
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only bot response
        if "GBT-2:" in full_text:
            # Take everything after GBT-2: and before next Fatima:
            parts = full_text.split("GBT-2:")
            if len(parts) > 1:
                response = parts[1].split("Fatima:")[0].strip()
            else:
                response = full_text.replace(prompt_text, "").strip()
        else:
            response = full_text.replace(prompt_text, "").strip()
        
        # Clean response
        response = clean_response(response)
        
        return response
        
    except Exception as e:
        print(f"Generation error: {e}")
        return f"⚠️ Generation error: {str(e)}"

def clean_response(text):
    """Clean up the generated response"""
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove any remaining prompt parts
    unwanted = ["Fatima:", "User:", "Bot:", "Human:", "Assistant:"]
    for word in unwanted:
        text = text.replace(word, "")
    
    # Trim
    text = text.strip()
    
    # Limit length
    if len(text) > 500:
        text = text[:500] + "..."
    
    return text

# ===== ROUTES =====
@app.route('/')
def home():
    """Home page - serve HTML template"""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy" if model else "error",
        "device": device,
        "model_loaded": model is not None,
        "model_name": "GPT-2 Fine-tuned with Fatima/GBT-2 format"
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Chat API endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        user_message = data.get("message", "").strip()
        max_length = data.get("max_length", 100)
        
        if not user_message:
            return jsonify({"error": "Message is empty"}), 400
        
        # Generate response
        start_time = time.time()
        bot_response = generate_response(user_message, max_length=max_length)
        response_time = time.time() - start_time
        
        return jsonify({
            "user_message": user_message,
            "bot_response": bot_response,
            "response_time": round(response_time, 2),
            "format": "Fatima/GBT-2"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===== TEST ENDPOINT =====
@app.route('/test', methods=['GET'])
def test():
    """Test the model with a sample prompt"""
    if model is None:
        return jsonify({"error": "Model not loaded"})
    
    test_prompts = [
        "Hello, how are you?",
        "What's your favorite hobby?",
        "Tell me about yourself",
        "What do you think about AI?"
    ]
    
    results = []
    for prompt in test_prompts:
        response = generate_response(prompt, max_length=80)
        results.append({
            "prompt": prompt,
            "response": response
        })
    
    return jsonify({
        "model": "GPT-2 Fine-tuned",
        "format": "Fatima: prompt\nGBT-2: response",
        "tests": results
    })

# ===== START SERVER =====
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)