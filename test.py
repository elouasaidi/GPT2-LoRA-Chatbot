from flask import Flask, request, jsonify, render_template
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel

app = Flask(__name__)

# =================== CHARGEMENT MODÈLE ===================
device = "cuda" if torch.cuda.is_available() else "cpu"

base_model = GPT2LMHeadModel.from_pretrained("gpt2")
model = PeftModel.from_pretrained(base_model, "chatbot-gpt2")
model = model.to(device)
model.eval()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# =================== GÉNÉRATION ===================
def generate_text(prompt,max_length=50, device=device):
    model.eval()
    prompt_text = f"Fatima: {prompt}\nGBT-2:"
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    
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
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extraire uniquement la réponse du bot
    response = full_text.split("GBT-2:")[1].split("Fatima:")[0].strip()
    return response

# =================== ROUTES ===================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"response": "Message vide."})

    response = generate_text(user_message)
    return jsonify({"response": response})

# =================== LANCEMENT ===================
if __name__ == "__main__":
    app.run(debug=True)