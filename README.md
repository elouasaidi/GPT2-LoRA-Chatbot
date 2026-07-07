# GPT-2 Fine-Tuning with LoRA for Conversational AI

## Overview

This project demonstrates the fine-tuning of the **GPT-2** language model using **Low-Rank Adaptation (LoRA)**, a Parameter-Efficient Fine-Tuning (PEFT) technique for Large Language Models (LLMs).

The objective is to build a conversational AI system capable of generating coherent responses while significantly reducing the number of trainable parameters compared to traditional full-model fine-tuning.

The project covers the complete workflow, from dataset preparation and model training to inference and deployment through a Flask-based web application.

---

## Objectives

- Fine-tune GPT-2 using the LoRA technique.
- Reduce computational cost through Parameter-Efficient Fine-Tuning (PEFT).
- Build a conversational AI chatbot.
- Compare model behavior before and after fine-tuning.
- Deploy the trained model using a simple web interface.

---

## Features

- GPT-2 fine-tuning with LoRA
- Hugging Face Transformers integration
- PEFT implementation
- Dataset preprocessing and tokenization
- Model training and evaluation
- Loss visualization
- Text generation
- Interactive chatbot interface using Flask

---

## Project Structure

```text
PROJET_DL/
│
├── images/
│   ├── Fine-Tuning-Process.jpg
│   ├── Image1.jpg
│   ├── Image2.jpg
│   ├── Image3.png
│   ├── image4.png
│   ├── intro.jpg
│   ├── LORA.png
│   └── loss.png
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── app.py
├── DL.ipynb
├── DL_1.ipynb
├── test.py
├── requirements.txt
├── .hintrc
└── README.md
```

---

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- LoRA
- Datasets
- Flask
- Jupyter Notebook

---

## Model

**Base Model**

- GPT-2

**Fine-Tuning Method**

- Low-Rank Adaptation (LoRA)

LoRA freezes the original model weights and introduces trainable low-rank matrices, allowing efficient adaptation with significantly fewer trainable parameters.

---

## Dataset

The project uses the **Synthetic Persona-Chat** dataset provided by Hugging Face.

The dataset contains conversational exchanges between personas and is designed for training dialogue generation models.

---

## Training Pipeline

The workflow consists of the following steps:

1. Load the GPT-2 tokenizer and model.
2. Load the conversational dataset.
3. Tokenize the dialogue data.
4. Configure LoRA parameters.
5. Fine-tune GPT-2 using PEFT.
6. Save the trained model.
7. Evaluate the model.
8. Generate responses.
9. Deploy the chatbot using Flask.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your_username/GPT2-LoRA-Chatbot.git
```

Move into the project directory:

```bash
cd PROJET_DL
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Train the model

Open and execute:

```text
DL.ipynb
```

### Run the chatbot

```bash
python app.py
```

Open your browser and navigate to:

```
http://127.0.0.1:5000
```

---

## Training Configuration

| Parameter | Value |
|------------|------:|
| Base Model | GPT-2 |
| Fine-Tuning Method | LoRA |
| Framework | PyTorch |
| Library | Transformers |
| PEFT | Yes |
| Dataset | Synthetic Persona-Chat |

---

## Results

The project demonstrates:

- Successful fine-tuning of GPT-2 using LoRA.
- Reduced number of trainable parameters.
- Improved conversational response generation.
- Stable training process through loss monitoring.
- Interactive chatbot deployment using Flask.

---

## Applications

This project can be applied to:

- Conversational AI
- Customer support chatbots
- Virtual assistants
- Natural Language Processing
- Large Language Model adaptation
- Efficient fine-tuning of foundation models

---

## Future Improvements

Possible extensions include:

- Fine-tuning larger language models (GPT-Neo, LLaMA, Mistral)
- Quantized LoRA (QLoRA)
- Retrieval-Augmented Generation (RAG)
- Multi-turn conversation memory
- Streamlit or Gradio deployment
- Hugging Face Spaces deployment
- Docker containerization

---

## Author

**Fatima El Ouasaidi**

Master's Degree in Data Science and Intelligent Systems

Specialization in Artificial Intelligence, Machine Learning, Deep Learning, Natural Language Processing, and Large Language Models.