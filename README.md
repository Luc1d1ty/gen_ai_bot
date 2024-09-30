# AI-Powered Chatbot with Custom GPT-2 Model

This project implements an AI chatbot using a fine-tuned GPT-2 model, FastAPI backend, and a modern, responsive frontend. The chatbot is designed to provide intelligent responses based on custom training data.

## Features

- Custom-trained GPT-2 model for domain-specific responses
- FastAPI backend for efficient request handling
- Modern, responsive UI inspired by ChatGPT

## Technologies Used

- Backend: Python, FastAPI, PyTorch, Transformers
- Frontend: HTML5, CSS3, JavaScript
- Model: GPT-2 (fine-tuned)
- Deployment: Render.com
- Other: Pandas, Uvicorn, Hugging Face

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/Luc1d1ty/gen_ai_bot.git
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

## Usage

After starting the server, navigate to `http://localhost:8000` in your web browser to interact with the chatbot.

## Model Training

To retrain the model on custom data:

1. Prepare your Q&A dataset in CSV format
2. Update the file path in `custom_model/fine_tune.py`
3. Run the training script:
   ```
   python custom_model/fine_tune.py
   ```