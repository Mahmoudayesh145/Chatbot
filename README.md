# Unified ML Chatbot & Processing API

Welcome to the Unified ML Chatbot project! This application provides a comprehensive suite of Machine Learning and Natural Language Processing (NLP) tools, bundled together in a modern web application with a backend API powered by Flask and a rich frontend built with React and Vite.

## Overview

This project serves as an all-in-one platform for interacting with various AI models. It supports everything from real-time text translation and summarization to emotion detection and advanced conversational AI using Google's Gemini.

## Features

* **Text Summarization**: Automatically condense long blocks of text into concise summaries using state-of-the-art transformer models.
* **Language Translation**: Real-time translation between English and Arabic (both EN to AR and AR to EN) utilizing the MarianMT framework.
* **Emotion & Sentiment Analysis**: Detect underlying emotions and general sentiment (positive, negative, neutral) from text inputs using customized machine learning artifacts.
* **Text Generation**: Generate human-like text continuations based on user prompts using the DistilGPT-2 model.
* **Gemini Chat Integration**: Have dynamic, context-aware conversations powered by the Gemini AI API, complete with chat history management.
* **Modern Web Interface**: A beautifully crafted frontend built with React and Vite, featuring smooth animations and a responsive design.

## Architecture

The project is divided into two main components:

### Backend
* Framework: Flask (Python)
* ML Frameworks: Hugging Face Transformers, PyTorch, Scikit-Learn
* API Endpoints:
  - `GET /`: API Health Check
  - `POST /summarize`: Text Summarization
  - `POST /translate`: EN/AR Translation
  - `POST /emotion`: Emotion Detection
  - `POST /sentiment`: Sentiment Analysis
  - `POST /generate`: Text Generation (DistilGPT-2)
  - `POST /chat/gemini`: Conversational AI Chat

### Frontend
* Framework: React with Vite
* Styling: Custom CSS with fluid and animated backgrounds
* Key Components: Welcome Page, Chat Interface, Translation Panel, Model Selector

## Getting Started

### Prerequisites
* Python 3.9+
* Node.js 16+
* NPM or Yarn

### Backend Setup
1. Navigate to the `Backend` directory.
2. Install dependencies using `pip install -r requirements.txt`.
3. Set up your environment variables (e.g., your Gemini API key in a `.env` file).
4. Run the API server:
   ```bash
   python main.py
   ```
   The backend will be available at `http://localhost:8000`.

### Frontend Setup
1. Navigate to the `Frontend` directory.
2. Install the necessary packages using `npm install`.
3. Start the development server:
   ```bash
   npm run dev
   ```
   The application will be accessible at `http://localhost:5173`.

## Disclaimer

Large model weights (e.g., `.safetensors` files) are not tracked in this repository to maintain a lightweight source control history. They will need to be downloaded or generated locally when running the backend for the first time.

## License

This project is licensed for educational and developmental purposes.
