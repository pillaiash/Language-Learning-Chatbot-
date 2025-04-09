# Language Learning Chatbot using OpenRouter API

This project is an AI-powered chatbot that helps users learn a new language through real-time, scene-based conversations. It uses OpenRouter's LLM API for intelligent responses, detects grammar or vocabulary mistakes, logs them to a local database, and summarizes feedback to help users improve over time.

## 🔍 Features

- User onboarding: known language, target language, proficiency level
- Scene-based conversation context (e.g., restaurant, airport)
- Real-time chat using OpenRouter API
- Mistake detection and correction feedback
- Stores chats and mistakes in a local SQLite database
- Clean Gradio-based user interface

## 🛠 Technologies Used

- Python
- OpenRouter API (Mistral 7B Instruct model)
- SQLite
- Gradio
- Requests library

## 📦 Setup Instructions

### 1. Clone the Repository

## Install Dependencies

Make sure Python 3.8+ is installed.
pip install -r requirements.txt


## Set Your OpenRouter API Key

Open chatbot_ui.py and replace the line:
API_KEY = "your_openrouter_api_key_here"


## Run the Chatbot

python chatbot_ui.py


## How It Works
-The chatbot collects user info and selects a roleplay scene. 
-Every input is sent to OpenRouter's LLM API with contextual prompts.
-Responses are logged in a SQLite database (language_chatbot.db).
-Mistakes are detected and tracked during the conversation.

## Future Enhancements
-Real-Time Pronunciation Feedback using speech recognition
-Custom roleplay scenario builder
-Export chat and mistake history to PDF

