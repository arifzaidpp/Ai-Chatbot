import streamlit as st
from dotenv import load_dotenv
import os
from openai import OpenAI
from openai import RateLimitError
import requests
import json

load_dotenv()  # take environment variables from .env

# Initialize providers
openai_client = OpenAI()

st.set_page_config(page_title="Dubai Trip Assistant", page_icon="🧞", layout="wide")

# Fetch available Ollama models
def get_ollama_models():
    try:
        response = requests.get(f"{st.session_state.ollama_url}/api/tags")
        if response.status_code == 200:
            models = [model["name"] for model in response.json()["models"]]
            return models if models else ["no-models-found"]
        else:
            return ["ollama-connection-error"]
    except Exception:
        return ["ollama-connection-error"]

# Initialize session state for Ollama URL
if "ollama_url" not in st.session_state:
    st.session_state.ollama_url = "http://localhost:11434"

# AI provider options
AI_PROVIDERS = {
    "OpenAI": ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"],
    "Anthropic": ["claude-3-haiku", "claude-3-sonnet", "claude-3-opus"],
    "DeepSeek": ["deepseek-chat", "deepseek-coder"],
}

# Initial configuration for session state
if "ai_provider" not in st.session_state:
    st.session_state.ai_provider = "OpenAI"
if "ai_model" not in st.session_state:
    st.session_state.ai_model = "gpt-4o-mini"
if "ollama_models_fetched" not in st.session_state:
    st.session_state.ollama_models_fetched = False
if "error_shown" not in st.session_state:
    st.session_state.error_shown = False

# Update AI_PROVIDERS with Ollama models
if not st.session_state.ollama_models_fetched:
    ollama_models = get_ollama_models()
    AI_PROVIDERS["Ollama"] = ollama_models
    st.session_state.ollama_models_fetched = True

initial_message = [
    {"role": "system", "content": "You are a trip planner in Dubai. You're an expert in Dubai Tourism Locations, Food, Events, Hotels, etc. You are able to guide users to plan their vacation to Dubai. You should respond professionally. Your name is Dubai Genei, short name is DG. Response shouldn't exceed 200 words. Always ask questions to users and help them plan a trip. Finally give a day-wise itinerary. Deal with users professionally."},
    {
        "role": "assistant",
        "content": "Hello, I am Dubai Genei, your Expert Trip Planner. How can I help you plan your Dubai adventure?"
    }
]

if "messages" not in st.session_state:
    st.session_state.messages = initial_message

# Function to get response from OpenAI
def get_openai_response(messages, model):
    try:
        completion = openai_client.chat.completions.create(
            model=model,
            messages=messages
        )
        return completion.choices[0].message.content, False
    except RateLimitError:
        return "⚠️ Rate limit hit or insufficient quota. Please check your OpenAI account.", True
    except Exception as e:
        return f"⚠️ Error: {str(e)}", True

# Function to get response from Anthropic
def get_anthropic_response(messages, model):
    try:
        # Convert messages to Anthropic format
        formatted_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                formatted_messages.append({"role": msg["role"], "content": msg["content"]})
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": os.getenv("ANTHROPIC_API_KEY")
        }
        
        data = {
            "model": model,
            "messages": formatted_messages,
            "system": system_content,
            "max_tokens": 1000
        }
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            return response.json()["content"][0]["text"], False
        else:
            return f"⚠️ Error: {response.text}", True
    except Exception as e:
        return f"⚠️ Error: {str(e)}", True

# Function to get response from DeepSeek
def get_deepseek_response(messages, model):
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": 1000
        }
        
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"], False
        else:
            return f"⚠️ Error: {response.text}", True
    except Exception as e:
        return f"⚠️ Error: {str(e)}", True

# Function to get response from Ollama
def get_ollama_response(messages, model):
    try:
        # Convert to Ollama format
        ollama_messages = []
        system_content = ""
        
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                ollama_messages.append({"role": msg["role"], "content": msg["content"]})
                
        data = {
            "model": model,
            "messages": ollama_messages,
            "system": system_content,
            "stream": False
        }
        
        response = requests.post(
            f"{st.session_state.ollama_url}/api/chat",
            json=data
        )
        
        if response.status_code == 200:
            return response.json()["message"]["content"], False
        else:
            return f"⚠️ Error connecting to Ollama: {response.text}", True
    except Exception as e:
        return f"⚠️ Error: {str(e)}", True

# Function to route to the correct AI provider
def get_response_from_llm(messages, provider=None, model=None):
    provider = provider or st.session_state.ai_provider
    model = model or st.session_state.ai_model
    
    # Remove system message for display but keep it for API calls
    display_messages = [m for m in messages if m["role"] != "system"]
    
    if provider == "OpenAI":
        return get_openai_response(messages, model)
    elif provider == "Anthropic":
        return get_anthropic_response(messages, model)
    elif provider == "DeepSeek":
        return get_deepseek_response(messages, model)
    elif provider == "Ollama":
        return get_ollama_response(messages, model)
    else:
        return "⚠️ Unknown AI provider selected.", True

# Function to refresh Ollama models
def refresh_ollama_models():
    st.session_state.ollama_models_fetched = False
    ollama_models = get_ollama_models()
    AI_PROVIDERS["Ollama"] = ollama_models
    st.session_state.ollama_models_fetched = True
    
    # If the current model is not in the refreshed list, select the first available
    if st.session_state.ai_provider == "Ollama" and st.session_state.ai_model not in ollama_models:
        st.session_state.ai_model = ollama_models[0]

# UI layout
st.title("Dubai Trip Assistant")

# Sidebar for AI provider settings
with st.sidebar:
    st.header("AI Provider Settings")
    
    # AI provider selection
    new_provider = st.selectbox(
        "Select AI Provider",
        options=list(AI_PROVIDERS.keys()),
        index=list(AI_PROVIDERS.keys()).index(st.session_state.ai_provider)
    )
    
    # Update models if provider changed
    if new_provider != st.session_state.ai_provider:
        st.session_state.ai_provider = new_provider
        if new_provider == "Ollama" and "ollama-connection-error" in AI_PROVIDERS["Ollama"]:
            # Try to refresh Ollama models when switching to Ollama
            refresh_ollama_models()
        st.session_state.ai_model = AI_PROVIDERS[new_provider][0]
    
    # Model selection based on provider
    available_models = AI_PROVIDERS[st.session_state.ai_provider]
    model_index = 0
    if st.session_state.ai_model in available_models:
        model_index = available_models.index(st.session_state.ai_model)
    
    st.session_state.ai_model = st.selectbox(
        "Select Model",
        options=available_models,
        index=model_index
    )
    
    # Ollama specific settings
    if st.session_state.ai_provider == "Ollama":
        col1, col2 = st.columns([3, 1])
        with col1:
            new_ollama_url = st.text_input(
                "Ollama Server URL", 
                value=st.session_state.ollama_url
            )
        with col2:
            refresh_button = st.button("Refresh Models")
            
        if new_ollama_url != st.session_state.ollama_url:
            st.session_state.ollama_url = new_ollama_url
            refresh_ollama_models()
            
        if refresh_button:
            refresh_ollama_models()
            st.success("Models refreshed!")
            
        if "ollama-connection-error" in AI_PROVIDERS["Ollama"]:
            st.error("⚠️ Cannot connect to Ollama server. Check if Ollama is running.")
        elif "no-models-found" in AI_PROVIDERS["Ollama"]:
            st.warning("⚠️ No models found in Ollama. Please install models with 'ollama pull model_name'.")
    
    # Reset conversation button
    if st.button("Reset Conversation"):
        st.session_state.messages = initial_message
        st.session_state.error_shown = False
        st.rerun()

    st.markdown("---")
    st.markdown("### Current Settings")
    st.markdown(f"**Provider:** {st.session_state.ai_provider}")
    st.markdown(f"**Model:** {st.session_state.ai_model}")
    if st.session_state.ai_provider == "Ollama":
        st.markdown(f"**Ollama URL:** {st.session_state.ollama_url}")

# Main chat area
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

# User input
user_message = st.chat_input("Type your message here...")
if user_message:
    # Reset error shown flag when user sends a new message
    st.session_state.error_shown = False
    
    new_message = {
        "role": "user",
        "content": user_message
    }
    st.session_state.messages.append(new_message)
    
    with st.chat_message("user"):
        st.markdown(user_message)
    
    # Get response from selected AI provider
    with st.spinner("Dubai Genei is thinking..."):
        response, is_error = get_response_from_llm(st.session_state.messages)
    
    # Only add assistant response to messages if not an error or if error not shown yet
    if not is_error or not st.session_state.error_shown:
        response_message = {
            "role": "assistant",
            "content": response
        }
        
        # Only append to messages if not an error
        if not is_error:
            st.session_state.messages.append(response_message)
        else:
            st.session_state.error_shown = True
            
        with st.chat_message("assistant"):
            st.markdown(response)