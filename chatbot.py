import streamlit as st
from dotenv import load_dotenv
import os
from openai import OpenAI
from openai import RateLimitError
import requests
import json
import re

load_dotenv()  # take environment variables from .env

# Initialize providers
openai_client = OpenAI()

st.set_page_config(page_title="Dubai Trip Assistant", page_icon="🧞", layout="wide")

# Initialize session state for Ollama URL
if "ollama_url" not in st.session_state:
    st.session_state.ollama_url = "http://localhost:11434"

# Fetch available Ollama models
def get_ollama_models():
    try:
        response = requests.get(f"{st.session_state.ollama_url}/api/tags")
        if response.status_code == 200:
            models = [model["name"] for model in response.json()["models"]]
            if models:
                return models
        return ["no-models-found"]
    except Exception:
        return ["ollama-connection-error"]

# Base AI provider options (without Ollama models)
BASE_PROVIDERS = {
    "OpenAI": ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"],
    "Anthropic": ["claude-3-haiku", "claude-3-sonnet", "claude-3-opus"],
    "DeepSeek": ["deepseek-chat", "deepseek-coder"],
}

# Initialize AI_PROVIDERS with base providers
if "ai_providers" not in st.session_state:
    st.session_state.ai_providers = BASE_PROVIDERS.copy()
    # Add Ollama with placeholder until we fetch models
    st.session_state.ai_providers["Ollama"] = ["fetching-models..."]

# Initial configuration for other session state variables
if "ai_provider" not in st.session_state:
    st.session_state.ai_provider = "OpenAI"
if "ai_model" not in st.session_state:
    st.session_state.ai_model = "gpt-4o-mini"
if "error_shown" not in st.session_state:
    st.session_state.error_shown = False
if "show_thinking" not in st.session_state:
    st.session_state.show_thinking = True

# Updated system prompt to include thinking instruction
system_prompt = """You are a trip planner in Dubai. You're an expert in Dubai Tourism Locations, Food, Events, Hotels, etc. 
You are able to guide users to plan their vacation to Dubai. You should respond professionally. 
Your name is Dubai Genei, short name is DG. Response shouldn't exceed 200 words. 
Always ask questions to users and help them plan a trip. Finally give a day-wise itinerary. Deal with users professionally.

IMPORTANT: When you need to think through a response, especially for complex questions about Dubai attractions, 
hotel recommendations, or creating itineraries, place your thought process between <think> and </think> tags.
This helps the user understand your reasoning. For example:

<think>
This user is looking for a 3-day itinerary focused on family activities. Dubai has several family-friendly attractions 
like Dubai Parks and Resorts, Atlantis Aquaventure, and the Dubai Aquarium. I'll organize these by location to minimize travel time.
</think>

Your final response should follow after your thinking process.
"""

# Function to refresh Ollama models
def refresh_ollama_models():
    ollama_models = get_ollama_models()
    # Update the Ollama models in our providers dictionary
    st.session_state.ai_providers["Ollama"] = ollama_models
    
    # If the current provider is Ollama and model is not available, select the first one
    if st.session_state.ai_provider == "Ollama" and st.session_state.ai_model not in ollama_models:
        st.session_state.ai_model = ollama_models[0]

# Refresh Ollama models at startup
refresh_ollama_models()

initial_message = [
    {"role": "system", "content": system_prompt},
    {
        "role": "assistant",
        "content": "Hello, I am Dubai Genei, your Expert Trip Planner. How can I help you plan your Dubai adventure?"
    }
]

if "messages" not in st.session_state:
    st.session_state.messages = initial_message

# Function to parse thinking and response
def parse_thinking(response):
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, response, re.DOTALL)
    
    if think_match:
        thinking = think_match.group(1).strip()
        # Remove the thinking part from the response
        final_response = re.sub(think_pattern, '', response, flags=re.DOTALL).strip()
        return thinking, final_response
    else:
        return None, response

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
        system_content = ""
        
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

# Set custom CSS for the thinking section
st.markdown("""
<style>
    .thinking-box {
        background-color: #f0f7ff;
        border-left: 5px solid #1e88e5;
        padding: 10px 15px;
        margin-bottom: 10px;
        border-radius: 5px;
    }
    .thinking-header {
        color: #1e88e5;
        font-weight: bold;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# UI layout
st.title("Dubai Trip Assistant")

# Sidebar for AI provider settings
with st.sidebar:
    st.header("AI Provider Settings")
    
    # AI provider selection
    new_provider = st.selectbox(
        "Select AI Provider",
        options=list(st.session_state.ai_providers.keys()),
        index=list(st.session_state.ai_providers.keys()).index(st.session_state.ai_provider)
    )
    
    # Update models if provider changed
    if new_provider != st.session_state.ai_provider:
        st.session_state.ai_provider = new_provider
        # If switching to Ollama, refresh models
        if new_provider == "Ollama":
            refresh_ollama_models()
        # Set default model for the new provider
        st.session_state.ai_model = st.session_state.ai_providers[new_provider][0]
    
    # Model selection based on provider
    available_models = st.session_state.ai_providers[st.session_state.ai_provider]
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
            refresh_button = st.button("Refresh")
            
        if new_ollama_url != st.session_state.ollama_url:
            st.session_state.ollama_url = new_ollama_url
            refresh_ollama_models()
            
        if refresh_button:
            refresh_ollama_models()
            st.success("Models refreshed!")
            
        # Show appropriate warnings for Ollama
        ollama_models = st.session_state.ai_providers["Ollama"]
        if "ollama-connection-error" in ollama_models:
            st.error("⚠️ Cannot connect to Ollama server. Check if Ollama is running.")
        elif "no-models-found" in ollama_models:
            st.warning("""
            ⚠️ No models found in Ollama. 
            
            Install models with: `ollama pull modelname`
            
            Example models: llama3, mistral, gemma, phi3
            """)
    
    # Display options
    st.header("Display Settings")
    st.session_state.show_thinking = st.checkbox("Show AI Thinking Process", value=st.session_state.show_thinking)
    
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
            # Check if this message has thinking content
            if message["role"] == "assistant" and "<think>" in message["content"] and "</think>" in message["content"]:
                thinking, response = parse_thinking(message["content"])
                
                # Display in chat
                with st.chat_message("assistant"):
                    # Show thinking process if enabled
                    if thinking and st.session_state.show_thinking:
                        st.markdown(f"""
                        <div class="thinking-box">
                            <div class="thinking-header">🧠 AI Thinking Process:</div>
                            {thinking}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show final response
                    st.markdown(response)
            else:
                # Regular message display
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
            
            # Parse and display thinking and response
            thinking, final_response = parse_thinking(response)
            
            with st.chat_message("assistant"):
                # Show thinking process if available and enabled
                if thinking and st.session_state.show_thinking:
                    st.markdown(f"""
                    <div class="thinking-box">
                        <div class="thinking-header">🧠 AI Thinking Process:</div>
                        {thinking}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show final response
                st.markdown(final_response)
        else:
            st.session_state.error_shown = True
            with st.chat_message("assistant"):
                st.markdown(response)