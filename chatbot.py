from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

load_dotenv()  # take environment variables from .env.

client = OpenAI()

initial_message = [
        {"role": "system", "content": "your are a trip planner in Dubai. Yor are a expert in Dubai Torisum Locations, Food, Events, Hotels, etc. You are able to guide users to plan their vacation to Dubai. You should response professionally. your name is Dubai Genei, Short name is DG. Response shouldn't exceed 200 words. ALways ask questions to user and help them to plan a trip. Finally give a day wise itinerary. Deal with user professionally"},
        {
            "role": "assistant",
            "content": "Hello I am Dubai Geni, your Expert Trip Planner. How Can I Help You.? "
        }
]

def get_response_from_llm(messages):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages 
    )

    return completion.choices[0].message.content

if "messages" not in st.session_state:
    st.session_state.messages = initial_message

st.title("Dubai Trip Assistant App")

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


user_message = st.chat_input("Enter your Message")
if user_message:
    new_message = {
            "role": "user",
            "content": user_message
        }
    st.session_state.messages.append(new_message)
    with st.chat_message(new_message["role"]):
            st.markdown(new_message["content"])

    response = get_response_from_llm(st.session_state.messages)
    if response:
        response_message = {
            "role": "assistant",
            "content": response
        }
        st.session_state.messages.append(response_message)
        with st.chat_message(response_message["role"]):
            st.markdown(response_message["content"])

