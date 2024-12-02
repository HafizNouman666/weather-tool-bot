from langchain_openai import OpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
import streamlit as st
from gtts import gTTS
import requests
import os
import json
import tempfile  
import pygame  
import io
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
weathermap_api_key = os.getenv("WEATHERMAP_API_KEY")

# Step 1: Weather API function
def get_current_weather(latitude, longitude):
    """Get the current weather in a given latitude and longitude"""
    base = "https://api.openweathermap.org/data/2.5/weather"
    request_url = f"{base}?lat={latitude}&lon={longitude}&appid={weathermap_api_key}&units=metric"
    response = requests.get(request_url)
    
    if response.status_code == 200:
        result = {
            "latitude": latitude,
            "longitude": longitude,
            **response.json()["main"]
        }
        return json.dumps(result, indent=4)
    else:
        return f"Failed to fetch weather data. Status Code: {response.status_code}"

def text_to_speech(text):
    """Convert text to speech and play it without using pygame."""
    try:
        tts = gTTS(text=text, lang="en")
        
        # Save audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio_file:
            tts.save(temp_audio_file.name)
            
            # Play the audio in Streamlit
            st.audio(temp_audio_file.name , autoplay=True)
    
    except Exception as e:
        st.error(f"Error during TTS playback: {e}")


# Step 2: Wrap it in a LangChain Tool
weather_tool = Tool(
    name="current_weather_tool",
    func=lambda location: get_current_weather(
        *map(float, location.strip().replace("'", "").split(","))
    ),
    description=(
        "Fetches the real-time weather data for a given latitude and longitude. "
        "Provide input as 'latitude,longitude' (e.g., '33.6995,73.0363'). or location, area , place "
        "if user ask for weather you ask the location which they live. "
    )
)

general_tool = Tool(
    name="general_conversation",
    func=lambda query: f"I'm here to help you! how i can assist you about weather? ",
    description=(
        "you are general weather agent who Handles general conversation queries that don't require tool usage."
        "your response is short and friendly."
        "handle all the weather related question which are possible if user ask any question about anything your job is to response according to weather conditions"
    )
)



# Step 3: Initialize Memory
memory = ConversationBufferMemory(return_messages=True)

# Step 4: Chat Agent Setup
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
tools = [weather_tool , general_tool ]

# Create an agent with error handling for parsing
agent_chain = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=False,
    handle_parsing_errors=True  # Enables retry for parsing errors
)


page_bg_img = """
  <style>
  [data-testid="stAppViewContainer"]{
    background-image: url("https://images.unsplash.com/photo-1500964757637-c85e8a162699?q=80&w=1806&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: local;
}

[data-testid="stHeader"] {
background: rgba(0,0,0,0);
}
  </style>
""" 
st.markdown(page_bg_img , unsafe_allow_html = True)

st.title("Weather Chatbot 🌦️")
st.write("Hi! 👋 Ask me about the weather")

if "message" not in st.session_state:
    st.session_state.message = []
if "query" not in st.session_state:
    st.session_state.query = ""

# Function to process user query
def handle_query():
    user_query = st.session_state.query.strip()
    #print("-----user_query------",user_query)
    if user_query:
        st.session_state.message.append({"role": "user", "content": user_query})

        response_stream = agent_chain.run(input=user_query)
        #print("---------response--------------", response_stream)
        bot_response = ""
        if hasattr(response_stream, "__iter__") and not isinstance(response_stream, str):
            for chunk in response_stream:
                if hasattr(chunk, "choices") and hasattr(chunk.choices[0], "delta") and hasattr(chunk.choices[0].delta, "content"):
                    bot_response += chunk.choices[0].delta.content
        else:
            bot_response = response_stream 

        st.session_state.message.append({"role": "assistant", "content": bot_response})

        st.session_state.bot_response = bot_response

        st.session_state.query = ""


with st.container():
    st.write("### Conversation:")
    for chat in st.session_state.message:
        if chat["role"] == "user":
            st.markdown(f"👤 **You**: {chat['content']}")
        else:
            st.markdown(f"🤖 **Bot**: {chat['content']}")

st.text_input(
    "Your Query:",
    placeholder="E.g., What's the weather in Islamabad?",
    key="query",
    on_change=handle_query
)

if "bot_response" in st.session_state and st.session_state.bot_response:
    text_to_speech(st.session_state.bot_response)
    st.session_state.bot_response = "" 