import streamlit as st
from openai import OpenAI

st.title("OpenRouter Chatbot Demo")

api_key = st.secrets["API_KEY"]
base_url = st.secrets.get("BASE_URL", "https://openrouter.ai/api/v1")
model_name = st.secrets.get("MODEL", "mistralai/mistral-7b-instruct:free")

client = OpenAI(api_key=api_key, base_url=base_url)

if prompt := st.text_input("Zadaj pytanie"):
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        extra_headers={
            "HTTP-Referer": "https://twoja-apka.streamlit.app",
            "X-Title": "OpenRouter Chatbot Demo"
        }
    )
    st.write(response.choices[0].message.content)

