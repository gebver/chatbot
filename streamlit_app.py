import streamlit as st
from openai import OpenAI

# Konfiguracja strony
st.set_page_config(page_title="Chatbot", layout="wide")
st.title("💬 Chatbot")

# Ustawienia API
api_key = st.secrets.get("API_KEY")
if not api_key:
    st.error("Brakuje `API_KEY` w pliku `.streamlit/secrets.toml`.")
    st.stop()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# === Historia czatu ===
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Cześć! Jak mogę Ci pomóc?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# === Wprowadzenie pytania ===
if prompt := st.chat_input("Zadaj pytanie"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    try:
        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct:free",
            messages=[
                {"role": "system", "content": "Jesteś pomocnym asystentem."},
                {"role": "user", "content": prompt}
            ],
            extra_headers={
                "HTTP-Referer": "https://TWOJA-APLIKACJA.streamlit.app",
                "X-Title": "Chatbot"
            }
        )
        msg = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)

    except Exception as e:
        st.error(f"Wystąpił błąd: {e}")

