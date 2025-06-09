import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from openai import OpenAI

# Konfiguracja strony
st.set_page_config(page_title="Chatbot z RAG", layout="wide")
st.title("💬 Chatbot z RAG (Retrieval-Augmented Generation)")

# Ustawienia API
api_key = st.secrets.get("API_KEY")
if not api_key:
    st.error("Brakuje `API_KEY` w pliku `.streamlit/secrets.toml`.")
    st.stop()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# === RAG – Ładowanie danych i przygotowanie FAISS ===
@st.cache_resource
def load_rag(data_dir="data"):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = []
    embeddings = []

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        st.warning("Utworzono pusty katalog `data/`. Dodaj tam pliki .txt.")
        return None, [], model

    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            path = os.path.join(data_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                texts.append(text)
                emb = model.encode(text)
                embeddings.append(emb)

    if not embeddings:
        st.warning("Brak plików .txt w katalogu `data/`.")
        return None, [], model

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index, texts, model

index, docs, embedder = load_rag()

# === Historia czatu ===
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Cześć! Zadaj pytanie na podstawie dokumentów w folderze `data/`."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# === Wprowadzenie pytania ===
if prompt := st.chat_input("Zadaj pytanie"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Wyszukiwanie kontekstu
    context = "Brak dopasowanych dokumentów."
    if index:
        q_vec = embedder.encode([prompt])
        D, I = index.search(np.array(q_vec), k=2)
        context = "\n\n".join([docs[i] for i in I[0]])

    # Tworzenie zapytania z kontekstem
    full_prompt = f"""
Oto informacje z dokumentów:

{context}

Pytanie: {prompt}
"""

    try:
        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct:free",
            messages=[
                {"role": "system", "content": "Odpowiadasz na podstawie kontekstu."},
                {"role": "user", "content": full_prompt}
            ],
            extra_headers={
                "HTTP-Referer": "https://TWOJA-APLIKACJA.streamlit.app",
                "X-Title": "RAG Chatbot"
            }
        )
        msg = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)

    except Exception as e:
        st.error(f"Wystąpił błąd: {e}")

