import streamlit as st
from openai import OpenAI
import json

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

# === Załaduj dane z plików JSON ===
@st.cache_data
def load_data():
    with open("data/recipes_100.json", "r", encoding="utf-8") as f:
        recipes = json.load(f)
    with open("data/skladniki_baza.json", "r", encoding="utf-8") as f:
        skladniki = json.load(f)
    return recipes, skladniki

recipes, skladniki = load_data()

# === Funkcja wyszukiwania informacji ===
def find_relevant_info(question):
    question_lower = question.lower()

    # Filtruj przepisy — lista słowników
    relevant_recipes = [
        r for r in recipes
        if question_lower in r.get("title", "").lower() or question_lower in r.get("ingredients", "").lower()
    ]

    # Filtruj składniki — skladniki to słownik z listą pod kluczem "skladniki"
    relevant_skladniki = []
    for s in skladniki.get("skladniki", []):
        nazwa = s.get("nazwa", "").lower()
        synonimy = [syn.lower() for syn in s.get("synonimy", [])]
        if question_lower in nazwa or question_lower in synonimy:
            relevant_skladniki.append(s)

    results = []
    if relevant_recipes:
        results.append(
            "Przepisy:\n" + "\n".join(
                [f'{r.get("title", "")}: {r.get("ingredients", "")}' for r in relevant_recipes]
            )
        )
    if relevant_skladniki:
        results.append(
            "Składniki:\n" + "\n".join(
                [f'{s.get("nazwa", "")} - kalorie na 100g: {s.get("kalorie_na_100g", "brak danych")}' for s in relevant_skladniki]
            )
        )

    return "\n\n".join(results) if results else None  # Zwracamy None jeśli brak wyników

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

    # Najpierw sprawdź dane lokalne
    context = find_relevant_info(prompt)

    # Jeśli brak lokalnych danych, zapytaj model GPT bez dodatkowego kontekstu
    if context is None:
        system_msgs = [
            {"role": "system", "content": "Jesteś pomocnym asystentem."}
        ]
    else:
        # Jeśli mamy kontekst, podaj go jako dodatkową informację do modelu
        system_msgs = [
            {"role": "system", "content": "Jesteś pomocnym asystentem."},
            {"role": "system", "content": f"Oto dodatkowe informacje, które mogą pomóc:\n{context}"}
        ]

    try:
        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct:free",
            messages=system_msgs + [{"role": "user", "content": prompt}],
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
