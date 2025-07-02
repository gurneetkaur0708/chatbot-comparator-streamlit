import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
import requests
from together import Together

# Load environment variables
load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# Together client for DeepSeek
together_client = Together(api_key=TOGETHER_API_KEY)

def call_gemini(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Gemini Error] {e}"

def call_moonshot(prompt):
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://chatbot.com",
            "X-Title": "ChatbotCompare"
        }
        payload = {
            "model": "moonshotai/kimi-dev-72b:free",
            "messages": [{"role": "user", "content": prompt}]
        }
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        data = res.json()
        return data['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"[Moonshot Error] {e}"

def call_deepseek(prompt):
    try:
        res = together_client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"[DeepSeek Error] {e}"

def final_response_gemini_compare(question, g, m, d):
    try:
        prompt = f"""
You are an intelligent answer aggregator. Below are answers from three AI models for the same question.

Question: {question}

Gemini's Answer:
{g}

Moonshot's Answer:
{m}

DeepSeek's Answer:
{d}

Compare all three responses. Find the facts or statements that are common in at least two answers, and use them to generate a final, concise, and accurate response. Ignore hallucinations or uncommon details.

Return only the final, reliable answer with its accuracy rate with no explanation.
"""
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Final Answer Error] {e}"

# Streamlit UI
st.title("🤖 Chatbot Response Comparator")
question = st.text_input("Enter your question")

if st.button("Generate Response") and question.strip():
    with st.spinner("Calling Gemini..."):
        gemini = call_gemini(question)
    with st.spinner("Calling Moonshot..."):
        moonshot = call_moonshot(question)
    with st.spinner("Calling DeepSeek..."):
        deepseek = call_deepseek(question)
    with st.spinner("Generating Final Answer..."):
        final = final_response_gemini_compare(question, gemini, moonshot, deepseek)

    st.subheader("🔹 Gemini Response")
    st.write(gemini)

    st.subheader("🔹 Moonshot Response")
    st.write(moonshot)

    st.subheader("🔹 DeepSeek Response")
    st.write(deepseek)

    st.subheader("⭐ Final Best Answer")
    st.success(final)
