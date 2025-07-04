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

# --- MODEL CALL FUNCTIONS ---

def call_gemini(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Gemini Error] {e}"

def call_cypher(prompt):
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://chatbot.com",
            "X-Title": "ChatbotCompare"
        }
        payload = {
            "model": "openrouter/cypher-alpha:free",
            "messages": [{"role": "user", "content": prompt}]
        }
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        data = res.json()
        return data['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"[Cypher Error] {e}"

def call_deepseek(prompt):
    try:
        res = together_client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"[DeepSeek Error] {e}"

# --- FINAL AGGREGATED ANSWER USING GEMINI ---
def final_response_gemini_compare(question, g, c, d):
    try:
        prompt = f"""
You are an intelligent AI that compares answers from three different AI models to the same question.

Question: {question}

Gemini's Answer:
{g}

Cypher's Answer:
{c}

DeepSeek's Answer:
{d}

Instructions:
- Find the statements or meanings that are semantically similar in at least two answers.
- Based on those overlapping ideas, generate a final reliable answer.
- Highlight the common parts using **markdown bold** (`**like this**`).
Only output the final answer without similarity score.
"""
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Final Answer Error] {e}"

# --- SIMILARITY CALCULATION LOGIC ---
def calculate_similarity(g, c, d):
    g = g.strip().lower()
    c = c.strip().lower()
    d = d.strip().lower()
    
    match_count = 0
    if g == c:
        match_count += 1
    if g == d:
        match_count += 1
    if c == d:
        match_count += 1

    if match_count == 3:
        return 100
    elif match_count == 2:
        return 66
    elif match_count == 1:
        return 33
    else:
        return 0

# --- SIMILARITY BAR DISPLAY ---
def similarity_bar(similarity):
    green_width = similarity
    red_width = 100 - similarity

    bar_html = f"""
    <div style="width: 100%; background: #ddd; height: 25px; border-radius: 8px; display: flex;">
        <div style="width: {green_width}%; background: #4CAF50; height: 100%; border-top-left-radius: 8px; border-bottom-left-radius: 8px;"></div>
        <div style="width: {red_width}%; background: #F44336; height: 100%; border-top-right-radius: 8px; border-bottom-right-radius: 8px;"></div>
    </div>
    <p style="margin-top: 8px; font-weight: bold;">Similarity: {similarity}%</p>
    """
    st.markdown(bar_html, unsafe_allow_html=True)

# --- STREAMLIT UI ---
st.title("🤖 Chatbot Response Comparator")
question = st.text_input("Enter your question")

if st.button("Generate Response") and question.strip():
    with st.spinner("Calling Gemini..."):
        gemini = call_gemini(question)
    with st.spinner("Calling Cypher..."):
        cypher = call_cypher(question)
    with st.spinner("Calling DeepSeek..."):
        deepseek = call_deepseek(question)
    with st.spinner("Generating Final Answer..."):
        final = final_response_gemini_compare(question, gemini, cypher, deepseek)

    st.subheader("🔹 Gemini Response")
    st.write(gemini)

    st.subheader("🔹 Cypher Response")
    st.write(cypher)

    st.subheader("🔹 DeepSeek Response")
    st.write(deepseek)

    st.subheader("⭐ Final Best Answer")
    st.success(final)

    # Calculate and display similarity
    similarity = calculate_similarity(gemini, cypher, deepseek)
    similarity_bar(similarity)
