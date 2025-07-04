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

def call_cypher(prompt):  # renamed from call_moonshot
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://chatbot.com",  # Optional
            "X-Title": "ChatbotCompare"             # Optional
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
- Read all three responses.
- Find the statements or meanings that are **semantically similar** in at least **two out of three** answers.
- If wording is different but meaning is the same, consider them similar.
- Based on those overlapping meanings, generate a **single final answer**.
- Underline or highlight the parts in the final answer that are common (or equivalent) between at least two answers.
  - Use **markdown bold** (`**like this**`) to highlight the common parts.
- Estimate a **similarity level percentage** (0–100%) representing how much agreement exists among the models and highlight the .

Return format:
Final Answer: <the best merged answer>
Similarity Level: <percentage>%
"""

        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Final Answer Error] {e}"

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
    
# Streamlit UI
st.title(" Chatbot Response Comparator")
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

    st.subheader("Gemini Response")
    st.write(gemini)

    st.subheader("Cypher Response")
    st.write(cypher)

    st.subheader("DeepSeek Response")
    st.write(deepseek)

    st.subheader("⭐ Final Best Answer")
    st.success(final)

    similarity_match = re.search(r"Similarity Level: (\d+)%", final)
    similarity = int(similarity_match.group(1)) if similarity_match else 0
    similarity_bar(similarity)
