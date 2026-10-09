import json
import re
import sqlite3
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util

# 1. Download required NLTK helper files
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# 2. Setup SQLite Database for logging interactions
DB_FILE = "chatbot_logs.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT,
            bot_response TEXT,
            confidence REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_chat(user_msg, bot_msg, score):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_logs (user_message, bot_response, confidence, timestamp) VALUES (?, ?, ?, ?)",
       (user_msg, bot_msg, score, datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    conn.close()

# 3. Load FAQs and Map Multiple Questions to Answers
with open("faqs.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)

faq_questions = []
faq_answers = []

for item in faq_data:
    for q in item["questions"]:
        faq_questions.append(q)
        faq_answers.append(item["answer"])

# 4. Use the Higher-Accuracy Transformer Model
print(">>> Loading high-accuracy AI Model (all-mpnet-base-v2)...")
model = SentenceTransformer('all-mpnet-base-v2')
faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)

def clean_input(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def find_answer(user_query: str):
    cleaned_query = clean_input(user_query)
    query_vec = model.encode(cleaned_query, convert_to_tensor=True)
    similarity_scores = util.cos_sim(query_vec, faq_embeddings)[0]
    
    best_index = int(similarity_scores.argmax())
    confidence = float(similarity_scores[best_index])
    
    # 0.55 confidence threshold avoids guessing bad answers
   # Change 0.55 to 0.40
    if confidence >= 0.40:
        return faq_answers[best_index], confidence
    else:
        return "I'm sorry, I'm not sure about that. Would you like me to connect you with human support?", confidence
# 5. FastAPI App
app = FastAPI()
init_db()

class Query(BaseModel):
    message: str

@app.post("/chat")
def chat_api(data: Query):
    reply, score = find_answer(data.message)
    log_chat(data.message, reply, score)
    return {"reply": reply, "confidence": round(score, 2)}

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
      <title>AI Support Chatbot</title>
      <style>
        body { font-family: Arial, sans-serif; background: #eef2f5; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
        .chat-box { width: 380px; height: 500px; background: white; border-radius: 12px; display: flex; flex-direction: column; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        .header { background: #2563eb; color: white; padding: 16px; border-radius: 12px 12px 0 0; font-weight: bold; text-align: center; }
        .messages { flex: 1; padding: 15px; overflow-y: auto; display: flex; flex-direction: column; gap: 10px; }
        .msg { padding: 8px 12px; border-radius: 15px; max-width: 75%; font-size: 14px; }
        .user { align-self: flex-end; background: #2563eb; color: white; }
        .bot { align-self: flex-start; background: #e5e7eb; color: #111; }
        .input-row { display: flex; padding: 10px; border-top: 1px solid #ddd; }
        input { flex: 1; padding: 10px; border: 1px solid #ccc; border-radius: 8px; outline: none; }
        button { margin-left: 8px; padding: 10px 14px; background: #2563eb; color: white; border: none; border-radius: 8px; cursor: pointer; }
      </style>
    </head>
    <body>
      <div class="chat-box">
       <div class="header">AI Support Assistant</div>
        <div class="messages" id="msgs">
          <div class="msg bot">Hello! Ask me about returns, shipping, accounts, or live support.</div>
        </div>
        <div class="input-row">
          <input id="inputBox" placeholder="Type a message..." onkeydown="if(event.key==='Enter') sendMsg()" />
          <button onclick="sendMsg()">Send</button>
        </div>
      </div>
      <script>
        async function sendMsg() {
          const input = document.getElementById('inputBox');
          const text = input.value.trim();
          if (!text) return;
          
          const msgs = document.getElementById('msgs');
          msgs.innerHTML += `<div class="msg user">${text}</div>`;
          input.value = '';
          
          const res = await fetch('/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message: text})
          });
          const data = await res.json();
          msgs.innerHTML += `<div class="msg bot">${data.reply} <small style='opacity:0.6;'>(${Math.round(data.confidence * 100)}% match)</small></div>`;
          msgs.scrollTop = msgs.scrollHeight;
        }
      </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    print(">>> Starting Chatbot Server at http://127.0.0.1:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)

