import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import google.generativeai as genai
from pinecone import Pinecone
from flask import render_template

# Import your helpers and prompt
from helper import download_embeddings
from prompt import system_prompt  # Your template

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") or "medi-chatbot"

genai.configure(api_key=GOOGLE_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Use HF embeddings to match your index
embeddings = download_embeddings()

chat_model = genai.GenerativeModel("gemini-2.5-flash")   # ← start with this

app = Flask(__name__)

def get_embedding(text):
    """Generate embedding using HF (matches index)"""
    return embeddings.embed_query(text)

def search_pinecone(query, top_k=3):
    """Search Pinecone with embedded query"""
    query_vector = get_embedding(query)
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    contexts = [match["metadata"].get("text", "") for match in results["matches"]]
    return "\n".join(contexts)

def generate_response(question):
    """Generate final response using RAG"""
    try:
        context = search_pinecone(question)
        # Format prompt from prompt.py
        formatted_prompt = system_prompt.format(context=context) + f"\nUser Question: {question}\nAnswer:"
        response = chat_model.generate_content(formatted_prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"  # For debugging


@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    try:
        answer = generate_response(user_message)
        return jsonify({"bot_response": answer})
    except Exception as e:
        print(f"Error: {e}")  # Log to console
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    print("\n🏥 Medical RAG Chatbot Running on http://localhost:5000\n")
    app.run(debug=True)