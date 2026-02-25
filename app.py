import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import google.generativeai as genai
from pinecone import Pinecone

# ==========================================
# LOAD ENV
# ==========================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# ==========================================
# INIT SERVICES
# ==========================================

genai.configure(api_key=GOOGLE_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

embedding_model = "models/embedding-001"
chat_model = genai.GenerativeModel("gemini-1.5-flash")

app = Flask(__name__)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_embedding(text):
    """Generate embedding for text using Gemini"""
    response = genai.embed_content(
        model=embedding_model,
        content=text
    )
    return response["embedding"]

def search_pinecone(query, top_k=3):
    """Search Pinecone with embedded query"""
    query_vector = get_embedding(query)

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    contexts = []
    for match in results["matches"]:
        if "metadata" in match and "text" in match["metadata"]:
            contexts.append(match["metadata"]["text"])

    return "\n".join(contexts)

def generate_response(question):
    """Generate final response using RAG"""
    
    context = search_pinecone(question)

    prompt = f"""
You are a professional medical information assistant.

Use the provided medical context to answer the question.
Be accurate, empathetic, and clear.
Always add a disclaimer advising users to consult a healthcare professional.

Medical Context:
{context}

User Question:
{question}

Answer:
"""

    response = chat_model.generate_content(prompt)
    return response.text

# ==========================================
# ROUTES
# ==========================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Medical Chatbot API Running",
        "endpoints": {
            "/chat": "POST - Send message",
            "/health": "GET - Server status"
        }
    })

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        answer = generate_response(user_message)
        return jsonify({
            "user_message": user_message,
            "bot_response": answer,
            "status": "success"
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print("\n🏥 Medical RAG Chatbot Running on http://localhost:5000\n")
    app.run(debug=True)