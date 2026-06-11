from textblob import TextBlob
import os
from pypdf import PdfReader
from flask import Flask, render_template, request, jsonify, redirect
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)

# =========================
# LOAD FAQ DATA
# =========================
try:
    data = pd.read_csv("faq.csv")
    questions = data["question"].fillna("").astype(str).str.lower().str.strip()
    answers = data["answer"].fillna("").astype(str)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(questions)
except Exception as e:
    print(f"Warning: Could not load faq.csv. Error: {e}")
    questions, answers, vectorizer, X = None, None, None, None

# GLOBAL ANALYTICS VARIABLES
positive_count = 0
neutral_count = 0
negative_count = 0
total_messages = 0
uploaded_file = ""
pdf_text = ""

pdf_chunks = []
pdf_vectorizer = None
pdf_vectors = None

def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.2:
        return "😊 Positive"
    elif polarity < -0.2:
        return "😠 Negative"
    else:
        return "😐 Neutral"

# ==========================================
# ROBUST MULTI-SENTENCE RAG EXTRACTION LAYER
# ==========================================
def extract_clean_answer(query, context):
    """
    Surgically extracts the sentence containing key metrics plus 
    the contextually relevant succeeding sentence.
    """
    # Clean whitespace strings
    context_clean = " ".join(context.split())
    
    # Split text blocks by punctuation boundaries safely
    sentences = re.split(r'(?<=[.!?])\s+', context_clean)
    
    # Target Index Identification
    target_idx = -1
    
    # Intent 1: Business Hours
    if "hour" in query or "time" in query or "open" in query:
        for idx, sentence in enumerate(sentences):
            if any(w in sentence.lower() for w in ["hour", "operation", "open", "schedule", "monday", "am", "pm"]):
                target_idx = idx
                break
                
    # Intent 2: Order Cancellations
    elif "cancel" in query or "change order" in query or "stop order" in query:
        for idx, sentence in enumerate(sentences):
            if any(w in sentence.lower() for w in ["cancel", "cancellation", "void", "modify", "before dispatch"]):
                target_idx = idx
                break

    # Intent 3: Return Window / Refunds
    elif "return" in query or "refund" in query or "policy" in query:
        for idx, sentence in enumerate(sentences):
            if any(w in sentence.lower() for w in ["return", "refund", "30 days", "unopened", "window"]):
                target_idx = idx
                break

    # If an intent-matched sentence is located, group it with the NEXT sentence for complete info
    if target_idx != -1:
        extracted_group = sentences[target_idx : target_idx + 2]
        return " ".join(extracted_group)

    # Secondary dynamic fallback strategy: grab up to 2 sentences
    return " ".join(sentences[:2])

# =========================
# CHATBOT RETRIEVAL LOGIC
# =========================
def chatbot(query):
    global pdf_chunks, pdf_vectorizer, pdf_vectors

    original_query = query.lower().strip()
    query = original_query.replace("?", "").replace(".", "").replace("!", "")

    # Conversational Quick Handlers
    if query in ["thank you", "thanks", "thankyou"]:
        return "You're welcome! 😊"
    if query in ["hi", "hello", "hey"]:
        return "Hello! How can I help you today? 😊"
    if query in ["bye", "goodbye"]:
        return "Goodbye! Have a great day! 👋"    

    # 1. RAG Pipeline Stage: Search Uploaded PDF Document
    if pdf_chunks and pdf_vectorizer is not None:
        query_vector = pdf_vectorizer.transform([query])
        similarity = cosine_similarity(query_vector, pdf_vectors)[0]
        
        best_index = similarity.argmax()
        best_score = similarity.max()
        
        print("PDF Retrieval Score:", best_score)

        if best_score > 0.05:
            raw_context = pdf_chunks[best_index]
            print(f"Retrieved Context: {raw_context}")
            
            # Extract matching context plus subsequent sequence context sentences
            return extract_clean_answer(original_query, raw_context)

    # 2. Pipeline Stage: Fallback to CSV Base FAQ Database
    if vectorizer is not None and X is not None:
        query_vector = vectorizer.transform([query])
        similarity = cosine_similarity(query_vector, X)[0]

        score = similarity.max()
        index = similarity.argmax()

        if score >= 0.05:
            return answers.iloc[index]

    return "Sorry, I don't understand that question. Let me know if you would like me to transfer you to a human agent."

# =========================
# ROUTING CONTROLLERS
# =========================

@app.route("/")
def home():
    return render_template("index.html", uploaded_file=uploaded_file)


@app.route("/upload", methods=["POST"])
def upload():
    global uploaded_file, pdf_text, pdf_chunks, pdf_vectorizer, pdf_vectors

    if "pdf" not in request.files:
        return redirect("/")
        
    file = request.files["pdf"]

    if file.filename != "":
        os.makedirs("uploads", exist_ok=True)
        filepath = os.path.join("uploads", file.filename)
        file.save(filepath)
        uploaded_file = file.filename

        reader = PdfReader(filepath)
        pdf_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text = text.replace("–", "-").replace("—", "-")
                pdf_text += text + "\n"

        # Group text into context block sizes (8 lines each)
        lines = [line.strip() for line in pdf_text.split("\n") if line.strip()]
        pdf_chunks = []
        for i in range(0, len(lines), 8):
            combined_chunk = " ".join(lines[i:i+8])
            pdf_chunks.append(combined_chunk)

        if pdf_chunks:
            pdf_vectorizer = TfidfVectorizer()
            pdf_vectors = pdf_vectorizer.fit_transform(pdf_chunks)

    return redirect("/")


@app.route("/get", methods=["GET", "POST"])
def get_response():
    global positive_count, neutral_count, negative_count, total_messages
    
    if request.method == "POST":
        user_message = request.form.get("message", "").strip()
    else:
        user_message = request.args.get("message", "").strip()

    if not user_message:
        return jsonify({"response": "I didn't catch that.", "sentiment": "😐 Neutral"})

    response = chatbot(user_message)
    sentiment = get_sentiment(user_message)
    
    total_messages += 1

    if "Positive" in sentiment:
        positive_count += 1
    elif "Negative" in sentiment:
        negative_count += 1
    else:
        neutral_count += 1

    try:
        with open("chat_history.txt", "a", encoding="utf-8") as file:
            file.write(f"User: {user_message}\nSentiment: {sentiment}\nBot: {response}\n{'-'*50}\n")
    except Exception as e:
        print(f"Logging Error: {e}")

    return jsonify({"response": response, "sentiment": sentiment})


@app.route("/stats")
def stats():
    return jsonify({
        "total": total_messages,
        "positive": positive_count,
        "neutral": neutral_count,
        "negative": negative_count
    })

if __name__ == "__main__":
    app.run(debug=True)
