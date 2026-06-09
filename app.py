from textblob import TextBlob
import os
from pypdf import PdfReader
from flask import Flask, render_template, request, jsonify, redirect
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# =========================
# LOAD FAQ DATA
# =========================

data = pd.read_csv("faq.csv")

questions = data["question"]
answers = data["answer"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# =========================
# GLOBAL VARIABLES
# =========================
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

    print("Text:", text)
    print("Polarity:", polarity)

    if polarity > 0.2:
        return "😊 Positive"

    elif polarity < -0.2:
        return "😠 Negative"

    else:
        return "😐 Neutral"
# =========================
# CHATBOT
# =========================

def chatbot(query):

    global pdf_chunks
    global pdf_vectorizer
    global pdf_vectors

    query = query.lower().strip()

 
    if query in ["thank you", "thanks", "thankyou"]:
       return "You're welcome! 😊"

    if query in ["hi", "hello", "hey"]:
       return "Hello! How can I help you today? 😊"

    if query in ["bye", "goodbye"]:
       return "Goodbye! Have a great day! 👋"    
    
    # rest of your code...
    print("User Query:", query)
    print("PDF Chunks Count:", len(pdf_chunks))
    


    # ---------------------
    # Search PDF First
    # ---------------------

    if pdf_chunks and pdf_vectorizer is not None:

        query_vector = pdf_vectorizer.transform([query])

        similarity = cosine_similarity(
            query_vector,
            pdf_vectors
        )[0]

        print("Similarity:", similarity)

        best_index = similarity.argmax()

        best_score = similarity.max()

        print("Best Score:", best_score)

        # Return best matching chunk
        if best_score > 0.15:

            return pdf_chunks[best_index]

    # ---------------------
    # Search FAQ
    # ---------------------

    query_vector = vectorizer.transform([query])

    similarity = cosine_similarity(
        query_vector,
        X
    )

    score = similarity.max()

    index = similarity.argmax()

    print("FAQ Score:", score)

    if score < 0.1:
        return "Sorry, I don't understand that question."

    return answers.iloc[index]

# =========================
# HOME PAGE
# =========================

@app.route("/")
def home():

    return render_template(
        "index.html",
        uploaded_file=uploaded_file
    )


# =========================
# PDF UPLOAD
# =========================

@app.route("/upload", methods=["POST"])
def upload():

    global uploaded_file
    global pdf_text
    global pdf_chunks
    global pdf_vectorizer
    global pdf_vectors

    print("UPLOAD ROUTE CALLED")

    file = request.files["pdf"]

    if file.filename != "":

        os.makedirs("uploads", exist_ok=True)

        filepath = os.path.join(
            "uploads",
            file.filename
        )

        file.save(filepath)

        uploaded_file = file.filename

        # ---------------------
        # Read PDF
        # ---------------------

        reader = PdfReader(filepath)

        pdf_text = ""

        for page in reader.pages:

            text = page.extract_text()

            if text:
                pdf_text += text + "\n"

        print("PDF Loaded Successfully")
        print(pdf_text[:500])

        # ---------------------
        # Create Chunks
        # ---------------------

       # ---------------------
# Create Chunks
# ---------------------

    pdf_chunks = [
    chunk.strip()
    for chunk in pdf_text.split("\n")
    if chunk.strip()
]

    print("PDF Loaded:", uploaded_file)
    print("Total Chunks:", len(pdf_chunks))
    print("Chunks Created:", len(pdf_chunks))
        # ---------------------
        # Create Vectors
        # ---------------------

    if pdf_chunks:

            pdf_vectorizer = TfidfVectorizer()

            pdf_vectors = pdf_vectorizer.fit_transform(
                pdf_chunks
            )

            print("PDF Vector Store Ready")

    return redirect("/")


# =========================
# CHAT RESPONSE
# =========================

@app.route("/get", methods=["POST"])
def get_response():

    user_message = request.form["message"]

    response = chatbot(user_message)
    sentiment = get_sentiment(user_message)
    global positive_count
    global neutral_count  
    global negative_count
    global total_messages

    total_messages += 1

    if "Positive" in sentiment:
        positive_count += 1

    elif "Negative" in sentiment:
       negative_count += 1

    else:
      neutral_count += 1
      with open("chat_history.txt", "a", encoding="utf-8") as file:

        file.write(f"User: {user_message}\n")
        file.write(f"Sentiment: {sentiment}\n")
        file.write(f"Bot: {response}\n")
        file.write("-" * 50 + "\n")

    return jsonify({
        "response": response,
        "sentiment": sentiment
    })
@app.route("/stats")
def stats():

    return jsonify({
        "total": total_messages,
        "positive": positive_count,
        "neutral": neutral_count,
        "negative": negative_count
    })


# =========================
# RUN APP
# =========================

if __name__ == "__main__":
    app.run(debug=True)