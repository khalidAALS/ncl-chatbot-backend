import os
import json
import faiss
import numpy as np
import openai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load FAISS index and metadata
index = faiss.read_index("college_index.faiss")
with open("metadata.json", "r") as f:
    metadata = json.load(f)
texts = [entry["text"] for entry in metadata]

# Load structured knowledge
with open("structured_knowledge.json", "r") as f:
    structured_knowledge = json.load(f)

# Embed user query
def embed_query(query):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    return np.array(response.data[0].embedding).astype("float32")

# Vector search
def search_index(query, top_k=3):
    embedded_query = embed_query(query)
    scores, indices = index.search(np.array([embedded_query]), top_k)
    return [texts[i] for i in indices[0]]

# Structured Q&A logic
def check_structured_query(user_input):
    text = user_input.lower()
    
    # Individual course detail check
    for key, course in structured_knowledge.get("course_details", {}).items():
        title = course["title"].lower()
        if title in text:
            return (
                f"{course['title']}:\n{course['overview']}\n\n"
                f"📍 Location: {course['location']}\n"
                f"🎓 Progression: {course['progression']}\n"
                f"📘 Duration: {course['duration']} | Mode: {course['mode']}"
            )

    # Contact
    if "contact" in text or "phone" in text or "email" in text:
        contact = structured_knowledge["contact_info"]
        return f"You can contact Newcastle College at {contact['phone']} or email them at {contact['email']}."

    # Term Dates
    if "term dates" in text or "term start" in text or "term end" in text:
        terms = structured_knowledge["term_dates"]
        return (
            f"Term Dates:\n"
            f"• Autumn: {terms['autumn']}\n"
            f"• Spring: {terms['spring']}\n"
            f"• Summer: {terms['summer']}"
        )

    # Bursaries (Adult learners)
    if "bursary" in text or ("support" in text and "19" in text):
        bursaries = structured_knowledge["student_types"]["adult_19_plus"]["bursaries"]
        return f"Bursaries available for adult learners include: {', '.join(bursaries)}."

    # Apprenticeships
    if "apprenticeship" in text and "course" in text:
        routes = structured_knowledge["student_types"]["apprenticeships"]["routes"]
        return f"Available apprenticeships include: {', '.join(routes)}."

    # Level 3 courses
    if "level 3" in text:
        courses = structured_knowledge["courses"]["level_3"]
        return f"Level 3 courses include: {', '.join(courses)}."

    # FdSc courses
    if "fdsc" in text:
        courses = structured_knowledge["courses"]["fdsc"]
        return f"FdSc courses include: {', '.join(courses)}."

    # HTQs
    if "htq" in text or "higher technical qualification" in text:
        courses = structured_knowledge["courses"]["htq"]
        return f"HTQ courses include: {', '.join(courses)}."

    # Access to HE
    if "access to he" in text or "access course" in text:
        courses = structured_knowledge["courses"]["access_to_he"]
        return f"Access to Higher Education courses include: {', '.join(courses)}."


    return None


# Ask GPT-4
def ask_gpt4(question, context):
    system_msg = (
        "You are a friendly and helpful assistant for Newcastle College. "
        "Use the context provided to answer questions clearly. "
        "If the answer is not available, respond politely and suggest a helpful next step, "
        "but never mention that the data is missing or unavailable — always guide the user forward."
    )
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# Flask app
app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    structured_answer = check_structured_query(user_input)
    if structured_answer:
        answer = structured_answer
    else:
        context_chunks = search_index(user_input)
        combined_context = "\n\n".join(context_chunks)
        answer = ask_gpt4(user_input, combined_context)

    return jsonify({"response": answer})

@app.route("/", methods=["GET"])
def home():
    return render_template("chatbot.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)

