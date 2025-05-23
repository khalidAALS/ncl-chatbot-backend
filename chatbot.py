import os
import json
import faiss
import numpy as np
import openai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# loads FAISS  for index and metadata
index = faiss.read_index("college_index.faiss")
with open("metadata.json", "r") as f:
    metadata = json.load(f)
texts = [entry["text"] for entry in metadata]

# loads the structured knowledge
with open("structured_knowledge.json", "r") as f:
    structured_knowledge = json.load(f)

# embeds user querys
def embed_query(query):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    return np.array(response.data[0].embedding).astype("float32")

# searchs vector
def search_index(query, top_k=3):
    embedded_query = embed_query(query)
    scores, indices = index.search(np.array([embedded_query]), top_k)
    return [texts[i] for i in indices[0]]

# structure of Q&A logic
def check_structured_query(user_input):
    text = user_input.lower().strip()

    # common bot greetings
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon"]
    if text in greetings:
        return "Hello! How can I assist you with your Newcastle College queries today?"

    # fallback for generic course name
    if "software engineering" in text and "fdsc" not in text:
        return "Are you asking about the FdSc Software Engineering course? Try asking 'FdSc Software Engineering' to get full details."

    # FdSc course details
    for key, course in structured_knowledge.get("course_details", {}).items():
        title = course["title"].lower()
        if title in text:
            return (
                f"{course['title']}:\n{course['overview']}\n\n"
                f"📍 Location: {course['location']}\n"
                f"🎓 Progression: {course['progression']}\n"
                f"📘 Duration: {course['duration']} | Mode: {course['mode']}"
            )

    # contact details
    if "contact" in text or "phone" in text or "email" in text:
        contact = structured_knowledge["contact_info"]
        return f"You can contact Newcastle College at {contact['phone']} or email them at {contact['email']}."

    # term Dates
    if "term dates" in text or "term start" in text or "term end" in text:
        terms = structured_knowledge["term_dates"]
        return (
            f"Term Dates:\n"
            f"• Autumn: {terms['autumn']}\n"
            f"• Spring: {terms['spring']}\n"
            f"• Summer: {terms['summer']}"
        )

    # bursary (adult learners)
    if "bursary" in text or ("support" in text and "19" in text):
        bursaries = structured_knowledge["student_types"]["adult_19_plus"]["bursaries"]
        return f"Bursaries available for adult learners include: {', '.join(bursaries)}."

    # apprenticeships
    if "apprenticeship" in text and "course" in text:
        routes = structured_knowledge["student_types"]["apprenticeships"]["routes"]
        return f"Available apprenticeships include: {', '.join(routes)}."

    # level3
    if "level 3" in text:
        courses = structured_knowledge["courses"]["level_3"]
        return f"Level 3 courses include: {', '.join(courses)}."

    # FdSc
    if "fdsc" in text:
        courses = structured_knowledge["courses"]["fdsc"]
        return f"FdSc courses include: {', '.join(courses)}."

    # HTQ
    if "htq" in text or "higher technical qualification" in text:
        courses = structured_knowledge["courses"]["htq"]
        return f"HTQ courses include: {', '.join(courses)}."

    # aaccess to HE
    if "access to he" in text or "access course" in text:
        courses = structured_knowledge["courses"]["access_to_he"]
        return f"Access to Higher Education courses include: {', '.join(courses)}."

    return None

# gpt4 fallback
def ask_gpt4(question, context):
    system_msg = (
        "You are a friendly and helpful assistant for Newcastle College. "
        "Use the context provided to answer questions clearly. "
        "If the answer is not available, respond politely and suggest a helpful next step, "
        "but never say that the data is unavailable — always guide the user forward."
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

# flask app
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
    return (
        "<h2>Newcastle College ChatBot API is Live</h2>"
        "<p>To use the chatbot, make a POST request to <code>/chat</code>.</p>"
        "<p>This backend is used by the frontend chatbot hosted on GitHub Pages.</p>",
        200,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=10000)


