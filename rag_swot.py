# rag_swot.py

import numpy as np
import subprocess
import nltk
import json
from sentence_transformers import SentenceTransformer, util
from embed_store import model  # reuse the same model used for vectorstore

nltk.download('punkt')

# Fixed SWOT questions
SWOT_QUESTIONS = {
    "Strengths": "You are a senior business analyst. Identify the company's internal strengths based on the transcript context. If strengths are not explicitly stated, you may infer them from clearly supportive evidence (but mark them as 'inferred'). Return 2–5 clear bullet points.",
    "Weaknesses": "You are a senior business analyst. Identify the company's internal weaknesses. Return only what is clearly stated. Do NOT infer. Return 2–5 points.",
    "Opportunities": "You are a senior business analyst. Identify external opportunities for the company based on the transcript context. These may be explicit or inferred from market behavior, financials, or strategic goals. Return 2–5 points.",
    "Threats": "You are a senior business analyst. Identify explicit or strongly implied threats to the business. These may include regulation, market risk, tech shifts, or competition. Return 2–5 clear points."
}

# Sentence matcher
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

def find_support_excerpt(bullet, sentences, window=1):
    bullet_emb = sentence_model.encode(bullet, convert_to_tensor=True)
    sent_embs = sentence_model.encode(sentences, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(bullet_emb, sent_embs)
    best_idx = int(cosine_scores.cpu().argmax())

    start = max(0, best_idx - window)
    end = min(len(sentences), best_idx + window + 1)
    return " ".join(sentences[start:end])

# FAISS retrieval
def retrieve_top_k(question, vectorstore, k=5):
    question_embedding = model.encode([question])
    _, indices = vectorstore["index"].search(np.array(question_embedding), k)
    return [vectorstore["chunks"][i] for i in indices[0]]

# Run local LLM via Ollama
def query_ollama(prompt, model_name="llama3"):
    command = ["ollama", "run", model_name]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output, _ = process.communicate(input=prompt)
    return output.strip()

# Main SWOT generator
def generate_swot(vectorstore):
    swot_output = {}
    full_text = " ".join(vectorstore["chunks"])
    transcript_sentences = nltk.sent_tokenize(full_text)

    for category, question in SWOT_QUESTIONS.items():
        top_chunks = retrieve_top_k(question, vectorstore)
        context = "\n\n".join(top_chunks)

        prompt = f"""You are a senior competitive intelligence analyst tasked with generating a SWOT analysis from an earnings call transcript.

Below is a set of excerpts from the transcript. Your job is to create a structured SWOT for the category: {category}.

✅ Use bullet point format.
✅ You may infer points if clearly supported by the context, but mark those with (inferred).
✅ Include 2–5 clear, specific, business-relevant points.

### Context:
{context}

---

Now list the {category}:
"""

        response = query_ollama(prompt)
        raw_bullets = [line.strip("-• ").strip() for line in response.split("\n") if line.strip()]
        raw_bullets = [b for b in raw_bullets if len(b.strip()) > 5]

        clean_bullets = []
        for b in raw_bullets:
            if not any(x in b.lower() for x in ["based on the context", "as a senior", "here are"]):
                clean_bullets.append(b)

        swot_output[category] = []
        for bullet in clean_bullets:
            excerpt = find_support_excerpt(bullet, transcript_sentences)
            swot_output[category].append({
                "point": bullet,
                "support": excerpt
            })

    return swot_output
