# ner_financials.py

import spacy
import re

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Keywords to catch financial statements
FINANCIAL_KEYWORDS = [
    "revenue", "profit", "earnings", "margin", "growth", "guidance",
    "net income", "loss", "operating income", "EPS", "EBITDA"
]

def extract_entities_and_financials(text):
    doc = nlp(text)

    # Extract named entities
    people = list(set(ent.text for ent in doc.ents if ent.label_ == "PERSON"))
    orgs = list(set(ent.text for ent in doc.ents if ent.label_ == "ORG"))

    # Extract financial mentions (loose regex + keyword filtering)
    lines = text.split("\n")
    financials = []
    for line in lines:
        if any(keyword in line.lower() for keyword in FINANCIAL_KEYWORDS):
            if re.search(r"\$?[0-9,.]+", line):
                financials.append(line.strip())

    return {
        "People Mentioned": people,
        "Organizations Mentioned": orgs
    }, financials
