# extract_text.py

import pdfplumber
import docx2txt

def extract_text_from_file(uploaded_file):
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".txt"):
        return uploaded_file.read().decode("latin-1")

    elif file_name.endswith(".pdf"):
        text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text

    elif file_name.endswith(".docx"):
        return docx2txt.process(uploaded_file)

    else:
        return "Unsupported file type."
