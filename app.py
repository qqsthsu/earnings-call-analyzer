import streamlit as st
from extract_text import extract_text_from_file
from ner_financials import extract_entities_and_financials
from embed_store import create_faiss_index
from rag_swot import generate_swot
from utils import generate_summary
from fpdf import FPDF
import io
import os
import json

# ---------------- TEXT CLEANING ------------------
def clean_text(text):
    replacements = {
        "–": "-", "—": "-", "“": '"', "”": '"', "’": "'", "‘": "'", "•": "-", "📌": "->"
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("latin-1", errors="ignore").decode("latin-1")

# ---------------- SENTENCE TRIM ------------------
def trim_to_sentence(text):
    end = text.find(".")
    if end != -1:
        return text[:end+1]
    return text

# ---------------- PDF GENERATION ------------------
def generate_swot_pdf(swot, company_name="SWOT Report"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, clean_text(f"{company_name} - SWOT Analysis"), ln=True)

    pdf.set_font("Arial", '', 12)
    for section, bullets in swot.items():
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, clean_text(section), ln=True)
        pdf.set_font("Arial", '', 12)
        for bullet in bullets:
            pdf.multi_cell(0, 8, clean_text(f"- {bullet['point']}"))
            pdf.set_text_color(100)
            pdf.multi_cell(0, 6, clean_text(f"   -> {trim_to_sentence(bullet['support'])}"))
            implication = bullet.get("implication")
            if implication:
                pdf.multi_cell(0, 6, clean_text(f"   📍 {trim_to_sentence(implication)}"))
            pdf.set_text_color(0)

    return pdf.output(dest='S').encode('latin-1')

# ---------------- STREAMLIT APP ------------------
st.set_page_config(page_title="Competitor Profile Generator", layout="wide")
st.title("\U0001F4CA Competitor Profile Generator")

uploaded_file = st.file_uploader("Upload an earnings call transcript", type=["pdf", "txt", "docx"])

output_choice = st.multiselect(
    "What would you like to generate?",
    ["SWOT Analysis", "Financial Performance", "Executive Summary", "Leadership Insights"],
    default=["SWOT Analysis"]
)

if uploaded_file:
    with st.spinner("\U0001F50D Extracting text..."):
        raw_text = extract_text_from_file(uploaded_file)

    with st.spinner("\U0001F9E0 Extracting people, organizations & financials..."):
        entities, financials = extract_entities_and_financials(raw_text)

    with st.spinner("\U0001F4E6 Creating embeddings & FAISS index..."):
        vectorstore, chunks = create_faiss_index(raw_text)

    swot = None
    if "SWOT Analysis" in output_choice:
        with st.spinner("\U0001F9E9 Generating SWOT using Ollama..."):
            swot = generate_swot(vectorstore)

        # Save SWOT locally
        company_name = uploaded_file.name.replace(" ", "_").replace(".", "_")
        os.makedirs("saved_swots", exist_ok=True)
        with open(f"saved_swots/{company_name}_swot.json", "w") as f:
            json.dump(swot, f, indent=2)

        # Display SWOT bullets
        st.subheader("\U0001F9E0 SWOT Analysis")
        for section, bullets in swot.items():
            st.markdown(f"### {section}")
            for bullet in bullets:
                st.markdown(f"**• {bullet['point']}**")
                if bullet.get("support"):
                    st.caption(f"📌 {bullet['support']}")
                    if bullet.get("implication"):
                        st.caption(f"📍 {bullet['implication']}")


        # Download as PDF
        pdf_bytes = generate_swot_pdf(swot, company_name=uploaded_file.name)
        st.download_button(
            label="\U0001F4C4 Download SWOT as PDF",
            data=pdf_bytes,
            file_name=f"{company_name}_swot.pdf",
            mime="application/pdf"
        )

    summary = None
    if "Executive Summary" in output_choice:
        if swot is not None:
            with st.spinner("📝 Generating executive summary..."):
                summary = generate_summary(swot, entities, financials)
        else:
            st.warning("⚠️ Please select 'SWOT Analysis' as well to generate an Executive Summary.")

    if "Executive Summary" in output_choice and summary:
        st.subheader("📝 Executive Summary")
        st.write(summary)

    if "Leadership Insights" in output_choice:
        st.subheader("👤 Key People & Organizations")
        st.write(entities)

    if "Financial Performance" in output_choice:
        st.subheader("💰 Financial Highlights")
        st.write(financials)
