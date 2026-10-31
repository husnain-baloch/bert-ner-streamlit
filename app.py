import streamlit as st
from transformers import pipeline

st.title("BERT Named Entity Recognition")
st.write("Encoder-only model: finds people, places and organizations in text.")

@st.cache_resource
def load_model():
    return pipeline(
        "ner",
        model="sartajbhuvaji/bert-named-entity-recognition",
        aggregation_strategy="simple",
    )

ner = load_model()

text = st.text_area("Enter a sentence", "Elon Musk visited Google in California.")

if st.button("Run BERT"):
    results = ner(text)
    if not results:
        st.info("No entities found.")
    else:
        rows = [
            {"Word": r["word"], "Label": r["entity_group"], "Score": round(float(r["score"]), 2)}
            for r in results
        ]
        st.table(rows)
