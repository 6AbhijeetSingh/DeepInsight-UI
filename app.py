import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="DeepInsight QA", page_icon="🧠", layout="centered")

st.title("🤖 DeepInsight: AI Text Analysis")
st.markdown("#### Powered by Fine-Tuned RoBERTa (91.00% Accuracy)")
st.write("Paste any document below. The AI will read it and scientifically extract the exact answer to your question.")
st.markdown("---")

@st.cache_resource
def load_qa_engine():
    # This automatically streams your 500MB model from your free Hugging Face storage!
    return pipeline("question-answering", model="abhiMahi/DeepInsight-QA")

with st.spinner("Initializing DeepInsight Core..."):
    qa_pipe = load_qa_engine()

context_text = st.text_area("📄 Source Document / Context:", height=250)
user_question = st.text_input("❓ Question:")

if st.button("Extract Answer", type="primary", use_container_width=True):
    if context_text and user_question:
        with st.spinner("Processing semantics..."):
            result = qa_pipe(question=user_question, context=context_text)
            
            st.success("Analysis Complete!")
            st.metric(label="Target Answer", value=result['answer'])
            st.caption(f"Confidence Score: {result['score']:.2%}")
            st.progress(float(result['score']))
    else:
        st.warning("⚠️ Please provide both a source document and a question.")
