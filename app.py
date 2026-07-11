import streamlit as st
from transformers import pipeline
from pypdf import PdfReader
import re

# Professional page configuration
st.set_page_config(
    page_title="DeepInsight: Question Answering System", 
    page_icon="🧠", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Advanced CSS Injection for Premium UX/UI Customization
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 750px; }
    .main-title { font-family: 'Inter', -apple-system, sans-serif; font-weight: 800; letter-spacing: -0.5px; margin-bottom: 5px; }
    .answer-box { background-color: rgba(46, 204, 113, 0.15); border-left: 5px solid #2ecc71; padding: 20px; border-radius: 8px; margin-top: 15px; margin-bottom: 15px; transition: all 0.3s ease; }
    .answer-box:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(46, 204, 113, 0.2); }
    .answer-header { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; color: #2ecc71; margin-bottom: 8px; }
    .answer-body { font-size: 1.2rem; font-weight: 500; line-height: 1.6; word-wrap: break-word; }
    .stButton>button { background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important; color: white !important; border: none !important; padding: 12px 24px !important; font-weight: 600 !important; border-radius: 8px !important; width: 100% !important; transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important; box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.2) !important; }
    .stButton>button:hover { transform: translateY(-2px) !important; box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.4) !important; background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 45px; white-space: pre-wrap; background-color: transparent; border-radius: 6px; padding: 6px 16px; font-weight: 500; }
    </style>
""", unsafe_allow_html=True)

# Clean Intuitive Header
st.markdown('<h1 class="main-title">🧠 DeepInsight: Question Answering System</h1>', unsafe_allow_html=True)
st.markdown("---")

@st.cache_resource
def load_qa_engine():
    return pipeline("question-answering", model="abhiMahi/DeepInsight-QA")

with st.spinner("Initializing DeepInsight Core Engine..."):
    qa_pipe = load_qa_engine()

# Smart Filter to handle 400+ page documents without crashing the free server
def get_top_relevant_chunks(question, text, chunk_size=2000, top_n=3):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    q_tokens = set(re.findall(r'\w+', question.lower()))
    
    scored_chunks = []
    for chunk in chunks:
        c_tokens = set(re.findall(r'\w+', chunk.lower()))
        score = len(q_tokens.intersection(c_tokens))
        scored_chunks.append((score, chunk))
        
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [x[1] for x in scored_chunks[:top_n]]
    return "\n".join(top_chunks)

# Tabbed Interface
tab1, tab2 = st.tabs(["📁 Upload Document(s)", "✍️ Raw Text Input"])
global_context_text = ""

with tab1:
    # Enabled multiple file uploads
    uploaded_files = st.file_uploader("Upload any document or pdf (Multiple files allowed):", type=["pdf", "txt"], accept_multiple_files=True)
    
    if uploaded_files:
        extracted_texts = []
        for file in uploaded_files:
            file_extension = file.name.split(".")[-1].lower()
            if file_extension == "pdf":
                try:
                    pdf_reader = PdfReader(file)
                    text_list = [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
                    extracted_texts.append("\n".join(text_list))
                except Exception as e:
                    st.error(f"Error reading {file.name}: {e}")
            elif file_extension == "txt":
                extracted_texts.append(file.read().decode("utf-8"))
        
        global_context_text = "\n".join(extracted_texts)
        if global_context_text:
            st.success(f"System: Successfully processed {len(uploaded_files)} file(s).")

with tab2:
    context_text_input = st.text_area(
        label="Add text from which you want to get an answer:",
        value="",
        placeholder="Type or paste paragraphs here...",
        height=260
    )
    if context_text_input:
        global_context_text = context_text_input

# Clean Query Section
st.markdown("### ❓ Ask a Question")
user_question = st.text_input("Enter your question based on the provided data:")

if st.button("Extract Target Answer"):
    if not global_context_text.strip():
        st.warning("⚠️ Action Blocked: Please upload a file or write a paragraph first.")
    elif not user_question.strip():
        st.warning("⚠️ Action Blocked: Please type a descriptive question.")
    else:
        with st.spinner("Scanning documents and processing semantics..."):
            try:
                # Instantly filter down massive documents to the best 3 pages
                optimized_context = get_top_relevant_chunks(user_question, global_context_text)
                
                # Run the model on the optimized text
                result = qa_pipe(question=user_question, context=optimized_context)
                
                st.markdown(f"""
                    <div class="answer-box">
                        <div class="answer-header">✨ Analysis Complete</div>
                        <div class="answer-body">{result['answer']}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.caption(f"**Confidence Matrix:** {result['score']:.2%}")
                with col2:
                    st.progress(float(result['score']))
                    
            except Exception as e:
                st.error(f"Runtime Pipeline Error: {e}")
