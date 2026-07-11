import streamlit as st
from transformers import pipeline
from pypdf import PdfReader

# Professional page configuration
st.set_page_config(
    page_title="DeepInsight AI", 
    page_icon="🧠", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Advanced CSS Injection for Premium UX/UI Customization (Hover effects, shadows, transitions)
st.markdown("""
    <style>
    /* Global Container Adjustments for Responsiveness */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 750px;
    }
    
    /* Main Title Styling */
    .main-title {
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 5px;
    }
    
    /* Sleek Custom Answer Container (Fixes the text truncation/dot bug) */
    .answer-box {
        background-color: rgba(46, 204, 113, 0.15);
        border-left: 5px solid #2ecc71;
        padding: 20px;
        border-radius: 8px;
        margin-top: 15px;
        margin-bottom: 15px;
        transition: all 0.3s ease;
    }
    .answer-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(46, 204, 113, 0.2);
    }
    .answer-header {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
        color: #2ecc71;
        margin-bottom: 8px;
    }
    .answer-body {
        font-size: 1.2rem;
        font-weight: 500;
        line-height: 1.6;
        word-wrap: break-word;
    }

    /* Professional Premium Button Design with Hover Scale Transitions */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        width: 100% !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.2) !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.4) !important;
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
    }
    .stButton>button:active {
        transform: translateY(0px) !important;
    }
    
    /* Custom Styling for Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 6px;
        padding: 6px 16px;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# 1. Clean Header Architecture
st.markdown('<h1 class="main-title">🧠 DeepInsight: AI Text Analysis</h1>', unsafe_allow_html=True)
st.markdown("---")

# 2. Lazy Load Pipeline Core
@st.cache_resource
def load_qa_engine():
    return pipeline("question-answering", model="abhiMahi/DeepInsight-QA")

with st.spinner("Initializing DeepInsight Core Engine..."):
    qa_pipe = load_qa_engine()

# 3. Industry-Standard Tabbed Interface Layout (LeetCode / GFG Style)
tab1, tab2 = st.tabs(["📁 Upload Document", "✍️ Raw Text Input"])
context_text = ""

with tab1:
    uploaded_file = st.file_uploader("Upload any document or pdf:", type=["pdf", "txt"])
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension == "pdf":
            try:
                pdf_reader = PdfReader(uploaded_file)
                text_list = [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
                context_text = "\n".join(text_list)
                st.info(f"System: Safely parsed {len(pdf_reader.pages)} structural pages.")
            except Exception as e:
                st.error(f"Parser Alert: Unable to compile structural PDF contents. Check logs: {e}")
        elif file_extension == "txt":
            context_text = uploaded_file.read().decode("utf-8")
            st.info("System: Plain text stream read successfully.")

with tab2:
    context_text_input = st.text_area(
        label="Add text from which you want to get answer:",
        value=context_text if context_text else "",
        placeholder="Type or paste paragraphs here...",
        height=260
    )
    if context_text_input:
        context_text = context_text_input

# 4. Global Action & Inference Workspace
st.markdown("### ❓ Semantic Extraction Workspace")
user_question = st.text_input("Enter your question based on the data context source:")

if st.button("Extract Target Answer"):
    if not context_text.strip():
        st.warning("⚠️ Action Blocked: The data context buffer is completely empty. Please upload a file or write a paragraph.")
    elif not user_question.strip():
        st.warning("⚠️ Action Blocked: Please type a descriptive question for the engine to analyze.")
    else:
        with st.spinner("Running semantic scanning arrays..."):
            try:
                # Compute raw model inferences
                result = qa_pipe(question=user_question, context=context_text)
                
                # HTML Element injection ensuring text wrapping and structural layouts
                st.markdown(f"""
                    <div class="answer-box">
                        <div class="answer-header">✨ Analysis Complete / Targeted Extraction</div>
                        <div class="answer-body">{result['answer']}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Metadata Metrics Footer
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.caption(f"**Confidence Matrix:** {result['score']:.2%}")
                with col2:
                    st.progress(float(result['score']))
                    
            except Exception as e:
                st.error(f"Runtime Pipeline Error: Semantics processing terminated abruptly. Trace: {e}")
