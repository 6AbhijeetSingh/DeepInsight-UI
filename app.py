import streamlit as st
from transformers import pipeline
from pypdf import PdfReader
import re

# 1. Core Setup
st.set_page_config(
    page_title="DeepInsight | AI Intelligence", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# 2. Premium Adaptive CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    header, footer { visibility: hidden; }
    .block-container { padding-top: 2rem !important; max-width: 1200px; }
    
    .brand-container { display: flex; align-items: center; gap: 15px; margin-bottom: 0.5rem; }
    .logo-svg { width: 42px; height: 42px; filter: drop-shadow(0 0 10px rgba(139, 92, 246, 0.5)); }
    .animated-logo-text {
        font-size: 2.8rem !important; font-weight: 800 !important; letter-spacing: -1.5px !important; margin: 0 !important;
        background: linear-gradient(to right, #8b5cf6, #3b82f6, #8b5cf6); background-size: 200% auto; color: #fff;
        background-clip: text; -webkit-background-clip: text; -webkit-text-fill-color: transparent; animation: shine 3s linear infinite;
    }
    .brand-subtitle { color: var(--text-color); opacity: 0.7; font-size: 1.15rem; margin-bottom: 3rem; }
    
    [data-baseweb="tab-list"] { gap: 8px; background-color: var(--secondary-background-color); padding: 6px; border-radius: 12px; border: 1px solid rgba(128, 128, 128, 0.2); }
    [data-baseweb="tab"] { background-color: transparent !important; color: var(--text-color) !important; opacity: 0.6; border-radius: 8px !important; padding: 8px 16px !important; border: none !important; font-weight: 600 !important; }
    [aria-selected="true"] { background-color: var(--background-color) !important; color: var(--text-color) !important; opacity: 1 !important; box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important; }
    div[data-baseweb="tab-highlight"] { display: none !important; }

    .stTextInput input, .stTextArea textarea { background-color: var(--secondary-background-color) !important; border: 1px solid rgba(128, 128, 128, 0.2) !important; color: var(--text-color) !important; border-radius: 8px !important; padding: 12px 16px !important; }
    .stTextInput input:focus, .stTextArea textarea:focus { border-color: #8b5cf6 !important; box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.2) !important; }
    [data-testid="stFileUploader"] { background-color: var(--secondary-background-color); border: 1px dashed rgba(128, 128, 128, 0.4); border-radius: 12px; padding: 15px; }

    .stButton>button[kind="primary"] { background: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%) !important; color: white !important; border: none !important; border-radius: 8px !important; padding: 0.75rem 1.5rem !important; font-weight: 600 !important; width: 100% !important; }
    .stButton>button[kind="primary"]:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(139, 92, 246, 0.4) !important; }
    .stButton>button[kind="secondary"] { background: transparent !important; color: var(--text-color) !important; border: 1px solid rgba(128, 128, 128, 0.3) !important; border-radius: 20px !important; padding: 0.25rem 1rem !important; font-size: 0.85rem !important; opacity: 0.8; }
    
    [data-testid="stAlert"] { background-color: rgba(245, 158, 11, 0.1) !important; color: #fcd34d !important; border: 1px solid rgba(245, 158, 11, 0.2) !important; border-radius: 8px !important; }
    .success-card { background: var(--secondary-background-color); border-left: 4px solid #10b981; border-radius: 8px; padding: 1.5rem; margin-top: 1.5rem; border: 1px solid rgba(16, 185, 129, 0.2); }
    .success-label { color: #10b981; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; margin-bottom: 0.5rem; }
    .success-text { color: var(--text-color); font-size: 1.15rem; line-height: 1.6; }
    
    @keyframes shine { to { background-position: 200% center; } }
    </style>
""", unsafe_allow_html=True)

# 3. Engine Initialization
@st.cache_resource
def load_qa_engine():
    return pipeline("question-answering", model="abhiMahi/DeepInsight-QA")

with st.spinner("Initializing DeepInsight Neural Engine..."):
    qa_pipe = load_qa_engine()

def get_top_relevant_chunks(question, text, chunk_size=2000, top_n=3):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    q_tokens = set(re.findall(r'\w+', question.lower()))
    scored_chunks = []
    for chunk in chunks:
        c_tokens = set(re.findall(r'\w+', chunk.lower()))
        score = len(q_tokens.intersection(c_tokens))
        scored_chunks.append((score, chunk))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return "\n".join([x[1] for x in scored_chunks[:top_n]])

# 4. Premium Animated Header
st.markdown("""
    <div class="brand-container">
        <svg class="logo-svg" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="url(#paint0_linear)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M2 17L12 22L22 17" stroke="url(#paint0_linear)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M2 12L12 17L22 12" stroke="url(#paint0_linear)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <defs>
                <linearGradient id="paint0_linear" x1="2" y1="2" x2="22" y2="22" gradientUnits="userSpaceOnUse">
                    <stop stop-color="#8B5CF6"/><stop offset="1" stop-color="#3B82F6"/>
                </linearGradient>
            </defs>
        </svg>
        <h1 class="animated-logo-text">DeepInsight</h1>
    </div>
    <div class="brand-subtitle">Extract precise intelligence from any document.</div>
""", unsafe_allow_html=True)

# Initialize Session State tracking to prevent infinite processing loops
if "cached_text" not in st.session_state:
    st.session_state.cached_text = ""
if "last_uploaded_filenames" not in st.session_state:
    st.session_state.last_uploaded_filenames = []

# 5. Clean Grid Layout
left_panel, space, right_panel = st.columns([5, 0.5, 5])

with left_panel:
    st.markdown("<h3 style='font-weight: 600;'>Data Source</h3>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Upload Files", "Raw Text"])
    
    with tab1:
        uploaded_files = st.file_uploader(
            "Select reference documents", 
            type=["pdf", "txt"], 
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            current_filenames = [f.name + str(f.size) for f in uploaded_files]
            
            # CRITICAL OPTIMIZATION: Only parse the book if the file configuration actually changed!
            if current_filenames != st.session_state.last_uploaded_filenames:
                with st.spinner("Extracting book context securely..."):
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
                    
                    st.session_state.cached_text = "\n".join(extracted_texts)
                    st.session_state.last_uploaded_filenames = current_filenames
            
            for file in uploaded_files:
                st.caption(f"✓ {file.name} loaded")
            global_context_text = st.session_state.cached_text
            
        else:
            st.session_state.cached_text = ""
            st.session_state.last_uploaded_filenames = []
            global_context_text = ""
            
    with tab2:
        context_text_input = st.text_area("Paste raw data", placeholder="Paste raw text here...", height=250, label_visibility="collapsed")
        if context_text_input:
            global_context_text = context_text_input

with right_panel:
    st.markdown("<h3 style='font-weight: 600;'>Inquiry</h3>", unsafe_allow_html=True)
    
    if "current_query" not in st.session_state:
        st.session_state.current_query = ""

    user_question = st.text_input(
        "Ask a question", 
        value=st.session_state.current_query,
        placeholder="What specific detail are you looking for?",
        label_visibility="collapsed"
    )
    
    st.markdown("<div style='margin-top: 10px; margin-bottom: 5px; opacity: 0.7; font-size: 0.85rem;'>Quick filters:</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Core Theme", type="secondary", use_container_width=True):
            st.session_state.current_query = "What is the primary core theme?"
            st.rerun()
    with col2:
        if st.button("Key Variables", type="secondary", use_container_width=True):
            st.session_state.current_query = "What are the key variables mentioned?"
            st.rerun()
    with col3:
        if st.button("Conclusion", type="secondary", use_container_width=True):
            st.session_state.current_query = "What is the final conclusion?"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Extract Intelligence", type="primary"):
        if not global_context_text.strip():
            st.warning("Please provide a data source first.")
        elif not user_question.strip():
            st.warning("Please enter a valid question.")
        else:
            with st.spinner("Processing..."):
                try:
                    optimized_context = get_top_relevant_chunks(user_question, global_context_text)
                    result = qa_pipe(question=user_question, context=optimized_context)
                    
                    st.markdown(f"""
                        <div class="success-card">
                            <div class="success-label">Target Extraction</div>
                            <div class="success-text">{result['answer']}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.progress(float(result['score']), text=f"Confidence Matrix: {result['score']:.2%}")
                        
                except Exception as e:
                    st.error(f"Execution Error: {e}")
