import streamlit as st
from transformers import pipeline
from pypdf import PdfReader
import re

# 1. Core Setup
st.set_page_config(
    page_title="DeepInsight | Contextual Intelligence", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# 2. Premium Native CSS Injection (No broken HTML wrappers)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Clean up the default Streamlit header/footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main Layout Spacing */
    .block-container {
        padding-top: 3rem !important;
        max-width: 1200px;
    }
    
    /* Typography */
    h1 {
        font-weight: 700 !important;
        letter-spacing: -1px !important;
        background: linear-gradient(135deg, #ffffff 0%, #a5a5a5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem !important;
    }
    h3 {
        font-weight: 600 !important;
        color: #e2e8f0 !important;
        font-size: 1.2rem !important;
        margin-top: 1rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Target Native Streamlit Inputs */
    .stTextInput input, .stTextArea textarea {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #f8fafc !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        font-size: 1rem !important;
        transition: all 0.2s ease;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
    }
    
    /* File Uploader Restyling */
    [data-testid="stFileUploader"] {
        background-color: #1e293b;
        border: 1px dashed #475569;
        border-radius: 8px;
        padding: 15px;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #6366f1;
        background-color: #1e293b;
    }

    /* Primary Action Button */
    .stButton>button[kind="primary"] {
        background: #6366f1 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button[kind="primary"]:hover {
        background: #4f46e5 !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
    }

    /* Secondary Suggestion Pill Buttons */
    .stButton>button[kind="secondary"] {
        background: transparent !important;
        color: #94a3b8 !important;
        border: 1px solid #334155 !important;
        border-radius: 20px !important;
        padding: 0.25rem 1rem !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button[kind="secondary"]:hover {
        color: #f8fafc !important;
        border-color: #6366f1 !important;
        background: rgba(99, 102, 241, 0.1) !important;
    }
    
    /* Final Output Card */
    .success-card {
        background: #0f172a;
        border-left: 4px solid #10b981;
        border-radius: 0 8px 8px 0;
        padding: 1.5rem;
        margin-top: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-top: 1px solid #1e293b;
        border-right: 1px solid #1e293b;
        border-bottom: 1px solid #1e293b;
    }
    .success-label {
        color: #10b981;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .success-text {
        color: #f8fafc;
        font-size: 1.15rem;
        line-height: 1.6;
        font-weight: 400;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Engine Initialization
@st.cache_resource
def load_qa_engine():
    return pipeline("question-answering", model="abhiMahi/DeepInsight-QA")

with st.spinner("Initializing DeepInsight Engine..."):
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

# 4. App Header
st.markdown("<h1>DeepInsight</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; font-size: 1.1rem; margin-bottom: 3rem;'>Extract precise intelligence from any document.</p>", unsafe_allow_html=True)

# 5. Clean Grid Layout
left_panel, space, right_panel = st.columns([5, 0.5, 5])
global_context_text = ""

# LEFT PANEL: The Brain (Context)
with left_panel:
    st.markdown("### Data Source")
    
    tab1, tab2 = st.tabs(["Upload Files", "Raw Text"])
    
    with tab1:
        uploaded_files = st.file_uploader(
            "Select reference documents", 
            type=["pdf", "txt"], 
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            extracted_texts = []
            for file in uploaded_files:
                file_extension = file.name.split(".")[-1].lower()
                st.caption(f"✓ {file.name} loaded")
                
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
            
    with tab2:
        context_text_input = st.text_area(
            "Paste raw data",
            placeholder="Paste raw text here...",
            height=250,
            label_visibility="collapsed"
        )
        if context_text_input:
            global_context_text = context_text_input

# RIGHT PANEL: The Interface (Query)
with right_panel:
    st.markdown("### Inquiry")
    
    if "current_query" not in st.session_state:
        st.session_state.current_query = ""

    user_question = st.text_input(
        "Ask a question", 
        value=st.session_state.current_query,
        placeholder="What specific detail are you looking for?",
        label_visibility="collapsed"
    )
    
    # Elegant Suggestion Pills
    st.markdown("<div style='margin-top: 10px; margin-bottom: 5px; color: #64748b; font-size: 0.85rem;'>Quick filters:</div>", unsafe_allow_html=True)
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
    
    # Primary Action
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
                    
                    # Crisp Output Card
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
