import streamlit as st
from transformers import pipeline
from pypdf import PdfReader
import re

# Premium global dashboard initialization
st.set_page_config(
    page_title="DeepInsight | Advanced QA Workspace", 
    page_icon="🧠", 
    layout="wide", # Shifts to full-screen responsive widescreen mode
    initial_sidebar_state="collapsed"
)

# Advanced CSS injection for SaaS styling (Glassmorphism, custom scrollbars, animations)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Core Typography & Layout Defaults */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding: 2.5rem 5rem !important; max-width: 1400px; }
    
    /* App Header Design */
    .brand-header {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        letter-spacing: -1px;
        margin-bottom: 0.2rem;
    }
    .brand-subtitle { color: #888893; font-size: 1rem; font-weight: 400; margin-bottom: 2rem; }
    
    /* Workspace Card Panes */
    .workspace-card {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(5px);
    }
    
    /* High-Fidelity Result Display Panel */
    .result-panel {
        background: linear-gradient(145deg, rgba(16, 185, 129, 0.08) 0%, rgba(5, 150, 105, 0.02) 100%);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 14px;
        padding: 22px;
        margin-top: 20px;
        animation: fadeIn 0.4s ease-out;
    }
    .result-header { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700; color: #10b981; margin-bottom: 10px; }
    .result-text { font-size: 1.25rem; font-weight: 600; line-height: 1.6; color: inherit; }
    
    /* Premium Action Button Interactivity */
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: white !important;
        border: none !important;
        padding: 14px 28px !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        width: 100% !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.3) !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px 0 rgba(79, 70, 229, 0.5) !important;
        filter: brightness(1.1);
    }
    
    /* File Attachment Item Badges */
    .file-badge {
        display: inline-flex;
        align-items: center;
        background-color: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.25);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 4px;
        font-weight: 500;
    }
    
    @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
    </style>
""", unsafe_allow_html=True)

# Main Structural Dashboard Header
st.markdown('<div class="brand-header">🧠 DeepInsight Workspace</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-subtitle">Enterprise-grade document intelligence and contextual inquiry workspace.</div>', unsafe_allow_html=True)

# Pipeline Engine Management
@st.cache_resource
def load_qa_engine():
    return pipeline("question-answering", model="abhiMahi/DeepInsight-QA")

with st.spinner("Synchronizing neural processing engines..."):
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

# --- MASTER TWO-COLUMN WORKSPACE GRID ---
left_panel, right_panel = st.columns([5, 6], gap="large")

global_context_text = ""

with left_panel:
    st.markdown('<div class="workspace-card">', unsafe_allow_html=True)
    st.markdown("### 📁 Data Source Controls")
    
    tab1, tab2 = st.tabs(["Document Files", "Plain Text Stream"])
    
    with tab1:
        uploaded_files = st.file_uploader(
            "Upload reference assets:", 
            type=["pdf", "txt"], 
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            extracted_texts = []
            st.markdown("<div style='margin-top: 15px; margin-bottom: 5px;'>**Active Index Stack:**</div>", unsafe_allow_html=True)
            for file in uploaded_files:
                file_extension = file.name.split(".")[-1].lower()
                # Render elegant document item badges
                st.markdown(f'<div class="file-badge">📄 {file.name} ({file.size/1024:.1f} KB)</div>', unsafe_allow_html=True)
                
                if file_extension == "pdf":
                    try:
                        pdf_reader = PdfReader(file)
                        text_list = [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
                        extracted_texts.append("\n".join(text_list))
                    except Exception as e:
                        st.error(f"Error compiling {file.name}: {e}")
                elif file_extension == "txt":
                    extracted_texts.append(file.read().decode("utf-8"))
            
            global_context_text = "\n".join(extracted_texts)
            
    with tab2:
        context_text_input = st.text_area(
            label="Direct target text input allocation:",
            placeholder="Paste document text or standard unstructured paragraphs here...",
            height=280,
            label_visibility="collapsed"
        )
        if context_text_input:
            global_context_text = context_text_input
            
    st.markdown('</div>', unsafe_allow_html=True)

with right_panel:
    st.markdown('<div class="workspace-card">', unsafe_allow_html=True)
    st.markdown("### ❓ Query Interface")
    
    # Initialize session state tracking variable for dynamic chip value injection
    if "current_query" not in st.st.session_state:
        st.session_state.current_query = ""

    user_question = st.text_input(
        "Enter query text parameters:", 
        value=st.session_state.current_query,
        placeholder="What specific fact or variable would you like to extract?",
        label_visibility="collapsed"
    )
    
    # Quick Suggested Query Shortcut Row
    st.markdown("<div style='margin-top: -5px; margin-bottom: 15px;'><span style='font-size: 0.85rem; color: #888;'>Suggestions:</span></div>", unsafe_allow_html=True)
    chip_cols = st.columns(3)
    with chip_cols[0]:
        if st.button("📋 Core Summary", key="chip_1"):
            st.session_state.current_query = "What is the summary or core concept discussed?"
            st.rerun()
    with chip_cols[1]:
        if st.button("🔬 Target Variables", key="chip_2"):
            st.session_state.current_query = "What variables or requirements are listed?"
            st.rerun()
    with chip_cols[2]:
        if st.button("📈 Main Conclusion", key="chip_3"):
            st.session_state.current_query = "What is the final conclusion or result?"
            st.rerun()

    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    
    if st.button("Run Inference Analysis"):
        if not global_context_text.strip():
            st.warning("Analysis Blocked: No source context assets detected inside the workspace.")
        elif not user_question.strip():
            st.warning("Analysis Blocked: Please outline a query value first.")
        else:
            with st.spinner("Executing semantic segment extraction routines..."):
                try:
                    # Dynamically process heavy items down to top-match records
                    optimized_context = get_top_relevant_chunks(user_question, global_context_text)
                    result = qa_pipe(question=user_question, context=optimized_context)
                    
                    # Premium Output UI Render Block
                    st.markdown(f"""
                        <div class="result-panel">
                            <div class="result-header">✨ Extracted Target Segment</div>
                            <div class="result-text">{result['answer']}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Statistical Context Sub-Bar
                    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
                    m_col1, m_col2 = st.columns([1, 2])
                    with m_col1:
                        st.caption(f"**Confidence Matrix:** {result['score']:.2%}")
                    with m_col2:
                        st.progress(float(result['score']))
                        
                except Exception as e:
                    st.error(f"Execution Error: {e}")
                    
    st.markdown('</div>', unsafe_allow_html=True)
