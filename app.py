import streamlit as st
import tempfile
import uuid

from src.document_loader import load_pdf
from src.chunker import split_docs
from src.embedding import EmbeddingManager
from src.vector_store import VectorStore
from src.retriever import RAGRetriever
from src.rag_pipeline import RAGPipeline



st.set_page_config(
    page_title="ResearchGPT",
    page_icon="🔬",
    layout="wide"
)



st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

/* ── CSS Variables ── */
:root {
    --bg-root:      #080a12;
    --bg-surface:   #0e1018;
    --bg-raised:    #13161f;
    --bg-overlay:   #191d29;
    --border-dim:   rgba(255,255,255,0.06);
    --border-mid:   rgba(255,255,255,0.10);
    --amber-bright: #f59e0b;
    --amber-mid:    #d97706;
    --amber-dim:    rgba(245,158,11,0.15);
    --amber-glow:   rgba(245,158,11,0.08);
    --text-primary: #f0f2f8;
    --text-secondary: #8b92a8;
    --text-muted:   #4e5568;
    --green-ok:     #10b981;
}

/* ── Base ── */
*, *::before, *::after {
    box-sizing: border-box;
    font-family: 'Inter', system-ui, sans-serif;
}

.stApp {
    background-color: var(--bg-root) !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(245,158,11,0.07) 0%, transparent 60%),
        radial-gradient(circle at 0% 100%, rgba(245,158,11,0.04) 0%, transparent 40%);
    background-attachment: fixed;
    color: var(--text-primary) !important;
}

/* Dot grid texture on main bg */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: radial-gradient(circle, rgba(255,255,255,0.025) 1px, transparent 1px);
    background-size: 28px 28px;
    pointer-events: none;
    z-index: 0;
}

/* Global text */
body,
p,
span,
div,
li,
label {
    color: var(--text-primary);
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text-primary) !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text-primary) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border-dim) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* ── Metric Cards ── */
[data-testid="metric-container"] {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border-dim) !important;
    border-top: 2px solid var(--amber-mid) !important;
    padding: 18px 20px !important;
    border-radius: 12px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

[data-testid="metric-container"]:hover {
    border-top-color: var(--amber-bright) !important;
    box-shadow: 0 0 24px var(--amber-dim) !important;
}

[data-testid="stMetricLabel"] *,
[data-testid="stMetricLabel"] p {
    color: var(--text-secondary) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

[data-testid="stMetricValue"] *,
[data-testid="stMetricValue"] div {
    color: var(--text-primary) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}

[data-testid="stMetricDelta"] * {
    color: var(--green-ok) !important;
}

/* ── Alert / Info Boxes ── */
.stAlert {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border-dim) !important;
    border-left: 3px solid var(--amber-mid) !important;
    border-radius: 10px !important;
}

.stAlert * { color: var(--text-primary) !important; }
.stAlert svg { fill: var(--amber-bright) !important; }

div[data-testid="stNotification"] {
    background: var(--bg-raised) !important;
}

div[data-testid="stNotification"] * {
    color: var(--text-primary) !important;
}

/* ── Chat Messages ── */
[data-testid="stChatMessage"] {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: 14px !important;
    padding: 14px 18px !important;
    margin-bottom: 8px !important;
}

[data-testid="stChatMessage"] * {
    color: var(--text-primary) !important;
}

/* Distinguish user messages with amber left-stripe */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    border-left: 3px solid var(--amber-mid) !important;
    background: linear-gradient(135deg, rgba(245,158,11,0.05) 0%, var(--bg-raised) 60%) !important;
}

/* ── Chat Input ── */
[data-testid="stChatInputContainer"],
.stChatInputContainer {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: 14px !important;
    padding: 4px 6px !important;
    box-shadow: 0 0 0 0 transparent !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

/* Chat input wrapper */
[data-testid="stChatInputContainer"] {
    background: #000000 !important;
    border: 1px solid #333 !important;
    border-radius: 12px !important;
}

/* Actual textarea */
[data-testid="stChatInputContainer"] textarea {
    background: transparent !important;
    color: white !important;
    caret-color: white !important;
    border: none !important;
    box-shadow: none !important;
}

/* Placeholder */
[data-testid="stChatInputContainer"] textarea::placeholder {
    color: #aaa !important;
}
/* Focus */
[data-testid="stChatInputContainer"]:focus-within {
    border-color: #f59e0b !important;
    box-shadow: 0 0 10px rgba(245,158,11,0.5) !important;
}

[data-testid="stChatInputContainer"]:focus-within {
    border-color: var(--amber-mid) !important;
    box-shadow: 0 0 0 3px var(--amber-dim) !important;
}

[data-testid="stChatInputContainer"] textarea {
    background: transparent !important;
    color: var(--text-primary) !important;
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
}

[data-testid="stChatInputContainer"] textarea::placeholder {
    color: var(--text-muted) !important;
}

/* Send button */
[data-testid="stChatInputContainer"] button {
    background: var(--amber-mid) !important;
    border-radius: 10px !important;
    color: #000 !important;
    transition: background 0.2s, transform 0.15s !important;
}

[data-testid="stChatInputContainer"] button:hover {
    background: var(--amber-bright) !important;
    transform: scale(1.05) !important;
}

/* ── Sidebar Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--amber-mid) 0%, var(--amber-bright) 100%) !important;
    color: #09090b !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 20px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.01em !important;
    transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s !important;
    width: 100% !important;
    box-shadow: 0 4px 16px rgba(217,119,6,0.35) !important;
}

.stButton > button:hover {
    opacity: 0.92 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(245,158,11,0.45) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-raised) !important;
    border: 1.5px dashed rgba(245,158,11,0.35) !important;
    border-radius: 12px !important;
    padding: 14px !important;
    transition: border-color 0.2s !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--amber-mid) !important;
}

[data-testid="stFileUploader"] * {
    color: var(--text-primary) !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] * {
    color: var(--text-secondary) !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: var(--amber-mid) !important;
}

.stSpinner * { color: var(--text-secondary) !important; }

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border-dim) !important;
    margin: 24px 0 !important;
}

/* ── Caption ── */
.stCaption, .stCaption * {
    color: var(--text-muted) !important;
    font-size: 0.78rem !important;
}

/* ── Success ── */
.element-container .stAlert[data-baseweb="notification"] {
    background: rgba(16,185,129,0.08) !important;
    border-left: 3px solid var(--green-ok) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; background: var(--bg-root); }
::-webkit-scrollbar-thumb { background: var(--bg-overlay); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: rgba(245,158,11,0.4); }

/* ── Sidebar info card ── */
.sidebar-info-card {
    background: var(--bg-raised);
    border: 1px solid var(--border-dim);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
}

.sidebar-info-card .card-label {
    color: var(--amber-bright) !important;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.sidebar-info-card .card-item {
    color: var(--text-secondary) !important;
    font-size: 0.85rem;
    padding: 4px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

.sidebar-info-card .card-item::before {
    content: '';
    width: 4px;
    height: 4px;
    border-radius: 50%;
    background: var(--amber-mid);
    flex-shrink: 0;
}

/* ── Welcome card ── */
.welcome-card {
    background: var(--bg-raised);
    border: 1px solid var(--border-dim);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
}

.welcome-card h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text-primary) !important;
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 16px;
}

.welcome-card .prompt-label {
    color: var(--text-muted) !important;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 500;
    margin-bottom: 12px;
}

.welcome-card .prompt-chip {
    background: var(--bg-overlay);
    border: 1px solid var(--border-dim);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.88rem;
    color: var(--text-secondary) !important;
    cursor: default;
    transition: border-color 0.15s, color 0.15s;
}

.welcome-card .prompt-chip:hover {
    border-color: rgba(245,158,11,0.3);
    color: var(--text-primary) !important;
}

/* ── Upload prompt banner ── */
.upload-banner {
    background: var(--bg-raised);
    border: 1px solid var(--border-dim);
    border-radius: 14px;
    padding: 20px 24px;
    display: flex;
    align-items: center;
    gap: 14px;
    color: var(--text-secondary) !important;
    font-size: 0.93rem;
}

.upload-banner .icon {
    font-size: 1.5rem;
    flex-shrink: 0;
}

.upload-banner strong {
    color: var(--amber-bright) !important;
}


""", unsafe_allow_html=True)

# -----------------------------------
# Hero Section
# -----------------------------------

st.markdown("""
<div style='display:flex; align-items:center; gap:14px; margin-bottom:28px; padding-top:4px;'>
    <div style='
        width:40px; height:40px;
        background:linear-gradient(135deg,#d97706,#f59e0b);
        border-radius:10px;
        display:flex; align-items:center; justify-content:center;
        font-size:1.25rem; flex-shrink:0;
        box-shadow: 0 4px 16px rgba(245,158,11,0.3);
    '>🔬</div>
    <div>
        <h2 style='
            font-family:Space Grotesk,sans-serif;
            color:#f0f2f8 !important;
            font-size:1.5rem;
            font-weight:700;
            margin:0;
            letter-spacing:-0.02em;
        '>ResearchGPT</h2>
        <p style='color:#4e5568 !important; font-size:0.78rem; margin:0; letter-spacing:0.06em; text-transform:uppercase; font-weight:500;'>
            Intelligent Document Analysis
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------
# Session State
# -----------------------------------

if "indexed" not in st.session_state:
    st.session_state.indexed = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_count" not in st.session_state:
    st.session_state.pdf_count = 0

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "db_count" not in st.session_state:
    st.session_state.db_count = 0

# -----------------------------------
# Components
# -----------------------------------

embedding_manager = EmbeddingManager()
vector_store = VectorStore()

retriever = RAGRetriever(
    embedding_manager,
    vector_store
)

rag_pipeline = RAGPipeline(
    retriever
)

# -----------------------------------
# Sidebar
# -----------------------------------

with st.sidebar:

    st.markdown("""
    <div style='padding:8px 0 20px;'>
        <p style='
            color:#8b92a8 !important;
            font-size:0.7rem;
            letter-spacing:0.1em;
            text-transform:uppercase;
            font-weight:600;
            margin:0 0 16px;
        '>Document Library</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
<div class='sidebar-info-card'>
    <div class='card-label'>Supported Formats</div>
    <div class='card-item'>Research Papers</div>
    <div class='card-item'>Technical Reports</div>
    <div class='card-item'>Whitepapers</div>
    <div class='card-item'>PDF Documents</div>
</div>
""", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)

    if uploaded_files:

        if st.button("Process Documents →", use_container_width=True):

            with st.spinner("Indexing documents..."):

                all_chunks = []

                for uploaded_file in uploaded_files:

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as tmp:

                        tmp.write(uploaded_file.read())
                        pdf_path = tmp.name

                    docs = load_pdf(pdf_path)
                    chunks = split_docs(docs)

                    for chunk in chunks:
                        chunk.metadata["source"] = uploaded_file.name

                    all_chunks.extend(chunks)

                texts = [chunk.page_content for chunk in all_chunks]
                embeddings = embedding_manager.generate_embedding(texts)
                ids = [str(uuid.uuid4()) for _ in all_chunks]
                metadatas = [chunk.metadata for chunk in all_chunks]

                vector_store.reset_collection()
                vector_store.add_documents(
                    ids=ids,
                    documents=texts,
                    embeddings=embeddings.tolist(),
                    metadatas=metadatas
                )

                st.session_state.indexed = True
                st.session_state.pdf_count = len(uploaded_files)
                st.session_state.chunk_count = len(all_chunks)
                st.session_state.db_count = vector_store.collection.count()

            st.success(f"✓ {len(uploaded_files)} PDF(s) indexed · {len(all_chunks)} chunks ready")

# -----------------------------------
# Main Area
# -----------------------------------

if not st.session_state.indexed:

    st.markdown("""
<div class='upload-banner'>
    <span class='icon'>📄</span>
    <span>Upload one or more PDFs in the sidebar, then click
    <strong>Process Documents →</strong> to begin analysis.</span>
</div>
""", unsafe_allow_html=True)

else:

    # Dashboard Metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("PDFs Loaded", st.session_state.pdf_count)
    with col2:
        st.metric("Text Chunks", st.session_state.chunk_count)
    with col3:
        st.metric("Vectors Stored", st.session_state.db_count)

    st.divider()

    # Welcome Screen
    if len(st.session_state.messages) == 0:

        st.markdown("""
<div class='welcome-card'>
    <h3>What would you like to know?</h3>
    <div class='prompt-label'>Try asking</div>
    <div class='prompt-chip'>What is this research paper about?</div>
    <div class='prompt-chip'>Summarize the key findings across all papers</div>
    <div class='prompt-chip'>Compare the methodologies used</div>
    <div class='prompt-chip'>What are the main limitations discussed?</div>
    <div class='prompt-chip'>What future work is suggested?</div>
</div>
""", unsafe_allow_html=True)

    # Chat History
    for message in st.session_state.messages:

        avatar = "👤" if message["role"] == "user" else "🔬"

        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # User Input
    query = st.chat_input("Ask a question about your documents...")

    if query:

        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user", avatar="👤"):
            st.markdown(query)

        with st.chat_message("assistant", avatar="🔬"):
            with st.spinner("Thinking..."):
                answer = rag_pipeline.generate_answer(query)
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

# -----------------------------------
# Footer
# -----------------------------------

st.divider()

st.caption("Built with Streamlit · ChromaDB · SentenceTransformers · Groq · LangChain")