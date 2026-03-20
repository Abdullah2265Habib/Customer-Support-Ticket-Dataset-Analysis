import streamlit as st
import time
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class CountVectorizer:
    def __init__(self, max_features: int = 5000, max_n: int = 3):
        self.max_features = max_features
        self.max_n = max_n
        self.vocab: dict = {}
        self.vocab_size: int = 0

    def fit(self, corpus: list):
        counter = Counter()
        for doc in corpus:
            tokens = tokenize_with_ngrams(doc, self.max_n)
            counter.update(tokens)
        top_tokens = [tok for tok, _ in counter.most_common(self.max_features)]
        self.vocab = {tok: i for i, tok in enumerate(top_tokens)}
        self.vocab_size = len(self.vocab)
        return self

    def transform(self, corpus: list) -> torch.Tensor:
        rows, cols, vals = [], [], []
        for row_idx, doc in enumerate(corpus):
            tokens = tokenize_with_ngrams(doc, self.max_n)
            cnt = Counter(tok for tok in tokens if tok in self.vocab)
            for tok, c in cnt.items():
                rows.append(row_idx)
                cols.append(self.vocab[tok])
                vals.append(float(c))
        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.tensor(vals, dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, values, size=(len(corpus), self.vocab_size)).coalesce()

    def fit_transform(self, corpus: list) -> torch.Tensor:
        return self.fit(corpus).transform(corpus)


class TFIDFVectorizer:
    def __init__(self, max_features: int = 5000, max_n: int = 3):
        self.cv = CountVectorizer(max_features=max_features, max_n=max_n)
        self.idf_: torch.Tensor = None

    def _compute_idf(self, bow_sparse: torch.Tensor, N: int) -> torch.Tensor:
        bow_dense = bow_sparse.to_dense()
        df = (bow_dense > 0).sum(dim=0).float()
        idf = torch.log((1 + N) / (1 + df)) + 1
        return idf

    def fit(self, corpus: list):
        bow_sparse = self.cv.fit_transform(corpus)
        N = len(corpus)
        self.idf_ = self._compute_idf(bow_sparse, N)
        return self

    def transform(self, corpus: list) -> torch.Tensor:
        rows, cols, vals = [], [], []
        for row_idx, doc in enumerate(corpus):
            tokens = tokenize_with_ngrams(doc, self.cv.max_n)
            doc_len = max(len(tokens), 1)
            cnt = Counter(tok for tok in tokens if tok in self.cv.vocab)
            for tok, c in cnt.items():
                col_idx = self.cv.vocab[tok]
                tf = c / doc_len
                tfidf = tf * self.idf_[col_idx].item()
                rows.append(row_idx)
                cols.append(col_idx)
                vals.append(tfidf)
        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.tensor(vals, dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, values, size=(len(corpus), self.cv.vocab_size)).coalesce()

    def fit_transform(self, corpus: list) -> torch.Tensor:
        return self.fit(corpus).transform(corpus)

    def transform_single(self, text: str) -> torch.Tensor:
        tokens = tokenize_with_ngrams(text, self.cv.max_n)
        doc_len = max(len(tokens), 1)
        cnt = Counter(tok for tok in tokens if tok in self.cv.vocab)
        vec = torch.zeros(self.cv.vocab_size, dtype=torch.float32)
        for tok, c in cnt.items():
            col_idx = self.cv.vocab[tok]
            tf = c / doc_len
            vec[col_idx] = tf * self.idf_[col_idx].item()
        return vec


def tokenize(text: str) -> list:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text.split()


def generate_ngrams(tokens: list, n: int) -> list:
    return ['_'.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def tokenize_with_ngrams(text: str, max_n: int = 3) -> list:
    tokens = tokenize(text)
    combined = list(tokens)
    for n in range(2, max_n + 1):
        combined.extend(generate_ngrams(tokens, n))
    return combined


def load_glove(filepath: str, dim: int = 300):
    word2vec = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip().split(' ')
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            if len(vec) == dim:
                word2vec[word] = vec
    return word2vec


def text_to_glove_vector(text: str, tfidf_vec: TFIDFVectorizer, emb_layer: nn.Embedding, w2idx: dict, unk_idx: int,
                         dim: int = 300) -> torch.Tensor:
    tokens = tokenize(text)
    if not tokens:
        return torch.zeros(dim, device=device)
    doc_len = len(tokens)
    tfidf_map = Counter(tok for tok in tokens if tok in tfidf_vec.cv.vocab)
    weights = []
    for tok in tokens:
        if tok in tfidf_vec.cv.vocab:
            col = tfidf_vec.cv.vocab[tok]
            tf = tfidf_map.get(tok, 0) / doc_len
            w = tf * tfidf_vec.idf_[col].item()
        else:
            w = 1e-6
        weights.append(max(w, 1e-9))
    weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
    idx_list = [w2idx.get(tok, unk_idx) for tok in tokens]
    idx_t = torch.tensor(idx_list, dtype=torch.long, device=device)
    with torch.no_grad():
        embeds = emb_layer(idx_t)
    weighted = embeds * weights_t.unsqueeze(1)
    return (weighted.sum(dim=0) / weights_t.sum())


def cosine_similarity_sparse_dense(query_tfidf: torch.Tensor, corpus_tfidf: torch.Tensor,
                                   batch_size: int = 2048) -> torch.Tensor:
    q = query_tfidf.to(device).unsqueeze(0)
    q_n = q / (q.norm(dim=1, keepdim=True) + 1e-9)
    N = corpus_tfidf.shape[0]
    sims = torch.zeros(N, device=device)
    corpus_dense = corpus_tfidf.to_dense()
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        chunk = corpus_dense[start:end].to(device)
        c_n = chunk / (chunk.norm(dim=1, keepdim=True) + 1e-9)
        sims[start:end] = (c_n @ q_n.T).squeeze(1)
    return sims


def cosine_similarity_dense(query_vec: torch.Tensor, corpus_mat: torch.Tensor, batch_size: int = 4096) -> torch.Tensor:
    q = query_vec.to(device).unsqueeze(0)
    q_n = q / (q.norm() + 1e-9)
    N = corpus_mat.shape[0]
    sims = torch.zeros(N, device=device)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        chunk = corpus_mat[start:end].to(device)
        c_n = chunk / (chunk.norm(dim=1, keepdim=True) + 1e-9)
        sims[start:end] = (c_n @ q_n.T).squeeze(1)
    return sims


@st.cache_resource(show_spinner="Loading and preprocessing data... This may take a moment on first run.")
def setup_pipeline():
    MY_PATH = 'dataset/customer_support_tickets.csv'
    df = pd.read_csv(MY_PATH)
    FOCUS = ['Ticket Description', 'Ticket Subject', 'Ticket Priority', 'Ticket Type', 'Ticket Channel', 'Resolution']
    df = df[FOCUS].copy()
    df.dropna(subset=['Ticket Description', 'Ticket Priority', 'Ticket Channel', 'Resolution'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    descriptions = df['Ticket Description'].tolist()

    # TF-IDF
    tfidf_vectorizer = TFIDFVectorizer(max_features=5000, max_n=3)
    tfidf_sparse = tfidf_vectorizer.fit_transform(descriptions)

    # GloVe
    GLOVE_FILE = 'glove/glove.6B.300d.txt'
    GLOVE_DIM = 300
    glove = load_glove(GLOVE_FILE, GLOVE_DIM)

    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    glove_words = [PAD_TOKEN, UNK_TOKEN] + list(glove.keys())
    word2idx = {w: i for i, w in enumerate(glove_words)}
    vocab_size_gl = len(glove_words)

    embedding_matrix = np.zeros((vocab_size_gl, GLOVE_DIM), dtype=np.float32)
    rng = np.random.default_rng(SEED)
    embedding_matrix[1] = rng.normal(scale=0.6, size=(GLOVE_DIM,))
    for word, idx in word2idx.items():
        if word in glove:
            embedding_matrix[idx] = glove[word]

    embedding_layer = nn.Embedding(num_embeddings=vocab_size_gl, embedding_dim=GLOVE_DIM, padding_idx=0)
    embedding_layer.weight.data.copy_(torch.from_numpy(embedding_matrix))
    embedding_layer.weight.requires_grad = False
    embedding_layer = embedding_layer.to(device)

    UNK_IDX = word2idx[UNK_TOKEN]

    # Build Glove matrix
    N = len(descriptions)
    BATCH = 512
    glove_matrix = torch.zeros(N, GLOVE_DIM, dtype=torch.float32)
    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        batch = descriptions[start:end]
        vecs = torch.stack(
            [text_to_glove_vector(doc, tfidf_vectorizer, embedding_layer, word2idx, UNK_IDX) for doc in batch])
        glove_matrix[start:end] = vecs.cpu()

    return df, tfidf_vectorizer, tfidf_sparse, embedding_layer, word2idx, UNK_IDX, glove_matrix


def hybrid_search(query: str, df: pd.DataFrame, tfidf_vec: TFIDFVectorizer, tfidf_corpus: torch.Tensor,
                  glove_corpus: torch.Tensor, emb_layer: nn.Embedding, w2idx: dict, unk_idx: int, alpha: float = 0.4,
                  top_k: int = 3) -> pd.DataFrame:
    q_tfidf = tfidf_vec.transform_single(query)
    q_glove = text_to_glove_vector(query, tfidf_vec, emb_layer, w2idx, unk_idx)
    tfidf_scores = cosine_similarity_sparse_dense(q_tfidf, tfidf_corpus).cpu()
    glove_scores = cosine_similarity_dense(q_glove, glove_corpus).cpu()

    final_scores = alpha * tfidf_scores + (1.0 - alpha) * glove_scores

    topk_indices = torch.topk(final_scores, k=top_k).indices.numpy()
    result = df.iloc[topk_indices].copy()
    result['tfidf_score'] = tfidf_scores[topk_indices].numpy().round(4)
    result['glove_score'] = glove_scores[topk_indices].numpy().round(4)
    result['final_score'] = final_scores[topk_indices].numpy().round(4)
    return result


# --- Streamlit UI ---
st.set_page_config(page_title="HSRIS - Customer Support Ticket Retrieval", layout="wide", page_icon="🎫")

# Custom CSS matching the dark-blue analytical dashboard layout
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* App background */
    [data-testid="stAppViewContainer"] {
        background-color: #10213a;
        color: #ffffff;
    }
    
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #132c4a;
        color: #ffffff;
        border-right: none;
        box-shadow: 2px 0 10px rgba(0,0,0,0.2);
    }
    
    /* Sidebar Text / Labels */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] div {
        color: #dbeafe !important;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.9rem;
        letter-spacing: 0.05em;
    }

    /* Main headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        color: #ffffff;
        font-weight: 500;
    }

    .main-title {
        font-size: 2rem;
        font-weight: 400;
        color: #ffffff;
        margin-bottom: 0.2rem;
        border-bottom: 1px solid #1e3a5f;
        padding-bottom: 12px;
    }

    .sub-title {
        font-size: 1rem;
        color: #7dd3fc;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    /* Dashboard Cards */
    .st-card {
        background: #132c4a;
        border: none;
        border-radius: 4px;
        padding: 22px;
        margin-bottom: 22px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease;
    }
    
    .st-card:hover {
        transform: translateY(-2px);
    }
    
    .badge-primary {
        background-color: #2cc1cc;
        color: #0b1a2e;
        padding: 4px 10px;
        border-radius: 3px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 12px;
    }

    .badge-secondary {
        background-color: #f08155;
        color: #ffffff;
        padding: 4px 10px;
        border-radius: 3px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        display: inline-block;
        margin-right: 10px;
    }
    
    .resolution-box {
        background: #0d1e33;
        border-left: 3px solid #2cc1cc;
        padding: 12px 16px;
        border-radius: 2px;
        margin-top: 15px;
        color: #e0f2fe;
        font-weight: 400;
        font-size: 0.9rem;
    }
    
    .score-bar-container {
        width: 100%;
        background-color: #0d1e33;
        border-radius: 2px;
        margin-top: 10px;
        height: 6px;
        overflow: hidden;
    }
    
    .score-bar {
        height: 6px;
        background-color: #2cc1cc;
    }

    /* Target inputs and sliders */
    .stTextArea textarea {
        background-color: #0d1e33;
        color: #ffffff;
        border: 1px solid #1e3a5f;
        border-radius: 4px;
        font-size: 1rem;
        padding: 14px;
        transition: border 0.2s;
    }
    
    .stTextArea textarea:focus {
        border-color: #2cc1cc;
        box-shadow: 0 0 0 1px #2cc1cc;
    }

    /* Target buttons */
    .stButton button {
        background-color: #1b75ff;
        color: #ffffff;
        border: none;
        padding: 12px 24px;
        border-radius: 4px;
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        font-weight: 500;
        transition: all 0.2s ease;
        width: 100%;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    
    .stButton button:hover {
        background-color: #2cc1cc;
        color: #0b1a2e;
        box-shadow: 0 4px 12px rgba(44, 193, 204, 0.4);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>HSRIS Data Exchange Trend dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Hybrid Semantic Customer Support Ticket Retrieval • Dept. of Technology Integration</div>", unsafe_allow_html=True)

df, tfidf_vectorizer, tfidf_sparse, embedding_layer, word2idx, UNK_IDX, glove_matrix = setup_pipeline()

with st.sidebar:
    st.markdown("### 🎛️ Data Settings")
    st.markdown("Adjust configuration weights.")
    alpha = st.slider(
        "Alpha (α) Configuration", 
        min_value=0.0, max_value=1.0, value=0.4, step=0.1,
        help="α=1.0: Only Keyword, α=0.0: Only Semantic"
    )
    st.markdown("---")
    st.markdown("### 🌐 Network Explorer")
    st.markdown("Visualize the 3D analytical distributions models interactively.")
    if st.button("🚀 Launch 3D Modeling"):
        import subprocess
        import sys
        import os
        try:
            # Launch in background so it doesn't block Streamlit
            log_file = open("network_explorer.log", "w")
            # Use creationflags on Windows to fully detach, or just redirect streams
            kwargs = {}
            if os.name == 'nt':
                kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
            subprocess.Popen([sys.executable, "explore_network.py"], stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **kwargs)
            st.success("3D Processing started! ⏳ Please wait ~15-20 seconds for the server to load before clicking the link: http://127.0.0.1:8050")
            st.info("If the site says 'refused to connect', the data is still loading. Please refresh the page in a few seconds.")
        except Exception as e:
            st.error(f"Failed to launch module: {e}")
            
    st.markdown("---")

    st.markdown("---")
    page = st.radio("Navigation", ["🔍 Search Protocol", "📊 System Evaluation"])

    st.markdown("### ℹ️ Protocol Status")
    st.info("System Online. Ready for Hybrid TF-IDF and Neural GloVe pipeline mapping.")

if page == "🔍 Search Protocol":
    st.markdown("<h3 style='font-size:1.1rem; color:#bae6fd; margin-bottom:-0.5rem;'>Input Customer Data Query Protocol:</h3>", unsafe_allow_html=True)

    query = st.text_area("Ticket Extract", value="I need help with my billing and money matters.", height=120, label_visibility="collapsed")

    if st.button("Initialize Analytical Search"):
        if query:
            with st.spinner("Processing semantics engine mapping..."):
                # Prediction setup
                results = hybrid_search(
                    query=query, df=df, tfidf_vec=tfidf_vectorizer, tfidf_corpus=tfidf_sparse, 
                    glove_corpus=glove_matrix, emb_layer=embedding_layer, w2idx=word2idx, unk_idx=UNK_IDX, 
                    alpha=alpha, top_k=3
                )

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Predicted Ticket Type (from highest scored ticket)
            predicted_type = results.iloc[0]['Ticket Type']
            
            st.markdown(f"""
            <div style="background-color: #11253d; border: 1px solid #10b981; border-left: 4px solid #10b981; padding: 18px 24px; border-radius: 4px; margin-bottom: 24px; display: inline-block;">
                <div style="color: #94a3b8; font-weight: 500; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Predicted Protocol</div>
                <div style="color: #34d399; font-weight: 600; font-size: 1.4rem;">{predicted_type}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display side-by-side comparison (TF-IDF vs GloVe based on alpha)
            col1, col2 = st.columns([1.2, 1])
            
            with col1:
                st.markdown(f"<h3 style='color:#e0f2fe; font-size:1.1rem; margin-bottom:1.2rem; border-bottom: 1px solid #1e3a5f; padding-bottom: 8px;'>Hybrid Fusion Rank (α = {alpha})</h3>", unsafe_allow_html=True)
                for i, row in results.iterrows():
                    delay = 0.1 * i
                    st.markdown(f"""
                    <div class="st-card" style="animation-delay: {delay}s;">
                        <div class="badge-primary">{row['Ticket Type']}</div>
                        <div class="badge-secondary">{row['Ticket Priority']} Priority</div>
                        <h4 style="margin-top:10px; color:#ffffff; font-weight: 500; font-size: 1.1rem;">{row['Ticket Subject']}</h4>
                        <p style="color:#bfdbfe; font-size:0.95rem; line-height:1.6; font-weight: 300;">"{row['Ticket Description']}"</p>
                        <div class="resolution-box">
                            <strong style="color: #38bdf8;">Resolution Log:</strong> {row['Resolution']}
                        </div>
                        <div style="margin-top:20px; font-size: 0.85rem; color: #94a3b8; display: flex; justify-content: space-between; font-weight: 500;">
                            <span>TF·IDF: {row['tfidf_score']:.3f}</span>
                            <span>GloVe: {row['glove_score']:.3f}</span>
                            <span style="color:#34d399; font-weight:700;">Final Score (HSRIS): {row['final_score']:.3f}</span>
                        </div>
                        <div class="score-bar-container">
                            <div class="score-bar" style="width: {row['final_score']*100}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            with col2:
                st.markdown("<h3 style='color:#e0f2fe; font-size:1.1rem; margin-bottom:1.2rem; border-bottom: 1px solid #1e3a5f; padding-bottom: 8px;'>Component Pipeline Breakdown</h3>", unsafe_allow_html=True)
                
                # Pure TF-IDF top result
                pure_tfidf_results = hybrid_search(
                    query=query, df=df, tfidf_vec=tfidf_vectorizer, tfidf_corpus=tfidf_sparse, 
                    glove_corpus=glove_matrix, emb_layer=embedding_layer, w2idx=word2idx, unk_idx=UNK_IDX, 
                    alpha=1.0, top_k=1
                )
                
                st.markdown("<h4 style='color:#7dd3fc; font-size:0.9rem; text-transform:uppercase; letter-spacing: 0.05em;'>Tensor Frequency Model (TF-IDF)</h4>", unsafe_allow_html=True)
                for i, row in pure_tfidf_results.iterrows():
                    st.markdown(f"""
                    <div class="st-card" style="box-shadow: none; border: 1px solid #fbbf24; background-color: #0f2742; padding: 18px;">
                        <div class="badge-secondary" style="border: 1px solid #fbbf24; color:#fbbf24; background: #451a03;">Raw Score: {row['tfidf_score']:.3f}</div>
                        <p style="font-size:0.9rem; margin-top:12px; font-weight: 300; color: #bae6fd; line-height: 1.5;">{row['Ticket Description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Pure GloVe top result
                pure_glove_results = hybrid_search(
                    query=query, df=df, tfidf_vec=tfidf_vectorizer, tfidf_corpus=tfidf_sparse, 
                    glove_corpus=glove_matrix, emb_layer=embedding_layer, w2idx=word2idx, unk_idx=UNK_IDX, 
                    alpha=0.0, top_k=1
                )
                
                st.markdown("<h4 style='color:#7dd3fc; font-size:0.9rem; text-transform:uppercase; letter-spacing: 0.05em;'>Neural GloVe Semantic Model</h4>", unsafe_allow_html=True)
                for i, row in pure_glove_results.iterrows():
                    st.markdown(f"""
                    <div class="st-card" style="box-shadow: none; border: 1px solid #2dd4bf; background-color: #0f2742; padding: 18px;">
                        <div class="badge-secondary" style="border: 1px solid #2dd4bf; color:#2dd4bf; background: #042f2e;">Raw Score: {row['glove_score']:.3f}</div>
                        <p style="font-size:0.9rem; margin-top:12px; font-weight: 300; color: #bae6fd; line-height: 1.5;">{row['Ticket Description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("Please enter a ticket query to begin mapping.")


elif page == "📊 System Evaluation":
    st.markdown("<h3 style='font-size:1.1rem; color:#bae6fd; margin-bottom:1rem;'>System Performance & Evaluation Metrics</h3>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["⚡ Execution Time (GPU)", "🎯 Precision@5", "🥇 Qualitative: GloVe vs TF-IDF"])
    
    with tab1:
        st.markdown("#### Execution Time vs Query Batch Size")
        st.markdown("Measures the system\'s hybrid retrieval speed under different batch sizes.")
        if st.button("Run Execution Time Benchmark"):
            with st.spinner("Running Benchmark..."):
                batch_sizes = [1, 8, 16, 32, 64, 128, 256]
                exec_times = []
                descriptions = df['Ticket Description'].tolist()
                
                corpus_dense_n = tfidf_sparse.to_dense().to(device)
                corpus_dense_n = corpus_dense_n / (corpus_dense_n.norm(dim=1, keepdim=True) + 1e-9)
                glove_n = glove_matrix.to(device)
                glove_n = glove_n / (glove_n.norm(dim=1, keepdim=True) + 1e-9)
                
                for b in batch_sizes:
                    if b > len(descriptions): b = len(descriptions)
                    b_queries = descriptions[:b]
                    q_tfidf_list = []
                    q_glove_list = []
                    for q in b_queries:
                        q_tfidf_list.append(tfidf_vectorizer.transform_single(q))
                        q_glove_list.append(text_to_glove_vector(q, tfidf_vectorizer, embedding_layer, word2idx, UNK_IDX))
                    
                    q_tfidf_tensor = torch.stack(q_tfidf_list).to(device)
                    q_glove_tensor = torch.stack(q_glove_list).to(device)
                    
                    start = time.time()
                    with torch.no_grad():
                        q_n_t = q_tfidf_tensor / (q_tfidf_tensor.norm(dim=1, keepdim=True) + 1e-9)
                        q_n_g = q_glove_tensor / (q_glove_tensor.norm(dim=1, keepdim=True) + 1e-9)
                        _ = torch.matmul(q_n_t, corpus_dense_n.T)
                        _ = torch.matmul(q_n_g, glove_n.T)
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    end = time.time()
                    
                    exec_times.append(end - start)
                
                from scipy.interpolate import make_interp_spline
                import numpy as np
                
                fig, ax = plt.subplots(figsize=(8, 4))
                
                # Smooth curve interpolation
                if len(batch_sizes) > 3:
                    spline = make_interp_spline(batch_sizes, exec_times, k=3)
                    x_smooth = np.linspace(min(batch_sizes), max(batch_sizes), 300)
                    y_smooth = spline(x_smooth)
                else:
                    x_smooth = batch_sizes
                    y_smooth = exec_times
                    
                # Plot the smooth line and fill
                ax.plot(x_smooth, y_smooth, color='#7c3aed', linewidth=2.5)
                ax.fill_between(x_smooth, y_smooth, alpha=0.15, color='#7c3aed')
                ax.plot(batch_sizes, exec_times, 'o', color='#7c3aed', markersize=5) # subtle data points

                ax.set_title("Hybrid Search Execution Time", color='#f8fafc', fontsize=14, pad=12)
                ax.set_xlabel("Batch Size", color='#bae6fd', fontsize=11)
                ax.set_ylabel("Execution Time (seconds)", color='#bae6fd', fontsize=11)
                
                # Minimalist axes like the provided graph
                ax.grid(axis='y', alpha=0.2, color='#1e3a5f')
                ax.grid(axis='x', visible=False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_color('#1e3a5f')
                
                fig.patch.set_facecolor('#10213a')
                ax.set_facecolor('#10213a')
                ax.tick_params(colors='#bae6fd', labelsize=10, length=0)
                
                st.pyplot(fig)
                
    with tab2:
        st.markdown("#### Quantitative Evaluation: Precision@5 for Ticket Type Matching")
        if st.button("Run Random Sample Eval (n=500)"):
            with st.spinner("Computing Precision@5..."):
                num_samples = 500
                test_indices = np.random.choice(len(df), min(num_samples, len(df)), replace=False)
                precision_scores = []
                for idx in test_indices:
                    query_row = df.iloc[idx]
                    q_text = query_row['Ticket Description']
                    true_type = query_row['Ticket Type']
                    results = hybrid_search(q_text, df, tfidf_vectorizer, tfidf_sparse, glove_matrix, embedding_layer, word2idx, UNK_IDX, alpha=0.4, top_k=6)
                    top_results = results[results['Ticket Description'] != q_text].head(5)
                    if len(top_results) == 0: continue
                    matches = (top_results['Ticket Type'] == true_type).sum()
                    precision_scores.append(matches / len(top_results))
                
                avg_precision = np.mean(precision_scores) * 100
                st.success(f"**Average Precision@5:** {avg_precision:.2f}%")
                st.progress(avg_precision / 100)
                
    with tab3:
        st.markdown("#### Qualitative Evaluation: Semantic (GloVe) vs Keyword (TF-IDF)")
        if st.button("Find 5 Examples where GloVe matched Ticket Types better"):
            with st.spinner("Mining comparative examples..."):
                found = 0
                for idx in range(len(df)):
                    if found >= 5: break
                    q_text = df.iloc[idx]['Ticket Description']
                    true_type = df.iloc[idx]['Ticket Type']
                    
                    res_tfidf = hybrid_search(q_text, df, tfidf_vectorizer, tfidf_sparse, glove_matrix, embedding_layer, word2idx, UNK_IDX, alpha=1.0, top_k=6)
                    res_tfidf = res_tfidf[res_tfidf['Ticket Description'] != q_text].head(5)
                    
                    res_glove = hybrid_search(q_text, df, tfidf_vectorizer, tfidf_sparse, glove_matrix, embedding_layer, word2idx, UNK_IDX, alpha=0.0, top_k=6)
                    res_glove = res_glove[res_glove['Ticket Description'] != q_text].head(5)
                    
                    if len(res_tfidf) == 0 or len(res_glove) == 0: continue
                    p5_tfidf = (res_tfidf['Ticket Type'] == true_type).sum() / len(res_tfidf)
                    p5_glove = (res_glove['Ticket Type'] == true_type).sum() / len(res_glove)
                    
                    if p5_glove > p5_tfidf:
                        found += 1
                        with st.expander(f"Example {found}: Query Type [{true_type}]", expanded=True):
                            st.write(f"**Query:** {q_text}")
                            colA, colB = st.columns(2)
                            with colA:
                                st.markdown(f"**GloVe** (Precision: {p5_glove*100:.0f}%)")
                                st.info(res_glove.iloc[0]['Ticket Description'][:150] + "...")
                            with colB:
                                st.markdown(f"**TF-IDF** (Precision: {p5_tfidf*100:.0f}%)")
                                st.warning(res_tfidf.iloc[0]['Ticket Description'][:150] + "...")
