import streamlit as st
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
        background-color: #0b1a2e;
        color: #dbeafe;
    }
    
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #11253d;
        color: #dbeafe;
        border-right: 1px solid #1e3a5f;
    }
    
    /* Sidebar Text / Labels */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] div {
        color: #93c5fd !important;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #f0fdf4 !important;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.9rem;
        letter-spacing: 0.05em;
    }

    /* Main headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        color: #f8fafc;
        font-weight: 500;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 300;
        color: #ffffff;
        margin-bottom: 0.2rem;
        border-bottom: 2px solid #1e3a5f;
        padding-bottom: 12px;
    }

    .sub-title {
        font-size: 1.05rem;
        color: #7dd3fc;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* Dashboard Cards */
    .st-card {
        background: #153152;
        border: 1px solid #234773;
        border-radius: 4px;
        padding: 22px;
        margin-bottom: 22px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        transition: border 0.2s ease, transform 0.2s ease;
    }
    
    .st-card:hover {
        border-color: #38bdf8;
        transform: translateY(-2px);
    }
    
    .badge-primary {
        background-color: #0284c7;
        color: #ffffff;
        padding: 4px 10px;
        border-radius: 3px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 12px;
        letter-spacing: 0.02em;
    }

    .badge-secondary {
        background-color: #0c4a6e;
        color: #38bdf8;
        padding: 4px 10px;
        border-radius: 3px;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        display: inline-block;
        margin-right: 10px;
        border: 1px solid #0284c7;
    }
    
    .resolution-box {
        background: #0f2742;
        border-left: 3px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 2px;
        margin-top: 15px;
        color: #bae6fd;
        font-weight: 400;
        font-size: 0.9rem;
    }
    
    .score-bar-container {
        width: 100%;
        background-color: #1e3a5f;
        border-radius: 2px;
        margin-top: 10px;
        height: 6px;
        overflow: hidden;
    }
    
    .score-bar {
        height: 6px;
        background-color: #2dd4bf;
    }

    /* Target inputs and sliders */
    .stTextArea textarea {
        background-color: #0f2742;
        color: #ffffff;
        border: 1px solid #1e3a5f;
        border-radius: 4px;
        font-size: 1rem;
        padding: 14px;
        transition: border 0.2s;
    }
    
    .stTextArea textarea:focus {
        border-color: #38bdf8;
        box-shadow: 0 0 0 1px #38bdf8;
    }

    /* Target buttons */
    .stButton button {
        background-color: #0284c7;
        color: #ffffff;
        border: 1px solid #38bdf8;
        padding: 12px 24px;
        border-radius: 4px;
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        font-weight: 600;
        transition: all 0.2s ease;
        width: 100%;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    
    .stButton button:hover {
        background-color: #0369a1;
        color: #ffffff;
        border-color: #7dd3fc;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.4);
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
        try:
            # Launch in background so it doesn't block Streamlit
            subprocess.Popen([sys.executable, "explore_network.py"])
            st.success("3D Processing running! Tunnel ready locally at http://127.0.0.1:8050")
        except Exception as e:
            st.error(f"Failed to launch module: {e}")
            
    st.markdown("---")
    st.markdown("### ℹ️ Protocol Status")
    st.info("System Online. Ready for Hybrid TF-IDF and Neural GloVe pipeline mapping.")

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
