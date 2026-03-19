import os
import re
import numpy as np
import pandas as pd
from collections import Counter
from embedding_explorer import show_network_explorer

# Duplicated minimal logic to avoid importing streamlit and running app.py UI
class CountVectorizer:
    def __init__(self, max_features=5000, max_n=3):
        self.max_features = max_features
        self.max_n = max_n
        self.vocab = {}
        self.vocab_size = 0

    def fit(self, corpus):
        counter = Counter()
        for doc in corpus:
            tokens = tokenize_with_ngrams(doc, self.max_n)
            counter.update(tokens)
        top_tokens = [tok for tok, _ in counter.most_common(self.max_features)]
        self.vocab = {tok: i for i, tok in enumerate(top_tokens)}
        self.vocab_size = len(self.vocab)
        return self

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

if __name__ == "__main__":
    print("Loading Dataset and extracting Top 5000 vocabulary...")
    # Load vocabulary
    MY_PATH = 'dataset/customer_support_tickets.csv'
    df = pd.read_csv(MY_PATH)
    df.dropna(subset=['Ticket Description'], inplace=True)
    descriptions = df['Ticket Description'].tolist()
    
    cv = CountVectorizer(max_features=5000, max_n=3)
    cv.fit(descriptions)
    vocab_words = list(cv.vocab.keys())
    
    # Load GloVe
    print("Loading GloVe vectors...")
    glove = load_glove('glove/glove.6B.300d.txt', 300)
    
    # Build embeddings array matching vocab_words
    print("Matching embeddings...")
    embeddings_list = []
    final_vocab = []
    rng = np.random.default_rng(42)
    unknown_vec = rng.normal(scale=0.6, size=(300,))
    
    for word in vocab_words:
        # unigrams exist in glove, n-grams won't
        if word in glove:
            embeddings_list.append(glove[word])
            final_vocab.append(word)
        else:
            # Check if it's an n-gram composed of individual GloVe words
            parts = word.split('_')
            vecs = [glove.get(p, unknown_vec) for p in parts]
            mean_vec = np.mean(vecs, axis=0) if vecs else unknown_vec
            embeddings_list.append(mean_vec)
            final_vocab.append(word)
            
    embeddings_matrix = np.array(embeddings_list)
    
    print(f"Launching Network Explorer with {len(final_vocab)} nodes...")
    show_network_explorer(corpus=final_vocab, embeddings=embeddings_matrix)
