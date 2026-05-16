# \src\database.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 你的所有文档向量
doc_embeddings = np.load("doc_embeddings.npy")  # shape: (n_docs, 384)

# 查询词向量
query_vec = model.encode(["意识"])  # shape: (1, 384)

# 计算相似度，找Top-K
similarities = cosine_similarity(query_vec, doc_embeddings)[0]
top_k_idx = similarities.argsort()[-5:][::-1]