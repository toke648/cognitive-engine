import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

from sentence_transformers import SentenceTransformer

print("开始...")

# 测试用简单文本
texts = ["你好世界", "这是一个测试"]

print("加载模型...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print("向量化...")
emb = model.encode(texts)

print(f"成功！向量维度: {emb.shape}")  # 应该是 (2, 384)