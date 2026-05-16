# \src\vectorizer.py
import config
import json
from gensim.models import Word2Vec  # 导入Word2Vec模型用于训练

with open(config.RAW_DATA_DIR / "tokenized.json", encoding='utf-8') as f:
    sectences = json.load(f) # 按行读取，过滤空行

# 训练
model = Word2Vec(
    sentences=sectences,
    vector_size=64, # 词向量的维度
    window=5, # 上下文窗口大小
    min_count=2, # 最小出现次数
    epochs=50, # 训练轮数
    workers=4, # 并行数
)

# 保存模型
model.save(str(config.MODEL_DIR / "word2vec.model")) # 转换为字符串，因为Word2Vec模型的save方法需要字符串路径

# 测试
print(model.wv.most_similar("意识", topn=10))

