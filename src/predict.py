# \src\predict.py
import config
import json
import gensim.models as models

# 加载模型
model = models.Word2Vec.load(str(config.MODEL_DIR / "word2vec.model"))
# 测试
print(model.wv.most_similar("意识", topn=10))
