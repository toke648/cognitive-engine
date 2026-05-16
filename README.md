我来帮你整理一份完整的技术文档，基于我们长达数小时的对话和你实际完成的工作。

---

# 个人认知引擎 - 技术文档

## 一、项目概述

### 1.1 项目定义
**个人认知引擎** —— 将个人思考过程数据化、向量化、可检索的系统。它不是笔记软件，而是个人认知结构的外部镜像。

### 1.2 核心目标
- 自动筛选有价值的个人思考内容（86%准确率）
- 构建个人专属的语义空间（64维词向量）
- 实现语义检索 + 关键词高亮
- 辅助阅读、写作和跨领域联想

### 1.3 技术栈
| 类别 | 技术 | 用途 |
|------|------|------|
| 语言 | Python 3.10 | 主开发语言 |
| 数据 | Pandas, NumPy | 数据处理、向量存储 |
| NLP | jieba | 中文分词 |
| 机器学习 | gensim (Word2Vec) | 词向量训练 |
| AI | DeepSeek API | 价值分类 |
| 框架 | LangChain | Prompt 管理 |
| 存储 | JSON, NPY | 结构化数据、向量 |

---

## 二、系统架构

### 2.1 三层架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      应用层（待开发）                        │
│    检索界面  │  荧光标注  │  阅读插件  │  可视化           │
├─────────────────────────────────────────────────────────────┤
│                      服务层（核心）                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 混合检索引擎：语义检索 + 词元匹配 + 扩展推荐        │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 认知推理层：Word2Vec + Sentence-BERT               │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                      数据层（已完成）                        │
│  原始txt → CSV → 分类CSV → tokenized.json → NPY向量        │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 数据流水线

```
DeepSeek聊天记录(.txt)
    ↓ preprocessor.py
原始CSV (title, content, created_time)
    ↓ classifier.py (AI分类，86%准确率)
分类CSV (加上 pred_label: 1/0/-1)
    ↓ tokenizer.py (清洗+分词)
tokenized.json (每条笔记的词列表)
    ↓ vectorizer.py (Word2Vec训练)
word2vec.model (64维词向量)
    ↓ sentence.py (Sentence-BERT)
note_embeddings.npy (文档向量，用于检索)
```

---

## 三、模块详解

### 3.1 模块清单

| 模块 | 文件 | 功能 | 状态 |
|------|------|------|------|
| 数据导出 | preprocessor.py | txt → CSV | ✅ 完成 |
| 价值分类 | classifier.py | AI判断1/0/-1 | ✅ 86%准确率 |
| 文本清洗 | tokenizer.py | 去重、去标点、分词 | ✅ 完成 |
| 词向量 | vectorizer.py | Word2Vec训练 | ✅ 完成 |
| 语义向量 | sentence.py | Sentence-BERT向量化 | ⚠️ 环境问题 |
| 检索 | retriever.py | 相似度计算 | 🔄 进行中 |

### 3.2 核心模块代码示例

#### tokenizer.py - 分词器

```python
class Tokenizer:
    def clean(self, text):
        # 只保留中文、英文、数字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def dedup_lines(self, text):
        # 去重 + 去空行
        lines = text.split('\n')
        unique = []
        for line in lines:
            line_clean = line.strip()
            if line_clean and line_clean not in unique:
                unique.append(line_clean)
        return ' '.join(unique)
    
    def tokenize(self, text):
        words = jieba.lcut(text)
        return [w for w in words if len(w) >= 2]
```

#### classifier.py - 价值分类提示词

```python
PROMPT = """
判断以下笔记是否有长期价值。

规则：
- 输出 1：你自己的思考（自我剖析、原创观点、学习试错、灵感画面）
- 输出 0：AI给你的信息（科普解释、信息查询、书单推荐）
- 输出 -1：介于两者之间

示例：
"我总在和他人的比较里内耗" → 1
"特斯拉可以自己接单吗" → 0

只输出数字。
"""
```

#### vectorizer.py - Word2Vec训练

```python
# 参数配置
model = Word2Vec(
    sentences=sentences,
    vector_size=64,      # 维度
    window=5,            # 上下文窗口
    min_count=2,         # 最小词频
    workers=4
)
model.save(str(MODEL_DIR / "word2vec.model"))
```

---

## 四、数据格式

### 4.1 输入格式（原始txt）

```
标题: 意义与存在的困境
ID: xxx
创建时间: 2026-04-08T14:00:00
更新时间: 2026-04-08T16:30:00
模型: deepseek-chat
============================================================
[user]: 我想我该给自己做一次自我校准了...
[user]: 人死后，亲友提起你，只剩一句"那是个怪人"...
```

### 4.2 中间格式（CSV）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | string | 笔记唯一标识 |
| title | string | 从第一行提取 |
| content | string | 完整对话内容 |
| created_time | string | ISO格式时间 |
| pred_label | int | 1有价值 / 0无价值 / -1人工判断 |

### 4.3 分词结果（JSON）

```json
[
    ["意识", "神经元", "信号处理", "涌现"],
    ["内耗", "虚无", "存在", "困境"],
    ["线粒体", "共生", "能量", "细胞"]
]
```

### 4.4 向量存储（NPY）

- `word2vec.model`: gensim格式，64维词向量
- `note_embeddings.npy`: NumPy数组，shape=(n_docs, 384)

---

## 五、性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| 分类准确率 | 86% | 在50条测试集上 |
| 分类召回率(有价值) | 55% | 11条中认出6条 |
| 分类精确率(无价值) | 90% | 过滤垃圾内容可靠 |
| 词向量维度 | 64 | 适合小数据量 |
| 有价值笔记数量 | ~50条 | 当前数据量 |

---

## 六、已知问题与解决方案

| 问题 | 状态 | 解决方案 |
|------|------|----------|
| Word2Vec相似度偏低 | ⚠️ | 数据量小，改用Sentence-BERT |
| Sentence-BERT环境问题 | ⚠️ | 新建conda环境或用TF-IDF临时方案 |
| 有价值的笔记召回率低 | ⚠️ | 优化提示词，增加Few-shot示例 |
| 数据量不足 | 🔄 | 持续积累，目标500+条 |

---

## 七、下一步开发计划

### 7.1 短期（本周）

| 任务 | 优先级 | 预计时间 |
|------|--------|----------|
| 解决Sentence-BERT环境 | 高 | 30分钟 |
| 向量化所有有价值笔记 | 高 | 10分钟 |
| 实现检索函数 | 高 | 20分钟 |
| 用Streamlit搭简单界面 | 中 | 30分钟 |

### 7.2 中期（本月）

| 任务 | 优先级 |
|------|--------|
| 积累到200+条有价值笔记 | 高 |
| 重训Word2Vec，提升效果 | 中 |
| 实现混合检索（语义+词元） | 高 |
| 添加高亮功能 | 中 |

### 7.3 长期（未来）

| 任务 | 优先级 |
|------|--------|
| 浏览器插件 | 低 |
| 联网检索 | 低 |
| BCI接口 | 实验性 |

---

## 八、快速开始

### 8.1 环境安装

```bash
# 创建环境
conda create -n cognitive python=3.10 -y
conda activate cognitive

# 安装依赖
pip install pandas numpy scikit-learn jieba gensim
pip install langchain langchain-community
pip install sentence-transformers  # 可选
```

### 8.2 运行流程

```bash
# 1. 导出数据
python src/preprocessor.py

# 2. 价值分类（需要API key）
python src/classifier.py

# 3. 分词
python src/tokenizer.py

# 4. 训练词向量
python src/vectorizer.py

# 5. 语义检索（待完善）
python src/retriever.py
```

### 8.3 配置文件示例

```python
# config.py
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
API_KEY = os.getenv("DEEPSEEK_API_KEY")
```

---

## 九、项目文件结构

```
Cognitive engine/
├── src/
│   ├── config.py           # 配置
│   ├── preprocessor.py     # 数据导出
│   ├── classifier.py       # 价值分类
│   ├── tokenizer.py        # 清洗+分词
│   ├── vectorizer.py       # Word2Vec训练
│   ├── sentence.py         # Sentence-BERT
│   └── retriever.py        # 检索（待开发）
├── data/
│   ├── results.csv         # 分类结果
│   ├── tokenized.json      # 分词结果
│   └── note_embeddings.npy # 文档向量
├── models/
│   └── word2vec.model      # 词向量模型
├── deepseek_chats/         # 原始对话txt
└── README.md
```

---

## 十、技术决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| 只保留[user]发言 | 是 | 引擎存个人思考，不存AI科普 |
| 分类准确率阈值 | 80% | 超过手动筛选一致性 |
| 向量维度 | 64 | 小数据用低维，避免过拟合 |
| 检索方式 | 混合（语义+词元） | 既要相关，又要精确 |
| 存储方案 | NPY + JSON | 文档<5000条时足够 |

---

## 十一、联系方式与资源

- 项目位置：`C:\Users\16673\Desktop\Cognitive engine`
- 原始数据：`./deepseek_chats/` 目录下的txt文件
- 模型文件：`./models/word2vec.model`
- 测试集：50条手工标注，86%准确率

---

**文档版本**: 1.0  
**更新日期**: 2026-05-16  
**作者**: toke648

---
