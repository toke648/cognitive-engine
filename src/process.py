# \src\process.py
from sklearn.metrics import classification_report, confusion_matrix
from langchain_community.chat_models import ChatTongyi  # 或 ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm
import pandas as pd
import config


LOAD_FILE = config.RAW_DATA_DIR / "label_50_test.csv"
OUTPUT_FILE = config.RAW_DATA_DIR / "results.csv"

# 配置模型
api_key = config.API_KEY  # 你的key
model = ChatTongyi(api_key=api_key)  # 或 ChatDeepSeek

# 
value_prompt = ChatPromptTemplate.from_messages([
    ("system", """
你是一个严格的个人知识库管理助手。判断以下笔记是否有长期价值，值得存入个人认知引擎数据库中。
判断这条笔记的核心内容：是“你自己的思考过程”，还是“AI给你的信息/科普”？

规则：
- 输出 1：你自己的思考（包括：自我剖析、原创观点、学习试错、灵感画面、跨领域联想、能力声明、文学表达）
- 输出 0：AI给你的信息（包括：科普解释、信息查询、书单推荐、代码调试、幻想堆叠、宏大叙事）
- 输出 -1：介于两者之间，或内容太短（<20字）

只输出数字 1、0 或 -1。不要输出任何其他文字。

示例：
"我XXX总XXXX"（包含自我质疑） → 1
"XXX定义与应用解析" + AI的长篇科普回答 → 0 
"请系统性的讲解面向小白的XXX" → 0 （属于科普解释）
"有哪些书籍推荐？" + AI的书单 → 0 （书单推荐，属于信息查询）
"我发现了XXX那XXXX？" → 1 （思考过程）
（在复述哲学史，没有自己的思考） → 0
（知识查询结果，没有自己的分析） → 0
    """),
    ("user", "{content}")
])

chain = value_prompt | model | StrOutputParser() # 定义链，包含提示词、模型和输出解析器

# 加载标注数据
df = pd.read_csv(LOAD_FILE, encoding='utf-8-sig')[:50] # 只取前50条

# 预测
predictions = []
for content in tqdm(df["content"].tolist(), desc="AI分类中"):
    try:
        pred = chain.invoke({"content": content[:1500]})  # 限制输入长度，避免模型超时
        predictions.append(pred.strip())  # 去掉空格和换行
    except Exception as e:
        print(f"预测失败: {e}")
        predictions.append("-1")

df["pred_label"] = predictions  # 转成字符串，方便比较

# 保存结果
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"\n结果已保存到 {OUTPUT_FILE}")



# # 假设你的标签列叫 "label"，值是 1/0/-1
# df["true_label"] = df["label"].astype(str)  # 转成字符串，方便比较

# # 预测
# predictions = []
# for content in tqdm(df["content"].tolist(), desc="AI分类中"):
#     try:
#         pred = chain.invoke({"content": content[:1500]})  # 限制输入长度，避免模型超时
#         predictions.append(pred.strip())  # 去掉空格和换行
#     except Exception as e:
#         print(f"预测失败: {e}")
#         predictions.append("-1")

# df["pred_label"] = predictions  # 转成字符串，方便比较

# correct = (df["true_label"] == df["pred_label"]).sum()  # 计算正确预测的样本数
# total = len(df)
# print(f"\n准确率: {correct/total:.2%} ({correct}/{total})")

# print("\n=== 分类报告 ===")
# print(classification_report(df["true_label"], df["pred_label"])) # 分类报告，输出准确率、召回率、F1值等

# errors = df[df["true_label"] != df["pred_label"]]
# print(f"\n=== 错误案例 ({len(errors)}条) ===")
# for _, row in errors.iterrows():
#     print(f"\n[内容]: {row['content'][:80]}...")
#     print(f"[你的标注]: {row['true_label']} | [AI预测]: {row['pred_label']}")
#     print("-" * 40)

# # 保存结果
# df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
# print(f"\n结果已保存到 {OUTPUT_FILE}")




# import zai
# print(zai.__version__)

# from zai import ZhipuAiClient

# api_key = "fec3a7656a9947c3aacf333eff2f0480.OrnWK7pqcCd16RO8"

# # 初始化客户端
# client = ZhipuAiClient(api_key=api_key)

# # 创建聊天完成请求
# response = client.chat.completions.create(
#     model="glm-4.7-flash",
#     messages=[
#         {"role": "system", "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"},
#         {"role": "user", "content": "你好，请介绍一下自己"}
#     ],
#     max_tokens=4096,
#     temperature=0.7
# )

# # 获取回复
# print(response.choices[0].message.content)







# import requests

# url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

# api_key = "fec3a7656a9947c3aacf333eff2f0480.OrnWK7pqcCd16RO8"

# payload = {
#     "model": "glm-4.7-flash",
#     "messages": [
#         {
#             "role": "system",
#             "content": "你是一个有用的AI助手。"
#         },
#         {
#             "role": "user",
#             "content": "请介绍一下人工智能的发展历程。"
#         }
#     ],
#     "stream": False,
#     "temperature": 1
# }
# headers = {
#     "Authorization": f"Bearer {api_key}",
#     "Content-Type": "application/json"
# }

# response = requests.post(url, json=payload, headers=headers)

# print(response.text)