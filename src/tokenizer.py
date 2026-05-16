# tokenizer.py
import config
import pandas as pd
import jieba
import json
import re

# 读取数据
df = pd.read_csv(config.RAW_DATA_DIR / "results.csv", encoding='utf-8-sig')

data = []

for index, row in df.iterrows():
    # print(index, row)
    # print(row["pred_label"])
    content = row["content"].split('[user]: ')[1:]
    if row["pred_label"] == 1:
        data.append(content)

# 停用词
STOPWORDS = {
# 常见虚词
'的', '了', '是', '在', '和', '与', '或', '而', '且', '也', '都', '还',
'这', '那', '这些', '那些', '这个', '那个', '这么', '那么', '这样', '那样',

# 语气/连接词
'的话', '就是', '只是', '还有', '还有那', '可以说', '也就是',
'一种', '一个', '一些', '这种', '那种', '某种',
'那么', '然后', '接着', '于是', '因此', '所以', '但是', '然而', '不过',

# 数量/程度（保留“更”、“最”这类比较级？看情况）
'很多', '很多很多', '大量', '一些', '一点', '一点一点',

# 时间/方位（你可能想保留“之前”、“之后”用于理解你的思维时间线，可以不加）
'以前', '以后', '之前', '之后', '当时', '现在', '接下来',
}

# 分词器
class Tokenizer:
    def __init__(self):
        ...
    
    def clean(self, text):
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text) # 只保留中文、英文、数字，其他变空格
        text = re.sub(r'\s+', ' ', text) # 多个空格变成一个空格
        return text
    
    def dedup_lines(self, text):
        lines = text.split('\n')
        unique = []
        for line in lines:
            if line not in unique:
                unique.append(line.strip()) # strip() 去掉首尾空格
        text = ' '.join(unique) # 清洗后不需要保留换行，空格就够了
        return text
    
    def tokenize(self, text):
        words = jieba.lcut(text) # 分词 lcut: 最大切分模式
        return [w for w in words if len(w) >= 2] # 过滤掉长度小于等于2的词
    
    def remove_stopwords(self, text):
        return [w for w in text if w not in STOPWORDS] # 过滤掉停用词
    
    def forward(self, text):
        if isinstance(text, list): # 如果是列表
            content = '\n'.join(text)  # 如果是列表，先转字符串
        else:
            content = text  # 如果不是列表，直接赋值

        content = self.clean(content)
        content = self.dedup_lines(content)
        tokens = self.tokenize(content)
        tokens = self.remove_stopwords(tokens)
        return tokens

if __name__ == "__main__":
    # 将列表转换为字符串
    # data = ['\n'.join(sub_list) for sub_list in data]
    tokenizer = Tokenizer() # 初始化分词器

    all_results = []

    for sub_list in data: # 遍历列表
        if not sub_list: # 如果子列表为空，跳过
            continue

        res = '\n'.join(sub_list) # 提取列表中的文本, 每个子列表之间用换行符隔开

        tokens = tokenizer.forward(res) # 提取列表中的文本
        all_results.append(tokens) # 加入 all_results
        
    print(all_results)
    print(len(all_results))

    with open(config.RAW_DATA_DIR / "tokenized.json", 'w', encoding='utf-8') as f:
        # f.write('\n'.join(all_results)) # 保存分词后的文本
        json.dump(all_results, f, ensure_ascii=False, indent=2) # 保存分词后的文本 意思分别是 写入文件，编码，缩进2个空格









# # 数据清洗
# import re
# data = ['\n'.join(sub_list) for sub_list in data]
# text = data[0] # 提取列表中的文本
# # print(text)
# # 删除数字和字母[a-zA-Z0-9]
# # 删除特殊字符[^\u4e00-\u9fa5a-zA-Z0-9]
# text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text) # # 只保留中文、英文、数字，其他变空格
# text = re.sub(r'\s+', ' ', text) # 多个空格变成一个空格

# # print(text)

# # 数据去重
# lines = text.split('\n')
# unique = []
# for line in lines:
#     if line not in unique:
#         unique.append(line.strip()) # strip() 去掉首尾空格
#     text = '\n'.join(unique)
# # print(text)

# # 数据分词
# import jieba
# words = jieba.lcut(text)
# print(words)


# def clean_text(text: str) -> str:
#     text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
#     print(text)

# clean_text(text)



# # 然后再分词
# data = [' '.join(jieba.lcut(text)) for text in data]

# print(data)


