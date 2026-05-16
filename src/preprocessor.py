# \src\preprocessor.py
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import config

FILE_PATH = config.PROCESSED_DATA_DIR / "deepseek_chats"
OUTPUT_FILE = config.RAW_DATA_DIR / "label.csv"

data = []

files = enumerate(Path(FILE_PATH).glob("*.txt")) # 获取所有txt文件

for i, file in tqdm(files, desc="处理中"):
    text = file.read_text(encoding='utf-8')
    # 取第一行作为标题
    title = text.split('\n')[0].replace("标题:", "").strip()
    try:
        data.append({"id": f"{i:03d}", "title": title, "content": text.split('='*60)[1], "created_time": text.split('\n')[3]})  # 分割函数.split() text.split('='*60)[1]
    except Exception as e:
        print(e)    

df = pd.DataFrame(data)
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
