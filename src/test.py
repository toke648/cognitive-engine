import pandas as pd
import config

LOAD_FILE = config.RAW_DATA_DIR / "label.csv"
OUTPUT_FILE = config.RAW_DATA_DIR / "label_50_test.csv"

df = pd.read_csv(LOAD_FILE, encoding='utf-8-sig')
# 加载随机数据做标注
sample = df.sample(n=50, random_state=42)  # random_state固定，可复现
sample.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
