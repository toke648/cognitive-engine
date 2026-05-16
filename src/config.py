from pathlib import Path
from dotenv import load_dotenv
import os

ROOT_DIR = Path(__file__).parent.parent

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
STOPWORDS_FILE = ROOT_DIR / "data" / "stopwords-master"
MODEL_DIR = ROOT_DIR / "models"

# 加载环境变量
load_dotenv(r"C:\Users\16673\Desktop\OO\var\.env")
API_KEY = os.getenv("TONGYI_API_KEY")

if __name__ == '__main__':
    print(ROOT_DIR)
    print(RAW_DATA_DIR)
    print(PROCESSED_DATA_DIR)
    print("API_KEY:", API_KEY)
