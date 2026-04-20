"""
TriviaQA rc 数据集下载脚本
下载后会保存到本地，可以直接上传到 AutoDL
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/autodl-tmp/trustworthy-rag/download"

from huggingface_hub import snapshot_download

# ======================================================
# 📦 下载 TriviaQA rc（带 Wikipedia 段落的版本）
# ======================================================
DATASET_ID = "trivia_qa"
LOCAL_DIR  = "D:/my_datasets/triviaqa_rc"

print("🚀 正在下载 TriviaQA rc 数据集，请稍候...")
snapshot_download(
    repo_id=DATASET_ID,
    repo_type="dataset",          # ← 关键：下载数据集而不是模型
    local_dir=LOCAL_DIR,
    resume_download=True,
    force_download=False,
)
print(f"✅ 数据集已下载到：{LOCAL_DIR}")