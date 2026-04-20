"""
Hugging Face 模型下载与加载示例（AutoDL适配版）
"""

import os

# ======================================================
# 🧩 1️⃣ AutoDL环境配置：数据盘缓存 + 镜像加速
# ======================================================
os.environ["HF_HOME"] = "/root/autodl-tmp/cache/"        # ✅ 缓存到数据盘
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"      # ✅ 清华镜像加速

# 可选：创建缓存目录（确保存在）
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

# ======================================================
# 🤖 2️⃣ 模型信息
# ======================================================
MODEL_ID = "cross-encoder/nli-deberta-v3-small"
LOCAL_DIR = "/root/autodl-tmp/trustworthy-rag/models/nli-deberta-v3-small"

# ======================================================
# 💾 3️⃣ 下载模型
# ======================================================
print(f"🚀 缓存目录设置为: {os.environ['HF_HOME']}")
print("🚀 正在下载模型，请稍候...")
snapshot_download(
    repo_id=MODEL_ID,
    local_dir=LOCAL_DIR,
    resume_download=True,
    force_download=False,
)
print(f"✅ 模型已下载到：{LOCAL_DIR}")