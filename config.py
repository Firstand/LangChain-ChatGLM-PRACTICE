import logging
import os
import datetime
import torch.backends.mps

# 获取当前日期，格式化日期为字符串，例如：2023-09-23
date_str = datetime.date.today().strftime("%Y-%m-%d")

# 构建日志文件名
log_filename = f"chat_llm_log_{date_str}.log"

# 日志存储路径
log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs/")
if not os.path.exists(log_path):
    os.mkdir(log_path)

# 日志格式
log_format = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO, filename=log_path + log_filename)
logger = logging.getLogger(__name__)

# 知识库默认存储路径
kb_root_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base")

# Large Language Model (LLM) 大规模语言模型字典
llm_model_dict = {
    "chatglm": {
        "ChatGLM2-6B": "../models_data/chatglm2-6b",
        "ChatGLM2-6B-int4": "../models_data/chatglm2-6b-int4",
        "ChatGLM-6B-int8": "THUDM/chatglm-6b-int8",
        "ChatGLM-6b-int4-qe": "THUDM/chatglm-6b-int4-qe"
    },
    "belle": {
        "BELLE-LLaMA-Local": "/pretrainmodel/belle",
    },
    "vicuna": {
        "Vicuna-Local": "/pretrainmodel/vicuna",
    }
}

# 嵌入模型字典
embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "ernie-medium": "nghuyong/ernie-3.0-medium-zh",
    "ernie-xbase": "nghuyong/ernie-3.0-xbase-zh",
    "text2vec-base-chinese": "GanymedeNil/text2vec-base-chinese/GanymedeNil_text2vec-base-chinese",
    'simbert-base-chinese': 'WangZeJun/simbert-base-chinese',
    'paraphrase-multilingual-MiniLM-L12-v2': "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
}

# 指定初始化Large Language Model (LLM) 大规模语言模型
init_llm = "ChatGLM2-6B-int4"

# 指定初始化Large Language Model (LLM) 大规模语言模型类型
init_llm_type = "chatglm"

# 指定初始化嵌入模型
init_embedding_model = "text2vec-base-chinese"

# os.path.dirname(__file__)获取python所在目录然后组合相对路径构成最终模型的缓存路径
# 模型缓存路径
model_cache_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models_data/')

# nltk 模型存储路径
nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")

# 驱动配置
# CUDA是NVIDIA开发的并行计算平台和应用程序接口模型，允许开发者使用NVIDIA的GPU进行计算。
# MPS是Apple（苹果）开发的Metal Performance Shaders，用于进行高性能图形和数据并行计算。
# 嵌入式语言模型驱动
embedding_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# 大规模语言模型驱动
llm_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
