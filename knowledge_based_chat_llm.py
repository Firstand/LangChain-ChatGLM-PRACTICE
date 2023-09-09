from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
from chatllm import ChatLLM
from chinese_text_splitter import ChineseTextSplitter

import os
import config


def load_file(filepath):
    if filepath.lower().endswith(".md"):
        loader = UnstructuredFileLoader(filepath, mode="elements")
        docs = loader.load()
    elif filepath.lower().endswith(".pdf"):
        loader = UnstructuredFileLoader(filepath)
        text_splitter = ChineseTextSplitter(pdf=True)
        docs = loader.load_and_split(text_splitter)
    else:
        loader = UnstructuredFileLoader(filepath, mode="elements")
        text_splitter = ChineseTextSplitter(pdf=False)
        docs = loader.load_and_split(text_splitter=text_splitter)
    return docs


class KnowledgeBasedChatLLM:
    # 模型缓存位置
    model_cache_path: str = None
    # 大规模语言模型系列
    llm_model_type: str = None
    # 大规模语言模型名称
    llm_model_name: str = None
    # 嵌入式模型名称
    embedding_model_name: str = None
    # 大规模语言模型驱动
    llm_device: str = None
    # 可以获取的CUDA设备（即GPU）的数量
    num_gpus: int = None
    # 关键词数
    top_k: int = None
    # 温度参数
    temperature: float = None
    # 历史信息长度
    history_len: int = None
    # 大规模语言模型
    llm: ChatLLM = None
    # 嵌入式模型
    embeddings: Embeddings = None

    def __init__(self, llm_model_type: str = config.init_llm_type,
                 llm_model_name: str = config.llm_model_dict[config.init_llm_type][config.init_llm],
                 embedding_model_name: str = config.embedding_model_dict[config.init_embedding_model],
                 model_cache_path: str = config.model_cache_path,
                 llm_device: str = config.llm_device,
                 num_gpus: int = config.num_gpus):
        # 初始化参数
        self.model_cache_path = model_cache_path
        self.llm_model_type = llm_model_type
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        self.llm_device = llm_device
        self.num_gpus = num_gpus

    def init_model_config(self):
        # 初始化嵌入式模型
        # model_name 模型名称
        # cache_folder 模型的缓存位置
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name,
                                                cache_folder=os.path.join(self.model_cache_path,
                                                                          self.embedding_model_name))
        self.llm = ChatLLM()
        self.llm.model_type = self.llm_model_type
        self.llm.model_name_or_path = self.llm_model_name
        self.llm.load_llm(llm_device=self.llm_device, num_gpus=self.num_gpus)

    def init_knowledge_vector_store(self, filepath):
        # 初始化知识向量存储
        # 加载并解析文件，返回文件内容
        docs = load_file(filepath)
        # 将文件内容向量化（通过对输入的数据（如文档、图像等）进行向量化（或称为嵌入）处理，将高维数据转化为低维向量，从而能够更高效地进行相似性搜索）
        vector_store = FAISS.from_documents(docs, self.embeddings)
        # 保存
        vector_store.save_local('file_index')

    def get_knowledge_based_answer(self,
                                   query,
                                   web_content,
                                   top_k: int = 6,
                                   history_len: int = 3,
                                   temperature: float = 0.01,
                                   top_p: float = 0.1,
                                   history=None):
        if history is None:
            history = []
        self.top_k = top_k
        self.temperature = temperature
        self.history_len = history_len
        self.llm.top_p = top_p

        if web_content:
            prompt_template = f"""基于以下已知信息，简洁和专业的来回答用户的问题。
                                如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
                                已知网络检索内容：{web_content}""" + """
                                已知内容:
                                {context}
                                问题:
                                {question}"""
        else:
            prompt_template = """基于以下已知信息，请简洁并专业地回答用户的问题。
                如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"。不允许在答案中添加编造成分。另外，答案请使用中文。

                已知内容:
                {context}

                问题:
                {question}"""
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        # 历史消息最后一条消息
        self.llm.history = history[-self.history_len:] if self.history_len > 0 else []
        # 加载向量文件
        try:
            vector_store = FAISS.load_local('file_index', self.embeddings)
        except RuntimeError:
            return {'result': '请先上传本地知识库并点击【知识库文件向量化】后进行提问！'}

        # 创建一个基于预训练语言模型（LLM）的问答检索模型（RetrievalQA）
        # vector_store.as_retriever(search_kwargs={"k": self.top_k}) 返回与查询最相关的self.top_k个结果
        knowledge_chain = RetrievalQA.from_llm(llm=self.llm,
                                               retriever=vector_store.as_retriever(search_kwargs={"k": self.top_k}),
                                               prompt=prompt)

        resp = knowledge_chain({"query": query})
        return resp
