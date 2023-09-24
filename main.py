import gradio as gr
import config
from knowledge_based_chat_llm import KnowledgeBasedChatLLM

chat_llm = KnowledgeBasedChatLLM()


def init_model():
    chat_llm.init_model_config()


def reinitialize_model(history):
    return history + [[None, "暂不支持重新加载模型！"]]


def init_vector_store(file_obj, history):
    chat_llm.init_knowledge_vector_store(file_obj.name)
    return history + [[None, "知识库文件向量化完成！"]]


def predict(message, top_k, history_len, temperature, top_p, chat_history):
    resp = chat_llm.get_knowledge_based_answer(
        query=message,
        web_content='',
        top_k=top_k,
        history_len=history_len,
        temperature=temperature,
        top_p=top_p,
        history=chat_history
    )
    chat_history.append((message, resp['result']))
    return "", chat_history


def clear_predict():
    return '', None


if __name__ == '__main__':
    # Large Language Model (LLM) 大规模语言模型字典
    llm_model_dict = config.llm_model_dict
    # 嵌入模型
    embedding_model_dict = config.embedding_model_dict
    # 指定初始化Large Language Model (LLM) 大规模语言模型
    init_llm = config.init_llm
    # 指定初始化嵌入模型
    init_embedding_model = config.init_embedding_model

    # Large Language Model (LLM) 大规模语言模型列表
    llm_model_list = []
    for i in llm_model_dict:
        for j in llm_model_dict[i]:
            llm_model_list.append(j)

    # 根据配置初始化模型
    init_model()

    with gr.Blocks() as demo:
        gr.Markdown("""<h1><center>LangChain-ChatLLM-PRACTICE</center></h1>""")
        # 一行
        with gr.Row():
            # 占整行的多少
            with gr.Column(scale=1):
                # 折叠区
                with gr.Accordion("模型选择"):
                    # Dropdown 下拉框
                    large_language_model = gr.Dropdown(llm_model_list,
                                                       label="Large Language Model (LLM) 大规模语言模型",
                                                       value=init_llm,
                                                       interactive=True)
                    embedding_model_dict = gr.Dropdown(list(embedding_model_dict.keys()),
                                                       value=init_embedding_model,
                                                       label="Embedding model 嵌入模型",
                                                       interactive=True)
                    # 按钮
                    load_model_button = gr.Button("重新加载模型")
                # 折叠区
                with gr.Accordion("模型参数配置"):
                    # 滑块 最小值 minimum=1, 最大值 maximum=10, 默认值 value=6, 滑动一次涨几个数 step=1, 是否可互动/编辑 interactive=True
                    top_k = gr.Slider(minimum=1, maximum=10, value=6, step=1, label="矢量搜索关键词采样个数", interactive=True)
                    history_len = gr.Slider(minimum=0, maximum=5, value=3, step=1, label="历史消息个数", interactive=True)
                    temperature = gr.Slider(minimum=0, maximum=1, value=0.01, step=0.01, label="温度参数", interactive=True)
                    top_p = gr.Slider(minimum=0, maximum=1, value=0.9, step=0.1, label="核心采样阈值", interactive=True)
                # 文件上传
                file = gr.File(label="请上传本地知识库", file_types=['.txt', '.md', '.docx', '.pdf'])
                # 按钮
                init_vs = gr.Button("知识库文件向量化")
                # 单选框
                use_web = gr.Radio(["是", "否"], label="网站搜索", value="否", interactive=True)
            # 占整行的多少
            with gr.Column(scale=4):
                # 聊天框
                chat_bot = gr.Chatbot(label='ChatLLM')
                inp = gr.Textbox(placeholder="请输入您的问题！", label="提问")
                with gr.Row():
                    clear_history = gr.Button("🧹 清除历史对话")
                    send = gr.Button("🚀 发送")

        clear_history.click(fn=clear_predict, inputs=[], outputs=[inp, chat_bot])
        inp.submit(fn=predict, inputs=[inp, top_k, history_len, temperature, top_p, chat_bot], outputs=[inp, chat_bot])
        send.click(fn=predict, inputs=[inp, top_k, history_len, temperature, top_p, chat_bot], outputs=[inp, chat_bot])
        load_model_button.click(fn=reinitialize_model, inputs=[chat_bot], outputs=[chat_bot])
        init_vs.click(fn=init_vector_store, show_progress=True, inputs=[file, chat_bot], outputs=[chat_bot])
    demo.launch(inbrowser=True)
