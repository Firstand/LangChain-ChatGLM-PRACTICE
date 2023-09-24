import os
from abc import ABC
from typing import Optional, Dict, List, Any

import torch
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from torch import nn
from transformers import AutoTokenizer, AutoModel

import config

device = config.llm_device
device_id = "0"
cuda_device = f"{device}:{device_id}" if device_id else device


def torch_gc():
    # 检查当前系统是否支持CUDA（NVIDIA GPU上的计算能力）
    if torch.cuda.is_available():
        # 如果系统支持CUDA，这行代码将PyTorch的运行设备切换到CUDA_DEVICE上
        with torch.cuda.device(cuda_device):
            # 清空GPU缓存，释放GPU上的临时内存
            torch.cuda.empty_cache()
            # 收集GPU间进程通信（Inter-Process Communication，IPC）的垃圾，以帮助释放GPU资源。
            # IPC通常用于多进程训练深度学习模型时，以确保各个进程之间能够正确地共享GPU资源。
            torch.cuda.ipc_collect()


def auto_configure_device_map(cuda_nums: int, model):
    device_map = {}
    used_per_gpu = [0] * cuda_nums

    for name, module in model.named_modules():
        # 在这里，您可以根据模块的名称或类型来决定如何分配GPU
        # 以下示例假设所有的Transformer层都应分配到不同的GPU
        if 'transformer' in name and 'layers' in name:
            gpu_target = used_per_gpu.index(min(used_per_gpu))
            device_map[name] = gpu_target
            used_per_gpu[gpu_target] += 1

    return device_map


class ChatLLM(LLM, ABC):
    max_token: int = 10000
    temperature: float = 0.1
    top_p = 0.9
    history = []
    model_type: str = "chatglm"
    model_name_or_path: str = config.init_llm
    tokenizer: object = None
    model: nn.Module = None

    # @property 是 Python 中的一个装饰器，它允许您将一个方法（函数）转换为属性，以便在访问时像访问普通属性一样使用，而不需要调用方法的括号。
    # 在这种情况下，_llm_type 被定义为一个属性，当您访问 _llm_type 时，实际上调用了 _llm_type 方法，并返回其结果，即字符串 "ChatLLM"。
    # obj = YourClass()
    # llm_type = obj._llm_type  # 访问属性，无需括号
    # print(llm_type)  # 输出 "ChatLLM"
    @property
    def _llm_type(self):
        return "ChatLLM"

    def _call(self, prompt: str,
              stop: Optional[List[str]] = None,
              un_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any) -> str:
        if self.model_type == 'chatglm':
            response, history = self.model.chat(self.tokenizer,
                                                prompt,
                                                history=self.history,
                                                max_length=self.max_token if self.max_token else 2048,
                                                top_p=self.top_p if self.top_p else 0.7,
                                                temperature=self.temperature if self.temperature else 0.95)
            torch_gc()
            if stop is not None:
                response = enforce_stop_tokens(response, stop)
            self.history = self.history + [[history, response]]
        else:
            response = '暂不支持该模型！'
        return response

    def load_llm(self, llm_device=device, device_map: Optional[Dict[str, int]] = None, **kwargs):
        # trust_remote_code 是否信任远程代码
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path,
                                                       trust_remote_code=True,
                                                       calendar=os.path.join(config.model_cache_path,
                                                                             self.model_name_or_path))
        # 检查cuda是否可用
        if torch.cuda.is_available() and llm_device.lower().startswith("cuda"):
            self.model = AutoModel.from_pretrained(self.model_name_or_path,
                                                   trust_remote_code=True,
                                                   calendar=os.path.join(config.model_cache_path,
                                                                         self.model_name_or_path)).half().cuda()
            # cuda设备数量
            cuda_nums = torch.cuda.device_count()
            if device_map is None:
                device_map = auto_configure_device_map(cuda_nums, self.model)

            if cuda_nums > 1:
                from accelerate import dispatch_model
                self.model = dispatch_model(self.model, device_map=device_map)
        else:
            self.model = AutoModel.from_pretrained(self.model_name_or_path,
                                                   trust_remote_code=True).float().to(llm_device)

        if isinstance(self.model, nn.Module):
            # 切换到评估模式 告诉模型不再进行训练，因此不会进行反向传播和参数更新。这是为了确保在评估模型时不会意外地修改权重。
            self.model = self.model.eval()
