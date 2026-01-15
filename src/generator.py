"""
生成模块
负责使用LLM生成答案
"""
from typing import List, Dict
import json
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_core.documents import Document


class LLMGenerator:
    """LLM生成器"""
    
    # Prompt模板
    PROMPT_TEMPLATE = """你是一个专业的问答助手。请**仅基于以下参考资料**回答用户问题。

参考资料：
{context}

用户问题：{question}

要求：
1. 仅基于参考资料回答，不要编造信息
2. 如果资料中没有相关信息，请明确说明"参考资料中没有找到相关信息"
3. 回答要简洁明了

回答："""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B-Instruct",
        device: str = "cuda",
        load_in_4bit: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        dummy: bool = False,
        provider: str = "huggingface",
        ollama_url: str = "http://localhost:11434"
    ):
        """
        初始化生成器
        
        Args:
            model_name: 模型名称 (HF path or Ollama model name)
            device: 设备（cuda/cpu）
            load_in_4bit: 是否使用4bit量化
            max_new_tokens: 最大生成token数
            temperature: 生成温度
            dummy: 是否开启虚拟模式（不加载模型）
            provider: 模型提供商 ("huggingface" or "ollama")
            ollama_url: Ollama API URL
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.dummy = dummy
        self.provider = provider
        self.ollama_url = ollama_url
        
        if dummy:
            print(f"⚠️ 虚拟模式：跳过加载LLM模型 {model_name} ({provider})")
            self.tokenizer = None
            self.model = None
            return
            
        if self.provider == "ollama":
            print(f"Using Ollama provider: {model_name} at {ollama_url}")
            # Ollama 不需要加载本地模型权重
            self.tokenizer = None
            self.model = None
            # 测试连接
            try:
                requests.get(f"{self.ollama_url}/api/tags")
                print(f"✓ Ollama 服务连接成功")
            except Exception as e:
                print(f"⚠️ 警告: 无法连接到 Ollama 服务: {e}")
            return
            
        print(f"正在加载LLM模型: {model_name}")
        
        # 配置量化（节省显存）
        if load_in_4bit and device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            quantization_config = None
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        # CPU模式下手动移动模型
        if device == "cpu":
            self.model = self.model.to(device)
        
        self.model.eval()
        
        print(f"✓ LLM模型加载完成 (设备: {device})")
    
    def build_context(self, documents: List[Document]) -> str:
        """
        构建上下文字符串
        
        Args:
            documents: 检索到的文档列表（可能包含分数）
            
        Returns:
            上下文字符串
        """
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            # 如果是tuple（包含分数），取第一个元素
            if isinstance(doc, tuple):
                doc = doc[0]
            
            source = doc.metadata.get('file_name', '未知来源')
            content = doc.page_content.strip()
            
            context_parts.append(f"[资料{i}] 来源：{source}\n{content}")
        
        return "\n\n".join(context_parts)
    
    def generate(
        self,
        question: str,
        context_documents: List[Document],
        history: List[List[str]] = None,
        custom_prompt: str = None
    ) -> Dict[str, any]:
        """
        生成答案
        
        Args:
            question: 用户问题
            context_documents: 上下文文档列表
            history: 对话历史 [[user_msg, bot_msg], ...]
            custom_prompt: 自定义prompt模板（可选）
            
        Returns:
            包含答案和元信息的字典
        """
        if self.dummy:
            return {
                "answer": f"【虚拟回答】这是一个模拟生成的答案。当前处于开发模式，已跳过真实的模型推理过程。您提问的问题是：\"{question}\"",
                "question": question,
                "num_sources": len(context_documents),
                "model": f"{self.model_name} (Dummy)"
            }

        # 构建上下文
        context = self.build_context(context_documents)
        
        # Dispatch based on provider
        if self.provider == "ollama":
            return self._generate_ollama(question, context, history, custom_prompt)
        else:
            return self._generate_hf(question, context, history, custom_prompt)

    def _generate_ollama(
        self,
        question: str,
        context: str,
        history: List[List[str]],
        custom_prompt: str
    ) -> Dict[str, any]:
        """使用 Ollama API 生成"""
        # 构建当前问题的完整prompt
        prompt_template = custom_prompt or self.PROMPT_TEMPLATE
        current_prompt = prompt_template.format(
            context=context,
            question=question
        )
        
        # 构建 messages
        messages = []
        if history:
            if isinstance(history[0], dict):
                messages.extend(history)
            else:
                 for old_q, old_a in history:
                    messages.append({"role": "user", "content": old_q})
                    messages.append({"role": "assistant", "content": old_a})
        
        messages.append({"role": "user", "content": current_prompt})
        
        try:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_new_tokens
                }
            }
            
            response = requests.post(f"{self.ollama_url}/api/chat", json=payload)
            response.raise_for_status()
            
            result_json = response.json()
            answer = result_json.get("message", {}).get("content", "")
            
            return {
                "answer": answer.strip(),
                "question": question,
                "num_sources": 0, # 这里没有单独计数，或可以传进来
                "model": f"{self.model_name} (Ollama)"
            }
            
        except Exception as e:
            return {
                "answer": f"Ollama 调用失败: {str(e)}",
                "question": question,
                "num_sources": 0,
                "model": self.model_name
            }

    def _generate_hf(
        self,
        question: str,
        context: str,
        history: List[List[str]],
        custom_prompt: str
    ) -> Dict[str, any]:
        """使用本地 HF 模型生成"""
        
        # 构建当前问题的完整prompt（带参考资料）
        prompt_template = custom_prompt or self.PROMPT_TEMPLATE
        current_prompt = prompt_template.format(
            context=context,
            question=question
        )
        
        # 检测模型是否支持chat模板
        is_instruct_model = "Instruct" in self.model_name or "instruct" in self.model_name or "Chat" in self.model_name
        
        if is_instruct_model:
            # Instruct模型：构建消息列表
            try:
                messages = []
                # 添加历史记录
                if history and len(history) > 0:
                    # 如果 history 已经是 OpenAI 格式 (list of dicts)，直接添加
                    if isinstance(history[0], dict):
                        messages.extend(history)
                    else:
                        # 兼容旧的 List[List[str]] 格式
                        for old_question, old_answer in history:
                            messages.append({"role": "user", "content": old_question})
                            messages.append({"role": "assistant", "content": old_answer})
                
                # 添加当前带上下文的问题
                messages.append({"role": "user", "content": current_prompt})
                
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                print(f"警告: chat模板应用失败，使用直接输入: {e}")
                text = current_prompt
        else:
            # 基础模型：将历史拼接在最前面
            history_text = ""
            if history:
                if isinstance(history[0], dict):
                    for msg in history:
                        role_name = "用户" if msg["role"] == "user" else "助手"
                        history_text += f"{role_name}：{msg['content']}\n\n"
                else:
                    for old_question, old_answer in history:
                        history_text += f"用户问题：{old_question}\n回答：{old_answer}\n\n"
            
            text = history_text + current_prompt
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048  # 限制输入长度
        ).to(self.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取回答部分
        if is_instruct_model:
            # Instruct模型：提取assistant部分
            if "assistant\n" in full_response:
                answer = full_response.split("assistant\n")[-1].strip()
            else:
                # 去掉输入prompt部分
                answer = full_response[len(text):].strip()
        else:
            # 基础模型：去掉输入prompt部分
            answer = full_response[len(text):].strip()
            
        # 如果答案为空或太短，返回完整响应
        if not answer or len(answer) < 10:
            answer = full_response.strip()
        
        return {
            "answer": answer,
            "question": question,
            "num_sources": 0,
            "model": self.model_name
        }
    
    def get_info(self) -> Dict:
        """
        获取生成器信息
        
        Returns:
            信息字典
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "provider": self.provider
        }


if __name__ == "__main__":
    # 测试代码
    from langchain.schema import Document
    
    # 测试文档
    test_docs = [
        Document(
            page_content="RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术，先检索相关文档，再基于文档生成答案。",
            metadata={"file_name": "test.txt"}
        )
    ]
    
    # 初始化生成器（测试时用CPU）
    generator = LLMGenerator(
        model_name="Qwen/Qwen3-0.6B-Instruct",
        device="cpu",
        load_in_4bit=False
    )
    
    # 生成答案
    result = generator.generate(
        question="什么是RAG?",
        context_documents=test_docs
    )
    
    print(f"\n问题: {result['question']}")
    print(f"\n答案: {result['answer']}")
    print(f"\n使用模型: {result['model']}")
