"""
生成模块
负责使用LLM生成答案及查询改写
"""
from typing import List, Dict
import json
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_core.documents import Document


class LLMGenerator:
    """LLM生成器"""
    
    # RAG 回答 Prompt
    PROMPT_TEMPLATE = """你是一个专业的问答助手。请**仅基于以下参考资料**回答用户问题。

参考资料：
{context}

用户问题：{question}

要求：
1. 仅基于参考资料回答，不要编造信息
2. 如果资料中没有相关信息，请明确说明"参考资料中没有找到相关信息"
3. 回答要简洁明了

回答："""

    # 查询重写 Prompt
    REWRITE_PROMPT_TEMPLATE = """根据以下对话历史，重写用户的最后一个问题，使其成为一个独立、完整的搜索查询。
如果问题包含指代词（如"它"、"这个"），请将其替换为具体的指代对象。
不要回答问题，只输出重写后的查询。

对话历史：
{history_str}

原始问题：{question}

重写后的搜索查询："""
    
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
        
        # 配置量化
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
        """构建上下文字符串"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            if isinstance(doc, tuple): doc = doc[0]
            source = doc.metadata.get('file_name', '未知来源')
            content = doc.page_content.strip()
            context_parts.append(f"[资料{i}] 来源：{source}\n{content}")
        return "\n\n".join(context_parts)
    
    def rewrite_query(self, question: str, history: List[List[str]]) -> str:
        """
        基于历史对话重写用户问题 (Query Rewriting)
        """
        if not history or self.dummy:
            return question
            
        # 提取最近的3轮对话作为上下文
        recent_history = history[-3:] if len(history) > 3 else history
        
        # 格式化历史对话字符串
        history_str = ""
        if isinstance(recent_history[0], dict):
             for msg in recent_history:
                role = "用户" if msg.get("role") == "user" else "助手"
                history_str += f"{role}: {msg.get('content')}\n"
        else:
            for q, a in recent_history:
                history_str += f"用户: {q}\n助手: {a}\n"
                
        prompt = self.REWRITE_PROMPT_TEMPLATE.format(
            history_str=history_str,
            question=question
        )
        
        # 调用模型生成 (不带RAG上下文)
        # 这里为了简单，复用 generate 方法但传入空context
        # 注意：这里我们使用一次性的推理调用，不带 Chat Template 的 history，因为 Prompt 已经包含了 history
        
        try:
            if self.provider == "ollama":
                # Ollama 直接调用
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3} # 改写任务温度要低
                }
                res = requests.post(f"{self.ollama_url}/api/generate", json=payload)
                if res.status_code == 200:
                    rewritten = res.json().get("response", "").strip()
                    # 清洗可能带有的引号
                    return rewritten.strip('"').strip("'")
            else:
                # HF 本地调用
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=64,
                        temperature=0.3,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # 截取 Prompt 之后的内容
                rewritten = full[len(prompt):].strip()
                return rewritten.strip('"').strip("'")
                
        except Exception as e:
            print(f"⚠️ 查询重写失败: {e}")
            return question
            
        return question

    def generate(
        self,
        question: str,
        context_documents: List[Document],
        history: List[List[str]] = None,
        custom_prompt: str = None
    ) -> Dict[str, any]:
        """生成答案"""
        if self.dummy:
            return {
                "answer": f"【虚拟回答】这是一个模拟生成的答案。当前处于开发模式，已跳过真实的模型推理过程。您提问的问题是：\"{question}\"",
                "question": question,
                "num_sources": len(context_documents),
                "model": f"{self.model_name} (Dummy)"
            }

        # 构建上下文
        context = self.build_context(context_documents)
        
        if self.provider == "ollama":
            return self._generate_ollama(question, context, history, custom_prompt)
        else:
            return self._generate_hf(question, context, history, custom_prompt)

    def _generate_ollama(self, question, context, history, custom_prompt) -> Dict:
        prompt_template = custom_prompt or self.PROMPT_TEMPLATE
        current_prompt = prompt_template.format(context=context, question=question)
        
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
            res = requests.post(f"{self.ollama_url}/api/chat", json=payload)
            res.raise_for_status()
            answer = res.json().get("message", {}).get("content", "").strip()
            
            return {"answer": answer, "question": question, "num_sources": 0, "model": f"{self.model_name} (Ollama)"}
        except Exception as e:
            return {"answer": f"Ollama Error: {str(e)}", "question": question, "num_sources": 0, "model": self.model_name}

    def _generate_hf(self, question, context, history, custom_prompt) -> Dict:
        prompt_template = custom_prompt or self.PROMPT_TEMPLATE
        current_prompt = prompt_template.format(context=context, question=question)
        is_instruct = "Instruct" in self.model_name or "Chat" in self.model_name
        
        if is_instruct:
            try:
                messages = []
                if history:
                    if isinstance(history[0], dict):
                        messages.extend(history)
                    else:
                        for q, a in history:
                            messages.append({"role": "user", "content": q})
                            messages.append({"role": "assistant", "content": a})
                messages.append({"role": "user", "content": current_prompt})
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except:
                text = current_prompt
        else:
            history_text = ""
            if history:
                if isinstance(history[0], dict):
                    for msg in history:
                        role = "用户" if msg["role"]=="user" else "助手"
                        history_text += f"{role}：{msg['content']}\n\n"
                else:
                    for q, a in history:
                        history_text += f"用户问题：{q}\n回答：{a}\n\n"
            text = history_text + current_prompt
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, temperature=self.temperature, 
                do_sample=True, top_p=0.9, pad_token_id=self.tokenizer.eos_token_id
            )
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if is_instruct and "assistant\n" in full_response:
            answer = full_response.split("assistant\n")[-1].strip()
        else:
            answer = full_response[len(text):].strip()
        if not answer: answer = full_response
        
        return {"answer": answer, "question": question, "num_sources": 0, "model": self.model_name}
    
    def get_info(self) -> Dict:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_new_tokens": self.max_new_tokens,
            "provider": self.provider
        }


if __name__ == "__main__":
    # Test stub
    pass