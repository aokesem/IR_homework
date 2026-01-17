"""
生成模块
负责使用LLM生成答案及查询改写
支持: Local HF, Ollama, OpenAI-compatible API (DeepSeek, etc.)
"""
from typing import List, Dict, Optional
import json
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_core.documents import Document


class LLMGenerator:
    """LLM生成器"""
    
    # RAG 回答 Prompt
    PROMPT_TEMPLATE = """
你是一个专业的问答助手。请**仅基于以下参考资料**回答用户问题。

参考资料：
{context}

用户问题：{question}

要求：
1. 仅基于参考资料回答，不要编造信息
2. 如果资料中没有相关信息，请明确说明"参考资料中没有找到相关信息"
3. 回答要简洁明了

回答："""

    # 查询重写 Prompt
    REWRITE_PROMPT_TEMPLATE = """
根据以下对话历史，重写用户的最后一个问题，使其成为一个独立、完整的搜索查询。
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
        ollama_url: str = "http://localhost:11434",
        api_key: str = None,
        base_url: str = None
    ):
        """
        初始化生成器
        
        Args:
            provider: "huggingface", "ollama", or "openai_api"
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.dummy = dummy
        self.provider = provider
        self.ollama_url = ollama_url
        self.api_key = api_key
        self.base_url = base_url
        
        if dummy:
            return
            
        if self.provider == "openai_api":
            print(f"Using OpenAI-compatible API: {model_name} at {base_url}")
            return

        if self.provider == "ollama":
            print(f"Using Ollama provider: {model_name} at {ollama_url}")
            return
            
        # Local HF model loading...
        print(f"正在加载本地LLM模型: {model_name}")
        if load_in_4bit and device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            quantization_config = None
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            dtype=torch.float16 if device == "cuda" else torch.float32
        )
        if device == "cpu": self.model = self.model.to(device)
        self.model.eval()
        print(f"✓ 本地LLM模型加载完成")

    def build_context(self, documents: List[Document]) -> str:
        context_parts = []
        for i, doc in enumerate(documents, 1):
            if isinstance(doc, tuple): doc = doc[0]
            source = doc.metadata.get('file_name', '未知来源')
            content = doc.page_content.strip()
            context_parts.append(f"[资料{i}] 来源：{source}\n{content}")
        return "\n\n".join(context_parts)

    def generate(self, question: str, context_documents: List[Document], history: List[Dict] = None, custom_prompt: str = None) -> Dict:
        if self.dummy:
            return {"answer": f"Dummy answer for: {question}", "question": question, "num_sources": len(context_documents)}

        context = self.build_context(context_documents)
        
        if self.provider == "openai_api":
            return self._generate_openai_api(question, context, history, custom_prompt)
        elif self.provider == "ollama":
            return self._generate_ollama(question, context, history, custom_prompt)
        else:
            return self._generate_hf(question, context, history, custom_prompt)

    def rewrite_query(self, question: str, history: List[Dict]) -> str:
        """基于历史对话重写查询"""
        if not history or self.dummy: return question
        
        history_str = ""
        for msg in (history[-3:] if len(history) > 3 else history):
            role = "用户" if msg.get("role") == "user" else "助手"
            history_str += f"{role}: {msg.get('content')}\n"
                
        prompt = self.REWRITE_PROMPT_TEMPLATE.format(history_str=history_str, question=question)
        
        # 使用统一接口生成，但不带 RAG context
        res = self.generate(prompt, [], [], custom_prompt="{question}")
        return res['answer'].strip().strip('"').strip("'")

    def _generate_openai_api(self, question, context, history, custom_prompt) -> Dict:
        """调用 OpenAI 兼容 API (如 DeepSeek)"""
        prompt_template = custom_prompt or self.PROMPT_TEMPLATE
        # 如果是改写任务，context 为空，template 只是 {question}
        if "{context}" in prompt_template:
            content = prompt_template.format(context=context, question=question)
        else:
            content = prompt_template.replace("{question}", question)

        messages = []
        if history: messages.extend(history)
        messages.append({"role": "user", "content": content})

        try:
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_new_tokens
            }
            res = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=60)
            res.raise_for_status()
            answer = res.json()["choices"][0]["message"]["content"]
            return {"answer": answer.strip(), "question": question, "model": self.model_name}
        except Exception as e:
            return {"answer": f"API Error: {str(e)}", "question": question, "model": self.model_name}

    def _generate_ollama(self, question, context, history, custom_prompt) -> Dict:
        prompt_template = custom_prompt or self.PROMPT_TEMPLATE
        content = prompt_template.format(context=context, question=question) if "{context}" in prompt_template else prompt_template.replace("{question}", question)
        
        messages = []
        if history: messages.extend(history)
        messages.append({"role": "user", "content": content})
        
        try:
            res = requests.post(f"{self.ollama_url}/api/chat", json={
                "model": self.model_name, "messages": messages, "stream": False,
                "options": {"temperature": self.temperature, "num_predict": self.max_new_tokens}
            })
            answer = res.json().get("message", {}).get("content", "").strip()
            return {"answer": answer, "question": question, "model": f"{self.model_name} (Ollama)"}
        except Exception as e:
            return {"answer": f"Ollama Error: {str(e)}", "question": question, "model": self.model_name}

    def _generate_hf(self, question, context, history, custom_prompt) -> Dict:
        prompt_template = custom_prompt or self.PROMPT_TEMPLATE
        content = prompt_template.format(context=context, question=question) if "{context}" in prompt_template else prompt_template.replace("{question}", question)
        
        is_instruct = "Instruct" in self.model_name or "Chat" in self.model_name
        if is_instruct:
            messages = history.copy() if history else []
            messages.append({"role": "user", "content": content})
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = content # Simplified
            
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, temperature=self.temperature, do_sample=True)
        full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full[len(text):].strip() if not is_instruct else full.split("assistant\n")[-1].strip()
        return {"answer": answer or full, "question": question, "model": self.model_name}

    def get_info(self) -> Dict:
        return {"model_name": self.model_name, "provider": self.provider}
