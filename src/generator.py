"""
生成模块
负责使用LLM生成答案及查询改写
支持: Local HF, Ollama, OpenAI-compatible API (DeepSeek, etc.)
"""
from typing import List, Dict, Optional, Union
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
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.dummy = dummy
        self.provider = provider
        self.ollama_url = ollama_url
        self.api_key = api_key
        self.base_url = base_url
        
        if dummy: return
            
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

    def _extract_content(self, content) -> str:
        """
        从各种格式中提取纯文本内容 (增强版)
        """
        try:
            # 1. 如果是简单字符串，直接返回
            if isinstance(content, str):
                return content
            
            # 2. 如果是字典，尝试提取 text 或 content
            if isinstance(content, dict):
                if 'text' in content:
                    return self._extract_content(content['text'])
                if 'content' in content:
                    return self._extract_content(content['content'])
                # 最后的手段：转字符串
                return str(content)
            
            # 3. 如果是列表/元组，递归处理每个元素并拼接
            if isinstance(content, (list, tuple)):
                parts = []
                for item in content:
                    extracted = self._extract_content(item)
                    if extracted and extracted.strip():
                        parts.append(extracted)
                return "\n".join(parts)
            
            # 4. 其他类型
            return str(content)
            
        except Exception as e:
            print(f"Content extraction failed: {e}")
            return str(content)

    def build_context(self, documents: List[Document]) -> str:
        context_parts = []
        for i, doc in enumerate(documents, 1):
            if isinstance(doc, tuple): doc = doc[0]
            source = doc.metadata.get('file_name', '未知来源')
            content = doc.page_content.strip()
            context_parts.append(f"[资料{i}] 来源：{source}\n{content}")
        return "\n\n".join(context_parts)

    def generate(self, question: str, context_documents: List[Document], history: List = None, custom_prompt: str = None) -> Dict:
        if self.dummy:
            return {"answer": f"Dummy answer for: {question}", "question": question, "num_sources": len(context_documents)}

        context = self.build_context(context_documents)
        
        if self.provider == "openai_api":
            return self._generate_openai_api(question, context, history, custom_prompt)
        elif self.provider == "ollama":
            return self._generate_ollama(question, context, history, custom_prompt)
        else:
            return self._generate_hf(question, context, history, custom_prompt)

    def rewrite_query(self, question: str, history: List) -> str:
        """基于历史对话重写查询"""
        if not history or self.dummy: return question
        
        history_str = ""
        # 同样需要清洗 history
        clean_history = []
        if history and (isinstance(history[0], list) or isinstance(history[0], tuple)):
            for q, a in (history[-3:] if len(history) > 3 else history):
                clean_history.append((self._extract_content(q), self._extract_content(a)))
        
        for q, a in clean_history:
            history_str += f"用户: {q}\n助手: {a}\n"
                
        prompt = self.REWRITE_PROMPT_TEMPLATE.format(history_str=history_str, question=question)
        
        # 使用统一接口生成，但不带 RAG context
        res = self.generate(prompt, [], [], custom_prompt="{question}")
        return res['answer'].strip().strip('"').strip("'")

    def _generate_openai_api(self, question, context, history, custom_prompt) -> Dict:
        """调用 OpenAI 兼容 API (如 DeepSeek)"""
        prompt_template = custom_prompt or self.PROMPT_TEMPLATE
        if "{context}" in prompt_template:
            content = prompt_template.format(context=context, question=question)
        else:
            content = prompt_template.replace("{question}", question)

        messages = []
        # API通常只需要简单的文本history
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
        if history:
             if isinstance(history[0], list) or isinstance(history[0], tuple):
                for q, a in history:
                    messages.append({"role": "user", "content": self._extract_content(q)})
                    messages.append({"role": "assistant", "content": self._extract_content(a)})

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
        
        # 统一处理 history 格式转换 (Gradio List[List] -> OpenAI List[Dict])
        messages = []
        if history:
            if isinstance(history[0], list) or isinstance(history[0], tuple):
                # Gradio format: [[q, a], [q, a]]
                for q, a in history:
                    q_text = self._extract_content(q)
                    a_text = self._extract_content(a)
                    messages.append({"role": "user", "content": q_text})
                    messages.append({"role": "assistant", "content": a_text})
            elif isinstance(history[0], dict):
                # OpenAI format
                for msg in history:
                    messages.append({
                        "role": msg.get("role"),
                        "content": self._extract_content(msg.get("content"))
                    })
        
        # 添加当前问题
        if is_instruct:
            messages.append({"role": "user", "content": str(content)})
            
            # === DEBUG & SAFETY CHECK ===
            # 再次确保所有 content 都是 string，防止 _extract_content 漏网
            for i, m in enumerate(messages):
                if not isinstance(m['content'], str):
                    print(f"⚠️ Warning: Message {i} content is not str, forcing conversion: {type(m['content'])}")
                    messages[i]['content'] = str(m['content'])
            
            try:
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception as e:
                print(f"Chat template error: {e}. Fallback to raw.")
                text = ""
                for m in messages:
                    text += f"{m['role']}: {m['content']}\n"
                text += "assistant:"
        else:
            # Base model
            history_text = ""
            for msg in messages:
                role_name = "用户" if msg["role"] == "user" else "助手"
                history_text += f"{role_name}：{msg['content']}\n\n"
            text = history_text + content
            
        # 增加 max_length 防止 Prompt 被截断 (RAG 上下文通常较长)
        # Qwen2 支持长上下文，设置为 8192 较为安全
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=8192).to(self.device)
        
        input_length = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=self.max_new_tokens, 
                temperature=self.temperature, 
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 只解码新生成的 tokens
        generated_tokens = outputs[0][input_length:]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return {"answer": answer.strip(), "question": question, "num_sources": 0, "model": self.model_name}

    def get_info(self) -> Dict:
        return {"model_name": self.model_name, "provider": self.provider}