"""
生成模块
负责使用LLM生成答案
"""
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.schema import Document


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
        temperature: float = 0.7
    ):
        """
        初始化生成器
        
        Args:
            model_name: 模型名称
            device: 设备（cuda/cpu）
            load_in_4bit: 是否使用4bit量化
            max_new_tokens: 最大生成token数
            temperature: 生成温度
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
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
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
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
        custom_prompt: str = None
    ) -> Dict[str, any]:
        """
        生成答案
        
        Args:
            question: 用户问题
            context_documents: 上下文文档列表
            custom_prompt: 自定义prompt模板（可选）
            
        Returns:
            包含答案和元信息的字典
        """
        # 构建上下文
        context = self.build_context(context_documents)
        
        # 构建完整prompt
        prompt_template = custom_prompt or self.PROMPT_TEMPLATE
        prompt = prompt_template.format(
            context=context,
            question=question
        )
        
        # 检测模型是否支持chat模板
        is_instruct_model = "Instruct" in self.model_name or "instruct" in self.model_name or "Chat" in self.model_name
        
        if is_instruct_model:
            # Instruct模型：使用chat模板
            try:
                messages = [
                    {"role": "user", "content": prompt}
                ]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                print(f"警告: chat模板应用失败，使用直接输入: {e}")
                text = prompt
        else:
            # 基础模型：直接使用prompt
            text = prompt
        
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
            "num_sources": len(context_documents),
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
