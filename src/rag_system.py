"""
RAG系统主流程
整合检索和生成模块
"""
import yaml
from typing import List, Dict, Tuple
from pathlib import Path

from document_processor import DocumentProcessor
from retriever import VectorRetriever
from generator import LLMGenerator
from langchain_core.documents import Document


class RAGSystem:
    """RAG问答系统"""
    
    def __init__(self, config_path: str = "../config.yaml", dummy_mode: bool = False):
        """
        初始化RAG系统
        
        Args:
            config_path: 配置文件路径
            dummy_mode: 是否开启虚拟模式
        """
        self.dummy_mode = dummy_mode
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        print("=" * 50)
        print("初始化RAG系统")
        print("=" * 50)
        
        # 初始化各模块
        self._init_document_processor()
        self._init_retriever()
        self._init_generator()
        
        print("\n✓ RAG系统初始化完成\n")
    
    def _init_document_processor(self):
        """初始化文档处理器"""
        doc_config = self.config['document']
        self.doc_processor = DocumentProcessor(
            chunk_size=doc_config['chunk_size'],
            chunk_overlap=doc_config['chunk_overlap'],
            processed_dir=self.config['paths'].get('processed_docs')
        )
    
    def _init_retriever(self):
        """初始化检索器"""
        model_config = self.config['models']['embedding']
        paths_config = self.config['paths']
        
        self.retriever = VectorRetriever(
            embedding_model_name=model_config['name'],
            device=model_config['device'],
            vector_store_path=paths_config['vector_store'],
            dummy=self.dummy_mode
        )
    
    def _init_generator(self):
        """初始化生成器"""
        llm_config = self.config['models']['llm']
        
        self.generator = LLMGenerator(
            model_name=llm_config['name'],
            device=llm_config['device'],
            load_in_4bit=llm_config.get('load_in_4bit', True),
            max_new_tokens=llm_config['max_new_tokens'],
            temperature=llm_config['temperature'],
            dummy=self.dummy_mode
        )
    
    def build_knowledge_base(self, document_dir: str = None) -> None:
        """
        构建知识库
        
        Args:
            document_dir: 文档目录（默认使用配置中的路径）
        """
        if document_dir is None:
            document_dir = self.config['paths']['raw_docs']
        
        print(f"\n{'='*50}")
        print(f"构建知识库: {document_dir}")
        print(f"{'='*50}\n")
        
        # 1. 加载和处理文档
        chunks = self.doc_processor.process_directory(document_dir)
        
        if not chunks:
            print("警告: 没有文档可以处理")
            return
        
        # 2. 构建向量索引
        self.retriever.build_index(chunks)
        
        # 3. 保存向量数据库
        self.retriever.save()
        
        print(f"\n✓ 知识库构建完成 ({len(chunks)} 个文档块)")
    
    def load_knowledge_base(self) -> None:
        """加载已有的知识库"""
        vector_store_path = self.config['paths']['vector_store']
        
        print(f"\n正在加载知识库: {vector_store_path}")
        self.retriever.load(vector_store_path)
        print("✓ 知识库加载完成")
    
    def add_documents(self, file_paths: List[str]) -> None:
        """
        向现有知识库添加文档
        
        Args:
            file_paths: 文件路径列表
        """
        # 加载和处理文档
        documents = self.doc_processor.load_documents(file_paths)
        chunks = self.doc_processor.split_documents(documents)
        
        # 添加到向量索引
        self.retriever.add_documents(chunks)
        
        # 保存更新后的索引
        self.retriever.save()
        
        print(f"✓ 成功添加 {len(chunks)} 个文档块")
    
    def query(
        self,
        question: str,
        top_k: int = None,
        history: List[List[str]] = None,
        custom_prompt: str = None,
        return_sources: bool = True
    ) -> Dict:
        """
        问答查询
        
        Args:
            question: 用户问题
            top_k: 检索top k个文档（默认使用配置）
            history: 对话历史 [[user_msg, bot_msg], ...]
            custom_prompt: 自定义 Prompt 模板
            return_sources: 是否返回来源文档
            
        Returns:
            包含答案和来源的字典
        """
        if top_k is None:
            top_k = self.config['retrieval']['top_k']
        
        # 1. 检索相关文档
        retrieved_docs = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            score_threshold=self.config['retrieval'].get('score_threshold', 0.0)
        )
        
        if not retrieved_docs:
            return {
                "question": question,
                "answer": "抱歉，在知识库中没有找到相关信息。",
                "sources": [],
                "num_sources": 0
            }
        
        # 2. 生成答案
        result = self.generator.generate(
            question=question,
            context_documents=retrieved_docs,
            history=history,
            custom_prompt=custom_prompt
        )
        
        # 3. 整理返回结果
        response = {
            "question": question,
            "answer": result['answer'],
            "num_sources": len(retrieved_docs)
        }
        
        # 添加来源信息
        if return_sources:
            sources = []
            for doc, score in retrieved_docs:
                sources.append({
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "file_name": doc.metadata.get('file_name', '未知'),
                    "similarity": f"{score:.4f}"
                })
            response['sources'] = sources
        
        return response
    
    def get_knowledge_base_sources(self) -> List[Dict]:
        """获取当前知识库中的文档列表及统计信息"""
        return self.retriever.get_all_sources()
    
    def get_system_info(self) -> Dict:
        """
        获取系统信息
        
        Returns:
            系统信息字典
        """
        return {
            "document_processor": {
                "chunk_size": self.doc_processor.chunk_size,
                "chunk_overlap": self.doc_processor.chunk_overlap
            },
            "retriever": self.retriever.get_stats(),
            "generator": self.generator.get_info()
        }


if __name__ == "__main__":
    # 测试代码
    import sys
    
    # 初始化系统
    rag = RAGSystem(config_path="../config.yaml")
    
    # 检查是否需要构建知识库
    import os
    vector_store_path = rag.config['paths']['vector_store']
    
    if not os.path.exists(vector_store_path):
        print("\n首次运行，构建知识库...")
        rag.build_knowledge_base()
    else:
        print("\n加载已有知识库...")
        rag.load_knowledge_base()
    
    # 测试问答
    test_questions = [
        "什么是RAG?",
        "RAG的主要优势是什么?",
    ]
    
    print("\n" + "="*50)
    print("测试问答")
    print("="*50 + "\n")
    
    for q in test_questions:
        print(f"\n问题: {q}")
        result = rag.query(q)
        print(f"答案: {result['answer']}")
        print(f"使用来源数: {result['num_sources']}")
