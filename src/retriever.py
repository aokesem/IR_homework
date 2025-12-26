"""
检索模块
负责文档向量化和相似度检索
"""
import os
from typing import List, Dict, Tuple
import numpy as np

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document


class VectorRetriever:
    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-small-zh-v1.5",
        device: str = "cuda",
        vector_store_path: str = None
    ):
        """
        Args:
            embedding_model_name: Embedding模型名称
            device: 设备（cuda/cpu）
            vector_store_path: 向量数据库保存路径
        """
        self.embedding_model_name = embedding_model_name
        self.device = device
        self.vector_store_path = vector_store_path
        self.vector_store = None
        
        # 初始化embedding模型
        print(f"正在加载Embedding模型: {embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}  # 归一化向量
        )
        print("✓ Embedding模型加载完成")
    
    def build_index(self, documents: List[Document]) -> None:
        """
        构建向量索引
        
        Args:
            documents: 文档块列表
        """
        if not documents:
            raise ValueError("文档列表为空，无法构建索引")
        
        print(f"正在构建向量索引 ({len(documents)} 个文档块)...")
        
        # 使用FAISS构建向量数据库
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        print("✓ 向量索引构建完成")
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Args:
            documents: 文档块列表
        """
        if self.vector_store is None:
            self.build_index(documents)
        else:
            print(f"正在添加 {len(documents)} 个文档块到索引...")
            self.vector_store.add_documents(documents)
            print("✓ 文档添加完成")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Tuple[Document, float]]:
        """
        Args:
            query: 查询文本
            top_k: 返回top k个结果
            score_threshold: 分数阈值（过滤低相关度结果）
            
        Returns:
            (文档, 相似度分数) 列表
        """
        if self.vector_store is None:
            raise ValueError("向量索引尚未构建，请先调用build_index()")
        
        # 相似度搜索（带分数）
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=top_k
        )
        
        # 过滤低分结果（FAISS返回的是距离，越小越相似）
        # 转换为相似度分数（0-1之间，1表示完全相同）
        filtered_results = []
        for doc, distance in results:
            # L2距离转换为相似度
            similarity = 1 / (1 + distance)
            
            if similarity >= score_threshold:
                filtered_results.append((doc, similarity))
        
        return filtered_results
    
    def save(self, path: str = None) -> None:
        """
        Args:
            path: 保存路径（可选，默认使用初始化时的路径）
        """
        if self.vector_store is None:
            raise ValueError("向量索引尚未构建，无法保存")
        
        save_path = path or self.vector_store_path
        
        if save_path is None:
            raise ValueError("未指定保存路径")
        
        print(f"正在保存向量数据库到: {save_path}")
        self.vector_store.save_local(save_path)
        print("✓ 向量数据库保存完成")
    
    def load(self, path: str = None) -> None:
        """
        Args:
            path: 加载路径（可选，默认使用初始化时的路径）
        """
        load_path = path or self.vector_store_path
        
        if load_path is None:
            raise ValueError("未指定加载路径")
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"向量数据库不存在: {load_path}")
        
        print(f"正在加载向量数据库: {load_path}")
        self.vector_store = FAISS.load_local(
            load_path,
            embeddings=self.embeddings
        )
        print("✓ 向量数据库加载完成")
    
    def get_stats(self) -> Dict:
        """
        获取检索器统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            "embedding_model": self.embedding_model_name,
            "device": self.device,
        }
        
        if self.vector_store is None:
            stats["status"] = "未初始化"
            stats["document_count"] = 0
        else:
            stats["status"] = "已初始化"
            stats["document_count"] = self.vector_store.index.ntotal
            
        return stats


if __name__ == "__main__":
    # 测试代码
    from document_processor import DocumentProcessor
    
    # 测试文档
    test_docs = [
        Document(page_content="信息检索是从大规模数据中查找相关信息的过程", metadata={"source": "test"}),
        Document(page_content="大语言模型可以理解和生成自然语言", metadata={"source": "test"}),
        Document(page_content="RAG结合了检索和生成两种技术", metadata={"source": "test"}),
    ]
    
    # 初始化检索器
    retriever = VectorRetriever(device="cpu")  # 测试时用CPU
    
    # 构建索引
    retriever.build_index(test_docs)
    
    # 检索
    query = "什么是RAG?"
    results = retriever.retrieve(query, top_k=2)
    
    print(f"\n查询: {query}")
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n结果 {i} (相似度: {score:.4f}):")
        print(f"  {doc.page_content}")
