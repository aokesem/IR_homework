"""
检索模块 (混合检索 + 重排序增强版)
结合 Vector(语义) + BM25(关键词) + Cross-Encoder(重排序)
"""
import os
import pickle
import jieba
import numpy as np
from typing import List, Dict, Tuple, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import torch

class VectorRetriever:
    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-small-zh-v1.5",
        device: str = "cuda",
        vector_store_path: str = None,
        reranker_name: str = "BAAI/bge-reranker-base",
        enable_rerank: bool = True,
        dummy: bool = False
    ):
        """
        初始化混合检索器
        
        Args:
            embedding_model_name: Embedding模型名称
            device: 设备（cuda/cpu）
            vector_store_path: 向量数据库保存路径
            reranker_name: 重排序模型名称
            enable_rerank: 是否启用重排序
            dummy: 是否开启虚拟模式
        """
        self.embedding_model_name = embedding_model_name
        self.device = device
        self.vector_store_path = vector_store_path
        self.reranker_name = reranker_name
        self.enable_rerank = enable_rerank
        self.dummy = dummy
        
        # 索引状态
        self.vector_store = None
        self.bm25 = None
        self.bm25_docs = []
        self.reranker = None
        
        if dummy:
            print(f"⚠️ 虚拟模式：跳过模型加载")
            return

        # 1. 初始化Embedding模型
        print(f"正在加载Embedding模型: {embedding_model_name}")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
            print("✓ Embedding模型加载完成")
        except Exception as e:
            print(f"❌ Embedding模型加载失败: {e}")
            raise e
            
        # 2. 初始化Reranker模型
        if self.enable_rerank:
            print(f"正在加载Reranker模型: {reranker_name}")
            try:
                self.reranker = CrossEncoder(
                    reranker_name, 
                    device=device,
                    max_length=512
                )
                print("✓ Reranker模型加载完成")
            except Exception as e:
                print(f"⚠️ Reranker加载失败，将降级为仅检索模式: {e}")
                self.enable_rerank = False
    
    def _tokenize_zh(self, text: str) -> List[str]:
        """中文分词辅助函数"""
        return list(jieba.cut_for_search(text))

    def build_index(self, documents: List[Document]) -> None:
        """构建混合索引"""
        if not documents:
            raise ValueError("文档列表为空")
        
        print(f"正在构建索引 ({len(documents)} 个文档块)...")
        
        # 1. FAISS
        print("  - 构建向量索引...")
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        # 2. BM25
        print("  - 构建BM25索引...")
        tokenized_corpus = [self._tokenize_zh(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.bm25_docs = documents
        
        print("✓ 索引构建完成")
    
    def add_documents(self, documents: List[Document]) -> None:
        """添加文档"""
        if self.vector_store is None:
            self.build_index(documents)
            return

        print(f"正在添加 {len(documents)} 个文档块...")
        self.vector_store.add_documents(documents)
        
        # 重构BM25
        self.bm25_docs.extend(documents)
        tokenized_corpus = [self._tokenize_zh(doc.page_content) for doc in self.bm25_docs]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        print("✓ 文档添加完成")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        rrf_k: int = 60
    ) -> List[Tuple[Document, float]]:
        """
        混合检索 + 重排序
        """
        if self.dummy:
            return self._dummy_results(query)
            
        if self.vector_store is None:
            raise ValueError("索引未构建")
            
        # ---------------------------------------------------------
        # 第一阶段：粗排 (Retrieval) - 获取更多的候选集 (Top-N)
        # ---------------------------------------------------------
        # 如果启用了 Rerank，我们检索 10 倍于 top_k 的文档，否则只检索 top_k
        initial_k = top_k * 10 if self.enable_rerank else top_k
        
        # 1. 向量检索
        vector_results = self.vector_store.similarity_search_with_score(query, k=initial_k)
        
        # 2. BM25检索
        tokenized_query = self._tokenize_zh(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_n_indices = np.argsort(bm25_scores)[::-1][:initial_k]
        
        # 3. RRF 融合
        fused_scores = {}
        doc_map = {}
        
        for rank, (doc, dist) in enumerate(vector_results):
            similarity = 1 / (1 + dist)
            if similarity < score_threshold: continue
            
            # 使用内容哈希或ID作为Key会更稳健，这里简化用内容
            doc_key = doc.page_content
            doc_map[doc_key] = doc
            fused_scores[doc_key] = fused_scores.get(doc_key, 0.0) + 1.0 / (rrf_k + rank + 1)
            
        for rank, idx in enumerate(bm25_top_n_indices):
            doc = self.bm25_docs[idx]
            doc_key = doc.page_content
            if doc_key not in doc_map:
                doc_map[doc_key] = doc
            fused_scores[doc_key] = fused_scores.get(doc_key, 0.0) + 1.0 / (rrf_k + rank + 1)
        
        # 初步排序结果
        sorted_candidates = sorted(
            fused_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:initial_k]
        
        # 如果不启用 Rerank，直接返回 RRF 结果
        if not self.enable_rerank or not self.reranker:
            return [(doc_map[k], v) for k, v in sorted_candidates[:top_k]]
            
        # ---------------------------------------------------------
        # 第二阶段：精排 (Reranking)
        # ---------------------------------------------------------
        candidate_docs = [doc_map[k] for k, v in sorted_candidates]
        if not candidate_docs:
            return []
            
        # 构建 (Query, Document) 对
        rerank_pairs = [[query, doc.page_content] for doc in candidate_docs]
        
        # Cross-Encoder 打分
        rerank_scores = self.reranker.predict(rerank_pairs)
        
        # 组合结果并重新排序
        reranked_results = []
        for doc, score in zip(candidate_docs, rerank_scores):
            reranked_results.append((doc, float(score)))
            
        # 按 Reranker 分数降序排列
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        return reranked_results[:top_k]

    def save(self, path: str = None) -> None:
        """保存索引"""
        save_path = path or self.vector_store_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        print(f"保存索引到: {save_path}")
        if self.vector_store:
            self.vector_store.save_local(save_path)
        
        bm25_path = os.path.join(save_path, "bm25.pkl")
        with open(bm25_path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "bm25_docs": self.bm25_docs}, f)
        print("✓ 保存完成")
    
    def load(self, path: str = None) -> None:
        """加载索引"""
        load_path = path or self.vector_store_path
        print(f"加载索引: {load_path}")
        
        self.vector_store = FAISS.load_local(
            load_path, 
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        bm25_path = os.path.join(load_path, "bm25.pkl")
        if os.path.exists(bm25_path):
            with open(bm25_path, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.bm25_docs = data["bm25_docs"]
        print("✓ 加载完成")

    def get_stats(self) -> Dict:
        count = len(self.bm25_docs) if self.bm25_docs else 0
        if count == 0 and self.vector_store:
            try: count = self.vector_store.index.ntotal
            except: pass
            
        return {
            "embedding_model": self.embedding_model_name,
            "reranker_model": self.reranker_name if self.enable_rerank else "Disabled",
            "document_count": count,
            "device": self.device,
            "status": "Ready" if self.vector_store else "Not Initialized"
        }
        
    def get_all_sources(self) -> List[Dict]:
        """获取源文件统计"""
        if not self.bm25_docs: return []
        source_counts = {}
        for doc in self.bm25_docs:
            source = doc.metadata.get('file_name', '未知文件')
            source_counts[source] = source_counts.get(source, 0) + 1
        return [{"file_name": k, "chunk_count": v} for k, v in source_counts.items()]

    def _dummy_results(self, query):
        return [(Document(page_content="Dummy Doc"), 0.9)]

if __name__ == "__main__":
    # 测试代码
    docs = [
        Document(page_content="苹果很好吃", metadata={"id": 1}),
        Document(page_content="苹果是一家伟大的科技公司", metadata={"id": 2}),
    ]
    
    # 启用 Rerank 测试
    retriever = VectorRetriever(enable_rerank=True, device="cpu")
    retriever.build_index(docs)
    
    q = "乔布斯"
    print(f"\nQuery: {q}")
    results = retriever.retrieve(q, top_k=2)
    for doc, score in results:
        print(f"- {doc.page_content} (Score: {score:.4f})")
