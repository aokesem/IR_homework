"""
检索模块 (混合检索 + 重排序增强版)
结合 Vector(语义) + BM25(关键词) + Cross-Encoder(重排序)
支持: 动态文件过滤
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
        self.embedding_model_name = embedding_model_name
        self.device = device
        self.vector_store_path = vector_store_path
        self.reranker_name = reranker_name
        self.enable_rerank = enable_rerank
        self.dummy = dummy
        
        self.vector_store = None
        self.bm25 = None
        self.bm25_docs = []
        self.reranker = None
        
        if dummy:
            print(f"⚠️ 虚拟模式：跳过模型加载")
            return

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
        return list(jieba.cut_for_search(text))

    def build_index(self, documents: List[Document]) -> None:
        if not documents: raise ValueError("文档列表为空")
        print(f"正在构建索引 ({len(documents)} 个文档块)...")
        print("  - 构建向量索引...")
        self.vector_store = FAISS.from_documents(documents=documents, embedding=self.embeddings)
        print("  - 构建BM25索引...")
        tokenized_corpus = [self._tokenize_zh(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.bm25_docs = documents
        print("✓ 索引构建完成")
    
    def add_documents(self, documents: List[Document]) -> None:
        if self.vector_store is None:
            self.build_index(documents)
            return
        print(f"正在添加 {len(documents)} 个文档块...")
        self.vector_store.add_documents(documents)
        self.bm25_docs.extend(documents)
        tokenized_corpus = [self._tokenize_zh(doc.page_content) for doc in self.bm25_docs]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("✓ 文档添加完成")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        rrf_k: int = 60,
        file_filters: List[str] = None
    ) -> List[Tuple[Document, float]]:
        """
        混合检索 + 重排序 + 文件过滤
        Args:
            file_filters: 指定检索的文件名列表 (e.g. ["README.md", "doc1.pdf"])
                          如果为None或空，则检索所有文件
        """
        if self.dummy: return self._dummy_results(query)
        if self.vector_store is None: raise ValueError("索引未构建")
        
        initial_k = top_k * 10 if self.enable_rerank else top_k
        
        # --- 构造 FAISS 过滤器 ---
        faiss_filter = None
        if file_filters and len(file_filters) > 0:
            # LangChain FAISS 支持传入 Callable: lambda metadata: ...
            # metadata 包含 'file_name' 等字段
            # 注意: 这里的 lambda 必须能够被 pickle 序列化如果涉及多进程，但在简单场景下通常可行
            target_files = set(file_filters)
            faiss_filter = lambda metadata: metadata.get('file_name') in target_files
        
        # 1. 向量检索 (带过滤)
        try:
            vector_results = self.vector_store.similarity_search_with_score(
                query, 
                k=initial_k, 
                filter=faiss_filter
            )
        except Exception as e:
            print(f"⚠️ 向量检索过滤失败 (可能是FAISS版本问题): {e}")
            # 降级：全量检索后手动过滤
            vector_results = self.vector_store.similarity_search_with_score(query, k=initial_k * 2)
            if file_filters:
                vector_results = [
                    (doc, score) for doc, score in vector_results 
                    if doc.metadata.get('file_name') in file_filters
                ]
                vector_results = vector_results[:initial_k]

        # 2. BM25检索 (手动过滤)
        tokenized_query = self._tokenize_zh(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # 获取 top N 索引
        # 如果需要过滤，我们需要遍历更多结果来凑齐 top k
        # 这里简化处理：取 top_k * 5，然后过滤
        bm25_candidate_indices = np.argsort(bm25_scores)[::-1]
        
        bm25_results = []
        count = 0
        for idx in bm25_candidate_indices:
            if count >= initial_k: break
            
            doc = self.bm25_docs[idx]
            # 过滤逻辑
            if file_filters and doc.metadata.get('file_name') not in file_filters:
                continue
                
            bm25_results.append(idx)
            count += 1
            
        # 3. RRF 融合
        fused_scores = {}
        doc_map = {}
        
        for rank, (doc, dist) in enumerate(vector_results):
            similarity = 1 / (1 + dist)
            if similarity < score_threshold: continue
            
            doc_key = doc.page_content
            doc_map[doc_key] = doc
            fused_scores[doc_key] = fused_scores.get(doc_key, 0.0) + 1.0 / (rrf_k + rank + 1)
            
        for rank, idx in enumerate(bm25_results):
            doc = self.bm25_docs[idx]
            doc_key = doc.page_content
            if doc_key not in doc_map: doc_map[doc_key] = doc
            fused_scores[doc_key] = fused_scores.get(doc_key, 0.0) + 1.0 / (rrf_k + rank + 1)
        
        sorted_candidates = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:initial_k]
        
        if not self.enable_rerank or not self.reranker:
            return [(doc_map[k], v) for k, v in sorted_candidates[:top_k]]
            
        # 4. 重排序
        candidate_docs = [doc_map[k] for k, v in sorted_candidates]
        if not candidate_docs: return []
        
        rerank_pairs = [[query, doc.page_content] for doc in candidate_docs]
        rerank_scores = self.reranker.predict(rerank_pairs)
        
        reranked_results = []
        for doc, score in zip(candidate_docs, rerank_scores):
            reranked_results.append((doc, float(score)))
        
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        return reranked_results[:top_k]

    def save(self, path: str = None) -> None:
        save_path = path or self.vector_store_path
        if not os.path.exists(save_path): os.makedirs(save_path)
        print(f"保存索引到: {save_path}")
        if self.vector_store: self.vector_store.save_local(save_path)
        bm25_path = os.path.join(save_path, "bm25.pkl")
        with open(bm25_path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "bm25_docs": self.bm25_docs}, f)
        print("✓ 保存完成")
    
    def load(self, path: str = None) -> None:
        load_path = path or self.vector_store_path
        print(f"加载索引: {load_path}")
        self.vector_store = FAISS.load_local(load_path, embeddings=self.embeddings, allow_dangerous_deserialization=True)
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
        if not self.bm25_docs: return []
        source_counts = {}
        for doc in self.bm25_docs:
            source = doc.metadata.get('file_name', '未知文件')
            source_counts[source] = source_counts.get(source, 0) + 1
        return [{"file_name": k, "chunk_count": v} for k, v in source_counts.items()]

    def _dummy_results(self, query):
        return [(Document(page_content="Dummy Doc"), 0.9)]

if __name__ == "__main__":
    pass