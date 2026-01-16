"""
检索模块 (混合检索增强版)
结合 Vector(语义) + BM25(关键词) 并使用 RRF 进行融合
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


class VectorRetriever:
    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-small-zh-v1.5",
        device: str = "cuda",
        vector_store_path: str = None,
        dummy: bool = False
    ):
        """
        初始化混合检索器
        
        Args:
            embedding_model_name: Embedding模型名称
            device: 设备（cuda/cpu）
            vector_store_path: 向量数据库保存路径
            dummy: 是否开启虚拟模式
        """
        self.embedding_model_name = embedding_model_name
        self.device = device
        self.vector_store_path = vector_store_path
        self.dummy = dummy
        
        # 索引状态
        self.vector_store = None
        self.bm25 = None
        self.bm25_docs = []  # BM25对应的原始文档列表（用于索引映射）
        
        if dummy:
            print(f"⚠️ 虚拟模式：跳过加载Embedding模型 {embedding_model_name}")
            self.embeddings = None
            return

        # 初始化embedding模型
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
    
    def _tokenize_zh(self, text: str) -> List[str]:
        """中文分词辅助函数"""
        return list(jieba.cut_for_search(text))

    def build_index(self, documents: List[Document]) -> None:
        """
        构建混合索引 (Vector + BM25)
        
        Args:
            documents: 文档块列表
        """
        if not documents:
            raise ValueError("文档列表为空，无法构建索引")
        
        print(f"正在构建索引 ({len(documents)} 个文档块)...")
        
        # 1. 构建向量索引 (FAISS)
        print("  - 构建向量索引...")
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        # 2. 构建BM25索引
        print("  - 构建BM25关键词索引...")
        tokenized_corpus = [self._tokenize_zh(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.bm25_docs = documents  # 保存文档引用以供检索时返回
        
        print("✓ 混合索引构建完成")
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        添加文档到索引
        注意：BM25不支持增量更新，这里会全量重构BM25
        """
        if self.vector_store is None:
            self.build_index(documents)
            return

        print(f"正在添加 {len(documents)} 个文档块到索引...")
        
        # 1. 更新向量索引
        self.vector_store.add_documents(documents)
        
        # 2. 重构BM25 (将新文档加入现有列表并重构)
        # 这是一个简化处理，生产环境可能需要更高效的方案
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
        混合检索 (RRF融合)
        
        Args:
            query: 查询文本
            top_k: 最终返回结果数量
            score_threshold: 向量检索的相似度阈值 (辅助过滤)
            rrf_k: RRF算法中的平滑常数 (通常取60)
            
        Returns:
            (文档, 融合分数) 列表
        """
        if self.dummy:
            return self._dummy_results(query)

        if self.vector_store is None or self.bm25 is None:
            raise ValueError("索引尚未构建，请先调用build_index()")
        
        # 1. 向量检索 (Vector Search)
        # 获取更多的候选集以便融合 (通常取 top_k * 2 或更多)
        candidate_k = top_k * 5 
        
        vector_results = self.vector_store.similarity_search_with_score(query, k=candidate_k)
        
        # 2. BM25检索 (Keyword Search)
        tokenized_query = self._tokenize_zh(query)
        # BM25Okapi.get_scores 返回所有文档的分数，我们需要手动排序取top
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_n_indices = np.argsort(bm25_scores)[::-1][:candidate_k]
        
        # 3. RRF 融合 (Reciprocal Rank Fusion)
        # Score = 1 / (k + rank_i)
        
        fused_scores = {}  # Map: doc_content -> score
        doc_map = {}       # Map: doc_content -> Document obj
        
        # 处理向量结果
        for rank, (doc, dist) in enumerate(vector_results):
            # 过滤掉相似度过低的结果 (distance越小越好，1/(1+dist)越大越好)
            similarity = 1 / (1 + dist)
            if similarity < score_threshold:
                continue
                
            doc_content = doc.page_content
            doc_map[doc_content] = doc
            
            if doc_content not in fused_scores:
                fused_scores[doc_content] = 0.0
            fused_scores[doc_content] += 1.0 / (rrf_k + rank + 1)
            
        # 处理BM25结果
        for rank, idx in enumerate(bm25_top_n_indices):
            doc = self.bm25_docs[idx]
            doc_content = doc.page_content
            # 更新Map确保能找到对象 (如果该文档没在向量检索中出现)
            if doc_content not in doc_map:
                doc_map[doc_content] = doc
            
            if doc_content not in fused_scores:
                fused_scores[doc_content] = 0.0
            fused_scores[doc_content] += 1.0 / (rrf_k + rank + 1)
        
        # 4. 排序并返回最终 Top K
        sorted_results = sorted(
            fused_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        return [(doc_map[content], score) for content, score in sorted_results]

    def save(self, path: str = None) -> None:
        """保存混合索引"""
        if self.vector_store is None:
            raise ValueError("索引未构建")
        
        save_path = path or self.vector_store_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        print(f"正在保存索引到: {save_path}")
        
        # 保存向量索引
        self.vector_store.save_local(save_path)
        
        # 保存BM25索引和文档列表
        bm25_path = os.path.join(save_path, "bm25.pkl")
        with open(bm25_path, "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "bm25_docs": self.bm25_docs
            }, f)
            
        print("✓ 索引保存完成")
    
    def load(self, path: str = None) -> None:
        """加载混合索引"""
        load_path = path or self.vector_store_path
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"路径不存在: {load_path}")
        
        print(f"正在加载索引: {load_path}")
        
        # 加载向量索引
        self.vector_store = FAISS.load_local(
            load_path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        # 加载BM25
        bm25_path = os.path.join(load_path, "bm25.pkl")
        if os.path.exists(bm25_path):
            with open(bm25_path, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.bm25_docs = data["bm25_docs"]
        else:
            print("⚠️ 未找到BM25索引文件，将仅使用向量检索")
            
        print("✓ 索引加载完成")

    def _dummy_results(self, query):
        return [
            (Document(page_content=f"这是来自虚拟知识库的内容1...", metadata={"file_name": "dummy.pdf"}), 0.9),
            (Document(page_content=f"这是来自虚拟知识库的内容2...", metadata={"file_name": "dummy.txt"}), 0.8)
        ]

    def get_all_sources(self) -> List[Dict]:
        """获取源文件统计"""
        if not self.bm25_docs:
            if self.vector_store:
                # Fallback to vector store if bm25_docs empty (e.g. loaded from old version)
                return [] 
            return []
            
        source_counts = {}
        for doc in self.bm25_docs:
            source = doc.metadata.get('file_name', '未知文件')
            source_counts[source] = source_counts.get(source, 0) + 1
            
        return [
            {"file_name": name, "chunk_count": count}
            for name, count in source_counts.items()
        ]

    def get_stats(self) -> Dict:
        return {
            "embedding_model": self.embedding_model_name,
            "device": self.device,
            "doc_count": len(self.bm25_docs) if self.bm25_docs else 0,
            "status": "Ready" if self.vector_store else "Not Initialized"
        }

if __name__ == "__main__":
    # 简易测试
    docs = [
        Document(page_content="机器学习是人工智能的一个子集", metadata={"id": 1}),
        Document(page_content="深度学习依赖于神经网络", metadata={"id": 2}),
        Document(page_content="苹果是一种水果，很好吃", metadata={"id": 3}),
    ]
    
    retriever = VectorRetriever(device="cpu")
    retriever.build_index(docs)
    
    q = "神经网络"
    print(f"\n查询: {q}")
    results = retriever.retrieve(q, top_k=2)
    for doc, score in results:
        print(f"- {doc.page_content} (RRF Score: {score:.4f})")