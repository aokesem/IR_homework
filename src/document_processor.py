"""
文档处理模块
负责加载、清洗、切分文档及上下文增强
支持: 语义切分 (Semantic Chunking)
支持: 多版本共存 (A/B Test)
"""
import os
import json
import time
from typing import List, Dict, Optional
from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 尝试导入语义切分器
try:
    from langchain_experimental.text_splitter import SemanticChunker
    HAS_SEMANTIC_CHUNKER = True
except ImportError:
    HAS_SEMANTIC_CHUNKER = False
    print("⚠️ 未找到 langchain_experimental，语义切分不可用")

class DocumentProcessor:
    """文档处理器"""
    
    CONTEXT_PROMPT = """请为以下文本片段生成一个简短的上下文说明（20-30字以内）。
说明该片段来自文件《{filename}》，并概括其核心内容。
格式要求：[关于{filename}的...说明]

文本片段：
{chunk_content}

上下文说明："""

    def __init__(
        self, 
        chunk_size: int = 512, 
        chunk_overlap: int = 50, 
        processed_dir: str = None,
        use_semantic_chunking: bool = False,
        embeddings = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.processed_dir = Path(processed_dir) if processed_dir else None
        self.default_use_semantic = use_semantic_chunking # 保存默认设置
        self.embeddings = embeddings
        
        if self.processed_dir:
            self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 基础切分器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""],
            length_function=len,
        )
        
        # 2. 语义切分器 (预初始化)
        self.semantic_splitter = None
        if HAS_SEMANTIC_CHUNKER and embeddings:
            try:
                self.semantic_splitter = SemanticChunker(
                    embeddings,
                    breakpoint_threshold_type="percentile",
                    breakpoint_threshold_amount=90
                )
            except Exception as e:
                print(f"❌ 初始化语义切分器失败: {e}")
        
        self.loader_mapping = {
            '.txt': TextLoader,
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.md': UnstructuredMarkdownLoader,
        }
    
    def load_documents(self, file_paths: List[str]) -> List[Document]:
        documents = []
        for file_path in file_paths:
            try:
                docs = self._load_single_file(file_path)
                documents.extend(docs)
            except Exception as e:
                print(f"✗ 加载失败: {file_path}, {e}")
        return documents
    
    def _load_single_file(self, file_path: str) -> List[Document]:
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in self.loader_mapping:
            raise ValueError(f"不支持的文件格式: {file_ext}")
        
        loader_class = self.loader_mapping[file_ext]
        kwargs = {'encoding': 'utf-8'} if file_ext == '.txt' else {}
        loader = loader_class(file_path, **kwargs)
        
        documents = loader.load()
        for doc in documents:
            doc.metadata['source'] = file_path
            doc.metadata['file_name'] = os.path.basename(file_path)
        return documents
    
    def split_documents(self, documents: List[Document], use_semantic: bool = False) -> List[Document]:
        """
        切分文档
        Args:
            use_semantic: 是否强制使用语义切分
        """
        # 只有当启用了语义切分，且环境支持，且传入了embeddings时才执行
        if use_semantic and self.semantic_splitter:
            # print("  - 执行语义切分...")
            try:
                chunks = self.semantic_splitter.split_documents(documents)
                # 兜底超长块
                final_chunks = []
                for chunk in chunks:
                    if len(chunk.page_content) > self.chunk_size * 1.5:
                        sub_chunks = self.text_splitter.split_documents([chunk])
                        final_chunks.extend(sub_chunks)
                    else:
                        final_chunks.append(chunk)
                chunks = final_chunks
            except Exception as e:
                print(f"⚠️ 语义切分失败 ({e})，回退到基础切分")
                chunks = self.text_splitter.split_documents(documents)
        else:
            # print("  - 执行基础切分...")
            chunks = self.text_splitter.split_documents(documents)
        
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
        
        return chunks
    
    def clean_text(self, text: str) -> str:
        text = ' '.join(text.split())
        return text.strip()

    def augment_chunk_with_context(self, chunk: Document, generator) -> Document:
        if not generator: return chunk
        filename = chunk.metadata.get('file_name', '未知文件')
        content = chunk.page_content
        prompt = self.CONTEXT_PROMPT.format(filename=filename, chunk_content=content[:500])
        try:
            result = generator.generate(question=prompt, context_documents=[], history=[], custom_prompt="{question}")
            context_desc = result['answer'].strip().replace("上下文说明：", "").strip()
            chunk.page_content = f"{context_desc}\n{content}"
            chunk.metadata['is_augmented'] = True
            return chunk
        except Exception:
            return chunk

    def _get_cache_path(self, file_path: str, label: str = "") -> Path:
        """缓存路径包含label，防止冲突"""
        if not self.processed_dir: return None
        # label 清理文件名非法字符
        safe_label = label.replace("[", "").replace("]", "").replace(" ", "_")
        filename = f"{os.path.basename(file_path)}_{safe_label}.json"
        return self.processed_dir / filename

    def _save_cache(self, file_path: str, chunks: List[Document], label: str = ""):
        cache_path = self._get_cache_path(file_path, label)
        if not cache_path: return
        try:
            mtime = os.path.getmtime(file_path)
            cache_data = {
                "file_path": file_path,
                "mtime": mtime,
                "chunks": [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks]
            }
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"警告: 写入缓存失败 {file_path}: {e}")

    def _load_cache(self, file_path: str, label: str = "") -> List[Document]:
        cache_path = self._get_cache_path(file_path, label)
        if not cache_path or not cache_path.exists(): return None
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data.get("mtime") != os.path.getmtime(file_path): return None
            if "chunks" in data:
                 return [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data["chunks"]]
            return None
        except Exception:
            return None

    def process_directory(
        self, 
        directory: str, 
        generator=None, 
        label: str = "", 
        enable_semantic: bool = False,
        enable_augmentation: bool = False
    ) -> List[Document]:
        """
        处理目录
        Args:
            label: 附加到文件名的标签 (e.g. " [基础版]")
            enable_semantic: 是否启用语义切分
            enable_augmentation: 是否启用LLM增强
        """
        file_paths = []
        for ext in self.loader_mapping.keys():
            file_paths.extend(Path(directory).glob(f'**/*{ext}'))
        file_paths = [str(fp) for fp in file_paths]
        
        if not file_paths:
            return []
        
        final_chunks = []
        print(f"\n>> 开始处理任务: {label} (语义切分={enable_semantic}, 增强={enable_augmentation})")
        
        for fp in file_paths:
            # 1. 尝试从缓存加载 (带Label)
            cached_chunks = self._load_cache(fp, label)
            if cached_chunks:
                final_chunks.extend(cached_chunks)
                # print(f"🚀 [缓存] {os.path.basename(fp)}{label}")
                continue
            
            try:
                # Load
                docs = self._load_single_file(fp)
                for doc in docs: doc.page_content = self.clean_text(doc.page_content)
                
                # Split (传入特定的策略)
                file_chunks = self.split_documents(docs, use_semantic=enable_semantic)
                
                # Augment (仅当启用增强且传入了生成器时)
                if enable_augmentation and generator:
                    print(f"🤖 [增强] {os.path.basename(fp)} ({len(file_chunks)} chunks)...")
                    augmented_chunks = []
                    for chunk in file_chunks:
                        aug_chunk = self.augment_chunk_with_context(chunk, generator)
                        augmented_chunks.append(aug_chunk)
                        print(".", end="", flush=True)
                    print(" Done")
                    file_chunks = augmented_chunks
                
                # Tagging: 修改文件名元数据，加上 Label
                for chunk in file_chunks:
                    chunk.metadata['file_name'] = chunk.metadata['file_name'] + label
                
                # Save chunks to cache (带Label)
                self._save_cache(fp, file_chunks, label)
                final_chunks.extend(file_chunks)
                print(f"✅ [完成] {os.path.basename(fp)}{label}")
                
            except Exception as e:
                print(f"✗ 失败: {fp}, {e}")
                
        return final_chunks

if __name__ == "__main__":
    pass