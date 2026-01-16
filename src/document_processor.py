"""
文档处理模块
负责加载、清洗、切分文档及上下文增强
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

class DocumentProcessor:
    """文档处理器"""
    
    # 上下文增强 Prompt
    CONTEXT_PROMPT = """请为以下文本片段生成一个简短的上下文说明（20-30字以内）。
说明该片段来自文件《{filename}》，并概括其核心内容。
格式要求：[关于{filename}的...说明]

文本片段：
{chunk_content}

上下文说明："""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, processed_dir: str = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.processed_dir = Path(processed_dir) if processed_dir else None
        if self.processed_dir:
            self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""],
            length_function=len,
        )
        
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
                print(f"✓ 成功加载: {file_path} ({len(docs)} 个文档)")
            except Exception as e:
                print(f"✗ 加载失败: {file_path}, 错误: {str(e)}")
        return documents
    
    def _load_single_file(self, file_path: str) -> List[Document]:
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in self.loader_mapping:
            raise ValueError(f"不支持的文件格式: {file_ext}")
        
        loader_class = self.loader_mapping[file_ext]
        if file_ext == '.txt':
            loader = loader_class(file_path, encoding='utf-8')
        else:
            loader = loader_class(file_path)
        
        documents = loader.load()
        for doc in documents:
            doc.metadata['source'] = file_path
            doc.metadata['file_name'] = os.path.basename(file_path)
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunks = self.text_splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
        print(f"文档切分完成: {len(documents)} 个文档 -> {len(chunks)} 个文档块")
        return chunks
    
    def clean_text(self, text: str) -> str:
        text = ' '.join(text.split())
        return text.strip()

    def augment_chunk_with_context(self, chunk: Document, generator) -> Document:
        """
        使用 LLM 为 Chunk 生成上下文前缀
        """
        if not generator:
            return chunk
            
        filename = chunk.metadata.get('file_name', '未知文件')
        content = chunk.page_content
        
        # 构造 Prompt
        prompt = self.CONTEXT_PROMPT.format(
            filename=filename,
            chunk_content=content[:500]  # 限制长度以节省Token
        )
        
        try:
            # 调用生成器 (使用简单的生成模式，不做RAG)
            # 这里我们需要直接访问底层接口或者使用一个特定的方法
            # 假设 generator 有 _generate_hf 或类似的直接生成能力
            # 为了通用，我们构造一个 dummy history 调用 generate
            
            result = generator.generate(
                question=prompt,
                context_documents=[], # 空上下文
                history=[],
                custom_prompt="{question}" # 直接透传
            )
            
            context_desc = result['answer'].strip()
            # 清理可能的括号
            context_desc = context_desc.replace("上下文说明：", "").strip()
            
            # 拼接到原始内容前面
            new_content = f"{context_desc}\n{content}"
            chunk.page_content = new_content
            chunk.metadata['is_augmented'] = True
            
            return chunk
            
        except Exception as e:
            print(f"⚠️ 上下文增强失败: {e}")
            return chunk

    def _get_cache_path(self, file_path: str) -> Path:
        if not self.processed_dir: return None
        return self.processed_dir / f"{os.path.basename(file_path)}.json"

    def _save_cache(self, file_path: str, chunks: List[Document]):
        """保存处理后的Chunks到缓存"""
        cache_path = self._get_cache_path(file_path)
        if not cache_path: return
        try:
            mtime = os.path.getmtime(file_path)
            cache_data = {
                "file_path": file_path,
                "mtime": mtime,
                "chunks": [ # 注意这里改为保存 chunks
                    {"page_content": c.page_content, "metadata": c.metadata}
                    for c in chunks
                ]
            }
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"警告: 写入缓存失败 {file_path}: {e}")

    def _load_cache(self, file_path: str) -> List[Document]:
        cache_path = self._get_cache_path(file_path)
        if not cache_path or not cache_path.exists(): return None
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data.get("mtime") != os.path.getmtime(file_path): return None
            # 注意这里加载的是 chunks
            if "chunks" in data:
                 return [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data["chunks"]]
            return None # 旧版本缓存不兼容
        except Exception:
            return None

    def process_directory(self, directory: str, generator=None) -> List[Document]:
        """
        处理目录，支持传入 generator 进行增强
        """
        file_paths = []
        for ext in self.loader_mapping.keys():
            file_paths.extend(Path(directory).glob(f'**/*{ext}'))
        file_paths = [str(fp) for fp in file_paths]
        
        if not file_paths:
            print(f"警告: {directory} 为空")
            return []
        
        final_chunks = []
        
        for fp in file_paths:
            # 1. 尝试从缓存加载 (此时缓存里已经是切分且增强过的 chunks)
            cached_chunks = self._load_cache(fp)
            if cached_chunks:
                final_chunks.extend(cached_chunks)
                print(f"🚀 从缓存加载Chunks: {os.path.basename(fp)}")
                continue
            
            # 2. 如果无缓存，则重新处理
            try:
                # Load
                docs = self._load_single_file(fp)
                for doc in docs: doc.page_content = self.clean_text(doc.page_content)
                
                # Split
                file_chunks = self.split_documents(docs)
                
                # Augment (如果提供了生成器)
                if generator:
                    print(f"🤖 正在增强 {len(file_chunks)} 个切片 (此过程较慢)...")
                    augmented_chunks = []
                    for chunk in file_chunks:
                        aug_chunk = self.augment_chunk_with_context(chunk, generator)
                        augmented_chunks.append(aug_chunk)
                        print(".", end="", flush=True)
                    print(" 完成!")
                    file_chunks = augmented_chunks
                
                # Save chunks to cache
                self._save_cache(fp, file_chunks)
                final_chunks.extend(file_chunks)
                
            except Exception as e:
                print(f"✗ 处理失败: {fp}, {e}")
                
        return final_chunks

if __name__ == "__main__":
    pass