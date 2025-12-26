"""
文档处理模块
负责加载、清洗和切分文档
"""
import os
import json
import time
from typing import List, Dict
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
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, processed_dir: str = None):
        """
            chunk_size: 每个文档块的字符数
            chunk_overlap: 块之间重叠的字符数
            processed_dir: 缓存已解析文档的目录
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.processed_dir = Path(processed_dir) if processed_dir else None
        if self.processed_dir:
            self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化文本切分器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""],
            length_function=len,
        )
        
        # 支持的文件格式映射
        self.loader_mapping = {
            '.txt': TextLoader,
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.md': UnstructuredMarkdownLoader,
        }
    
    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Args:
            file_paths: 文件路径列表
            
        Returns:
            文档列表
        """
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
        """
        Args:
            file_path: 文件路径
            
        Returns:
            文档列表
        """
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext not in self.loader_mapping:
            raise ValueError(f"不支持的文件格式: {file_ext}")
        
        loader_class = self.loader_mapping[file_ext]
        
        # 特殊处理：PDF和文本文件使用UTF-8编码
        if file_ext == '.txt':
            loader = loader_class(file_path, encoding='utf-8')
        else:
            loader = loader_class(file_path)
        
        documents = loader.load()
        
        # 为每个文档添加元数据
        for doc in documents:
            doc.metadata['source'] = file_path
            doc.metadata['file_name'] = os.path.basename(file_path)
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Args:
            documents: 文档列表
            
        Returns:
            切分后的文档块列表
        """
        chunks = self.text_splitter.split_documents(documents)
        
        # 为每个chunk添加索引
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
        
        print(f"文档切分完成: {len(documents)} 个文档 -> {len(chunks)} 个文档块")
        
        return chunks
    
    def _get_cache_path(self, file_path: str) -> Path:
        """获取缓存文件路径"""
        if not self.processed_dir:
            return None
        # 使用原文件名 + .json 作为缓存扩展名
        rel_path = os.path.basename(file_path)
        return self.processed_dir / f"{rel_path}.json"

    def _save_cache(self, file_path: str, documents: List[Document]):
        """将解析后的文档保存到缓存"""
        cache_path = self._get_cache_path(file_path)
        if not cache_path:
            return
            
        try:
            # 记录原始内容和修改时间以进行校验
            mtime = os.path.getmtime(file_path)
            cache_data = {
                "file_path": file_path,
                "mtime": mtime,
                "documents": [
                    {"page_content": doc.page_content, "metadata": doc.metadata}
                    for doc in documents
                ]
            }
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"警告: 写入缓存失败 {file_path}: {e}")

    def _load_cache(self, file_path: str) -> List[Document]:
        """从缓存加载文档"""
        cache_path = self._get_cache_path(file_path)
        if not cache_path or not cache_path.exists():
            return None
            
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 校验文件是否被修改过
            if data.get("mtime") != os.path.getmtime(file_path):
                return None
                
            documents = [
                Document(page_content=d["page_content"], metadata=d["metadata"])
                for d in data["documents"]
            ]
            return documents
        except Exception:
            return None
    
    def clean_text(self, text: str) -> str:
        """
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本
        """
        # 移除多余的空白字符
        text = ' '.join(text.split())
        
        # 移除特殊字符（根据需要调整）
        # text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
        
        return text.strip()
    
    def process_directory(self, directory: str) -> List[Document]:
        """
        Args:
            directory: 目录路径
            
        Returns:
            处理后的文档块列表
        """
        # 收集所有支持的文件
        file_paths = []
        for ext in self.loader_mapping.keys():
            file_paths.extend(Path(directory).glob(f'**/*{ext}'))
        
        file_paths = [str(fp) for fp in file_paths]
        
        if not file_paths:
            print(f"警告: 在 {directory} 中未找到支持的文档")
            return []
        
        print(f"找到 {len(file_paths)} 个文档文件")
        
        # 加载文档（带有缓存逻辑）
        documents = []
        for fp in file_paths:
            # 尝试从缓存加载
            cached_docs = self._load_cache(fp)
            if cached_docs:
                documents.extend(cached_docs)
                print(f"🚀 从缓存加载: {os.path.basename(fp)}")
            else:
                # 正常解析
                try:
                    docs = self._load_single_file(fp)
                    # 清洗文本
                    for doc in docs:
                        doc.page_content = self.clean_text(doc.page_content)
                    
                    documents.extend(docs)
                    # 写入缓存
                    self._save_cache(fp, docs)
                    print(f"📂 解析新文件: {os.path.basename(fp)}")
                except Exception as e:
                    print(f"✗ 处理失败: {fp}, 错误: {str(e)}")
        
        # 统一进行文档切分
        chunks = self.split_documents(documents)
        
        return chunks


if __name__ == "__main__":
    # 测试代码
    processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)
    
    # 测试处理目录
    chunks = processor.process_directory("../data/raw")
    
    if chunks:
        print(f"\n示例文档块:")
        print(f"内容预览: {chunks[0].page_content[:200]}...")
        print(f"元数据: {chunks[0].metadata}")
