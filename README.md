# RAG问答系统

一个基于检索增强生成(RAG)技术的智能问答系统，结合 FAISS 向量检索和 Qwen3 大语言模型。

## 📌 项目简介

本项目是信息检索课程的期末项目，展示了如何将传统信息检索技术与大语言模型结合：
- **检索阶段**：使用向量数据库(FAISS)检索相关文档
- **生成阶段**：使用大语言模型基于检索结果生成答案

## ✨ 主要特性

- 🔍 **高效检索**：基于 BGE 中文 Embedding 模型的语义检索
- 🤖 **智能生成**：使用 Qwen3-0.6B 模型生成答案（支持 4bit 量化）
- 📄 **多格式支持**：支持 PDF、TXT、DOCX、Markdown 文档
- 🌐 **友好界面**：基于 Gradio 的 Web 交互界面
- 💾 **知识库管理**：支持文档上传、知识库构建和更新

## 🛠️ 技术栈

- **框架**: LangChain
- **向量数据库**: FAISS
- **Embedding**: bge-small-zh-v1.5 (中文)
- **LLM**: Qwen3-0.6B-Instruct (可替换为 3B/7B)
- **Web界面**: Gradio
- **推理加速**: 4bit 量化 (bitsandbytes)

## 📋 环境要求

- Python 3.8+
- CUDA GPU (推荐，8GB显存足够)
- 或 CPU (速度较慢但也能运行)

## 🚀 快速开始

### 1. 安装依赖

```bash
# 使用指定的Python环境
D:/Anaconda3/envs/LLM_env/python.exe -m pip install -r requirements.txt
```

### 2. 准备文档

将文档放入 `data/raw/` 目录：

```bash
data/raw/
├── document1.pdf
├── document2.txt
└── notes.md
```

### 3. 启动应用

```bash
D:/Anaconda3/envs/LLM_env/python.exe app.py
```

启动后访问: http://127.0.0.1:7860

### 4. 使用流程

1. **构建知识库**：在"文档管理"页面点击"从data/raw目录构建知识库"
2. **开始问答**：在"问答"页面输入问题，系统会检索相关文档并生成答案
3. **添加文档**：可以通过界面上传新文档到知识库

## 📁 项目结构

```
信息检索原理作业/
├── app.py                   # Web应用入口
├── config.yaml              # 配置文件
├── requirements.txt         # 依赖列表
├── data/                    # 数据目录
│   ├── raw/                # 原始文档
│   └── processed/          # 处理后的文档
├── vector_store/            # 向量数据库存储
├── models/                  # 模型缓存
└── src/                     # 源代码
    ├── document_processor.py  # 文档处理模块
    ├── retriever.py           # 检索模块
    ├── generator.py           # 生成模块
    └── rag_system.py          # RAG主流程
```

## ⚙️ 配置说明

编辑 `config.yaml` 可以修改配置：

```yaml
models:
  llm:
    # 可选模型（根据显存调整）：
    name: "Qwen/Qwen3-0.6B-Instruct"      # 最轻量
    # name: "Qwen/Qwen2.5-3B-Instruct"    # 更好效果
    # name: "Qwen/Qwen2.5-7B-Instruct"    # 最佳效果（8GB极限）
    
document:
  chunk_size: 512        # 文档块大小
  chunk_overlap: 50      # 重叠大小
  
retrieval:
  top_k: 5              # 检索文档数量
```

## 🎯 使用示例

### 命令行使用

```python
from src.rag_system import RAGSystem

# 初始化系统
rag = RAGSystem("config.yaml")

# 构建知识库（首次）
rag.build_knowledge_base("data/raw")

# 提问
result = rag.query("什么是信息检索？")
print(result['answer'])
```

### Web界面使用

1. 启动应用后打开浏览器
2. 上传文档或使用已有文档构建知识库
3. 在问答页面输入问题
4. 查看答案和参考来源

## 📊 性能优化提示

### 8GB 显存配置
- 使用 Qwen3-0.6B 或 Qwen2.5-3B
- 启用 4bit 量化
- chunk_size 保持 512

### 16GB+ 显存配置
- 可使用 Qwen2.5-7B 获得更好效果
- 可适当增大 chunk_size

### CPU 运行
修改 `config.yaml`:
```yaml
models:
  embedding:
    device: "cpu"
  llm:
    device: "cpu"
    load_in_4bit: false
```

## 🔧 常见问题

**Q: 首次运行很慢？**  
A: 首次运行会下载模型（约 2GB），请耐心等待。模型会缓存到 `models/` 目录。

**Q: 显存不足？**  
A: 尝试使用更小的模型或切换到 CPU 模式。

**Q: 回答质量不满意？**  
A: 可以尝试：
- 增加 top_k 检索更多文档
- 使用更大的模型（3B/7B）
- 调整文档切分策略

## 📚 相关资源

- [LangChain 文档](https://python.langchain.com/)
- [Qwen3 模型](https://huggingface.co/Qwen/Qwen3-0.6B-Instruct)
- [BGE Embedding](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- [FAISS](https://github.com/facebookresearch/faiss)

## 📄 许可证

本项目仅用于课程学习。

## 🙏 致谢

- Qwen 团队提供的开源模型
- LangChain 社区
- BAAI 的中文 Embedding 模型
