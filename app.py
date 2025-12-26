"""
RAG问答系统 Web界面
使用Gradio构建交互界面
"""
import gradio as gr
import os
import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

from src.rag_system import RAGSystem


class RAGWebApp:
    """RAG Web应用"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化Web应用
        
        Args:
            config_path: 配置文件路径
        """
        print("正在初始化RAG系统...")
        self.rag_system = RAGSystem(config_path)
        
        # 尝试加载已有知识库
        vector_store_path = self.rag_system.config['paths']['vector_store']
        if os.path.exists(vector_store_path):
            try:
                self.rag_system.load_knowledge_base()
                self.kb_loaded = True
            except Exception as e:
                print(f"加载知识库失败: {e}")
                self.kb_loaded = False
        else:
            self.kb_loaded = False
    
    def answer_question(self, question: str, top_k: int = 5) -> tuple:
        """
        回答问题
        
        Args:
            question: 用户问题
            top_k: 检索文档数量
            
        Returns:
            (答案, 来源文档)
        """
        if not self.kb_loaded:
            return "⚠️ 知识库尚未构建，请先上传文档", ""
        
        if not question.strip():
            return "请输入问题", ""
        
        # 查询
        result = self.rag_system.query(question, top_k=top_k)
        
        # 格式化答案
        answer = result['answer']
        
        # 格式化来源
        sources_text = f"**使用了 {result['num_sources']} 个参考资料**\n\n"
        for i, source in enumerate(result.get('sources', []), 1):
            sources_text += f"**[资料{i}]** {source['file_name']} (相似度: {source['similarity']})\n"
            sources_text += f"> {source['content']}\n\n"
        
        return answer, sources_text
    
    def build_kb_from_directory(self, progress=gr.Progress()):
        """从data/raw目录构建知识库"""
        try:
            progress(0, desc="开始处理文档...")
            
            raw_docs_path = self.rag_system.config['paths']['raw_docs']
            if not os.path.exists(raw_docs_path):
                return f"❌ 文档目录不存在: {raw_docs_path}"
            
            progress(0.3, desc="加载并切分文档...")
            self.rag_system.build_knowledge_base()
            
            progress(1.0, desc="完成!")
            self.kb_loaded = True
            
            stats = self.rag_system.retriever.get_stats()
            return f"✅ 知识库构建成功!\n文档块数量: {stats['document_count']}"
            
        except Exception as e:
            return f"❌ 构建失败: {str(e)}"
    
    def upload_files(self, files, progress=gr.Progress()):
        """
        上传文件到知识库
        
        Args:
            files: 上传的文件列表
        """
        if not files:
            return "请选择文件"
        
        try:
            progress(0, desc="处理文件...")
            
            # 保存文件到raw目录
            raw_docs_path = self.rag_system.config['paths']['raw_docs']
            os.makedirs(raw_docs_path, exist_ok=True)
            
            file_paths = []
            for file in files:
                file_path = os.path.join(raw_docs_path, os.path.basename(file.name))
                # 复制文件
                import shutil
                shutil.copy(file.name, file_path)
                file_paths.append(file_path)
            
            progress(0.5, desc="添加到知识库...")
            
            if not self.kb_loaded:
                # 首次构建
                self.rag_system.build_knowledge_base()
                self.kb_loaded = True
            else:
                # 添加到已有知识库
                self.rag_system.add_documents(file_paths)
            
            progress(1.0, desc="完成!")
            
            stats = self.rag_system.retriever.get_stats()
            return f"✅ 成功添加 {len(file_paths)} 个文件!\n当前文档块数量: {stats['document_count']}"
            
        except Exception as e:
            return f"❌ 上传失败: {str(e)}"
    
    def get_system_info(self):
        """获取系统信息"""
        info = self.rag_system.get_system_info()
        
        text = "### 系统配置\n\n"
        text += f"**Embedding模型**: {info['retriever']['embedding_model']}\n"
        text += f"**LLM模型**: {info['generator']['model_name']}\n"
        text += f"**设备**: {info['generator']['device']}\n\n"
        
        text += "### 文档处理\n\n"
        text += f"**切分大小**: {info['document_processor']['chunk_size']} 字符\n"
        text += f"**重叠大小**: {info['document_processor']['chunk_overlap']} 字符\n\n"
        
        text += "### 知识库状态\n\n"
        if self.kb_loaded:
            text += f"**状态**: ✅ 已加载\n"
            text += f"**文档块数量**: {info['retriever']['document_count']}\n"
        else:
            text += f"**状态**: ⚠️ 未加载\n"
        
        return text
    
    def create_interface(self):
        """创建Gradio界面"""
        with gr.Blocks(title="RAG问答系统", theme=gr.themes.Soft()) as demo:
            gr.Markdown("""
            # 📚 RAG问答系统
            ### 基于检索增强生成的智能问答助手
            """)
            
            with gr.Tabs():
                # Tab 1: 问答
                with gr.Tab("💬 问答"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            question_input = gr.Textbox(
                                label="输入问题",
                                placeholder="例如：什么是信息检索？",
                                lines=2
                            )
                            top_k_slider = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=5,
                                step=1,
                                label="检索文档数量 (Top-K)"
                            )
                            submit_btn = gr.Button("🔍 提交问题", variant="primary", size="lg")
                        
                    with gr.Row():
                        with gr.Column():
                            answer_output = gr.Textbox(
                                label="📝 答案",
                                lines=5,
                                interactive=False
                            )
                        
                    with gr.Accordion("📑 参考来源", open=False):
                        sources_output = gr.Markdown()
                    
                    # 示例问题
                    gr.Examples(
                        examples=[
                            ["什么是信息检索？"],
                            ["RAG的主要优势是什么？"],
                            ["如何评估检索系统的性能？"],
                        ],
                        inputs=question_input
                    )
                
                # Tab 2: 文档管理
                with gr.Tab("📁 文档管理"):
                    gr.Markdown("### 上传文档到知识库")
                    gr.Markdown("支持格式: PDF, TXT, DOCX, Markdown")
                    
                    with gr.Row():
                        file_upload = gr.File(
                            label="选择文件",
                            file_count="multiple",
                            file_types=[".pdf", ".txt", ".docx", ".md"]
                        )
                    
                    with gr.Row():
                        upload_btn = gr.Button("📤 上传并添加到知识库", variant="primary")
                        build_btn = gr.Button("🔨 从data/raw目录构建知识库", variant="secondary")
                    
                    upload_status = gr.Textbox(label="状态", interactive=False)
                
                # Tab 3: 系统信息
                with gr.Tab("ℹ️ 系统信息"):
                    info_output = gr.Markdown()
                    refresh_btn = gr.Button("🔄 刷新信息")
                    
                    # 自动显示信息
                    demo.load(self.get_system_info, outputs=info_output)
            
            # 事件绑定
            submit_btn.click(
                fn=self.answer_question,
                inputs=[question_input, top_k_slider],
                outputs=[answer_output, sources_output]
            )
            
            upload_btn.click(
                fn=self.upload_files,
                inputs=file_upload,
                outputs=upload_status
            )
            
            build_btn.click(
                fn=self.build_kb_from_directory,
                outputs=upload_status
            )
            
            refresh_btn.click(
                fn=self.get_system_info,
                outputs=info_output
            )
        
        return demo


def main():
    """主函数"""
    # 初始化应用
    app = RAGWebApp()
    
    # 创建界面
    demo = app.create_interface()
    
    # 启动服务
    web_config = app.rag_system.config['web']
    demo.launch(
        server_name=web_config['host'],
        server_port=web_config['port'],
        share=web_config['share']
    )


if __name__ == "__main__":
    main()
