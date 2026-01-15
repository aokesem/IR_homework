"""
RAG问答系统 Web界面
使用Gradio构建交互界面
"""
import gradio as gr
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import json
import time
from datetime import datetime

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
        # 环境变量控制虚拟模式快速查看开发效果
        dummy_mode = os.environ.get("RAG_DEV_MODE", "False").lower() == "true"
        self.rag_system = RAGSystem(config_path, dummy_mode=dummy_mode)
        
        # 尝试加载已有知识库 (虚拟模式下跳过)
        self.kb_loaded = False
        if not dummy_mode:
            vector_store_path = self.rag_system.config['paths']['vector_store']
            if os.path.exists(vector_store_path):
                try:
                    self.rag_system.load_knowledge_base()
                    self.kb_loaded = True
                except Exception as e:
                    print(f"加载知识库失败: {e}")
        else:
            self.kb_loaded = True 
            
        # 对话保存路径
        self.conv_dir = Path(self.rag_system.config['paths'].get('conversations', "data/conversations"))
        self.conv_dir.mkdir(parents=True, exist_ok=True)
        self.current_chat_file = None

    def answer_question(self,question: str,
    history: list,
    top_k: int = 5,
    custom_prompt: str = None):
        """
        回答问题（支持多轮对话、来源绑定及自定义 Prompt）
        """

        if not self.kb_loaded:
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": "⚠️ 知识库尚未构建，请先上传文档或重建库"})
        else:
            # 查询 RAG
            result = self.rag_system.query(
                question,
                top_k=top_k,
                history=history,
                custom_prompt=custom_prompt
            )

            # 构造来源 HTML
            sources_html = f"\n\n<details><summary>📑 查看 {result['num_sources']} 个参考来源</summary>\n\n"
            for i, source in enumerate(result.get('sources', []), 1):
                sources_html += f"**[资料{i}]** {source['file_name']} (相似度: {source['similarity'] or 'N/A'})\n"
                sources_html += f"> {source['content']}\n\n"
            sources_html += "</details>"

            full_answer = result["answer"] + sources_html

            # 按顺序入账
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": full_answer})
            self.save_chat(history)
            
        return history, "", gr.update(choices=self.list_chats())
    
    def save_chat(self, history: list):
        """保存对话历史到 JSON（messages 格式）"""
        if not history:
            return

        if not self.current_chat_file:
            # 找第一条 user 消息作为文件名
            first_q = "chat"
            for msg in history:
                if msg.get("role") == "user":
                    first_q = msg["content"]
                    break

            safe_first_q = (
                first_q[:15]
                .replace(" ", "_")
                .replace("?", "")
                .replace("/", "")
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_chat_file = f"chat_{timestamp}_{safe_first_q}.json"

        filepath = self.conv_dir / self.current_chat_file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "history": history,
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

            
    def list_chats(self) -> List[str]:
        """列出所有保存的对话"""
        chats = list(self.conv_dir.glob("*.json"))
        # 按修改时间排序（最新的在前）
        chats.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return [c.name for c in chats]

    def load_chat(self, filename: str) -> tuple:
        """从JSON加载对话"""
        if not filename:
            return [], "", gr.update()
            
        self.current_chat_file = filename
        filepath = self.conv_dir / filename
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                history = data['history']
                return history, f"✅ 已载入对话: {filename}", gr.update(value=filename)
        except Exception as e:
            return [], f"❌ 载入失败: {str(e)}", gr.update()

    def handle_clear(self):
        """处理清空对话"""
        self.current_chat_file = None
        return [], "", gr.update(value=None)

    def refresh_kb_list(self):
        """获取并格式化知识库文档列表"""
        if not self.kb_loaded:
            return []
        sources = self.rag_system.get_knowledge_base_sources()
        # 转换为 DataFrame 格式需要的列表
        return [[s['file_name'], s['chunk_count']] for s in sources]
    
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
        text += f"**LLM模型**: {info['generator']['model_name']} ({info['generator'].get('provider', 'huggingface')})\n"
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
    
    def get_available_models(self) -> List[Tuple[str, str]]:
        """获取所有可用模型 (name as display, id as value)"""
        models = self.rag_system.config['models'].get('available_models', [])
        # 返回格式: [(Description, Value), ...]
        # Value format: "provider:model_name"
        return [(f"{m['desc']} ({m['provider']})", f"{m['provider']}:{m['name']}") for m in models]

    def handle_model_change(self, selected_value: str):
        """处理模型切换"""
        if not selected_value:
            return "❌ 无效选择", self.get_system_info()
            
        try:
            provider, model_name = selected_value.split(":", 1)
            msg = self.rag_system.reload_generator(model_name, provider)
            return msg, self.get_system_info()
        except Exception as e:
            return f"❌ 切换失败: {e}", self.get_system_info()
    


    def create_interface(self):
        
        self.custom_css = """
        #chat-main { 
            height: 700px !important; 
            overflow-y: auto; 
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            background-color: #f9fafb;
        }
        
        #input-row { 
            margin-top: 10px;
        }
        
        #kb-table { 
            max-height: 300px !important; 
            overflow-y: auto; 
        }
        
        footer { visibility: hidden !important; }
        """

        with gr.Blocks(title="RAG 智能助手") as demo:
            
            with gr.Row():
                
                #左侧历史文件
                with gr.Column(scale=2, min_width=250):
                    gr.Markdown("### 🗂️ 历史与文件")
                    
                    # 历史记录
                    with gr.Group():
                        with gr.Row():
                            new_chat_btn = gr.Button("➕ 新对话", variant="primary", size="sm")
                            refresh_chats_btn = gr.Button("🔄", size="sm", scale=0)
                        
                        chat_selector = gr.Dropdown(
                            label="历史记录",
                            choices=self.list_chats(),
                            interactive=True,
                            allow_custom_value=True,
                            container=False
                        )

                    gr.Markdown("---")
                    
                    # 知识库管理
                    gr.Markdown("#### 📁 知识库")
                    file_upload = gr.File(
                        label="上传文件",
                        file_count="multiple",
                        file_types=[".pdf", ".txt", ".docx", ".md"]
                    )
                    
                    with gr.Row():
                        upload_btn = gr.Button("📤 上传", size="sm")
                        build_btn = gr.Button("🔨 重建库", size="sm")
                    
                    upload_status = gr.Textbox(show_label=False, placeholder="等待操作...", interactive=False, lines=1)
                    
                    kb_table = gr.Dataframe(
                        headers=["文件名", "切片"],
                        datatype=["str", "number"],
                        value=self.refresh_kb_list(),
                        interactive=False,
                        elem_id="kb-table",
                        wrap=True
                    )
                    refresh_kb_btn = gr.Button("🔄 刷新列表", size="sm")


                # 中间核心对话区
                with gr.Column(scale=6):
                    # 聊天框
                    chatbot = gr.Chatbot(
                        label=None,
                        show_label=False,
                        elem_id="chat-main"
                    )
                    
                    # 输入区
                    with gr.Row(elem_id="input-row"):
                        question_input = gr.Textbox(
                            show_label=False,
                            placeholder="输入您的问题... (Shift+Enter 换行)",
                            scale=8,
                            lines=1,
                            max_lines=10,
                            autofocus=True,
                            container=False
                        )
                        submit_btn = gr.Button("发送", variant="primary", scale=1, min_width=60)


                #右侧设置信息
                with gr.Column(scale=2, min_width=250):
                    gr.Markdown("### ⚙️ 设置与监控")
                    
                    # 参数设置
                    with gr.Group():
                        gr.Markdown("#### 检索设置")
                        top_k_slider = gr.Slider(
                            minimum=1, maximum=10, value=5, step=1, 
                            label="Top-K"
                        )
                    
                    with gr.Accordion("📝 Prompt 设置", open=True):
                        prompt_input = gr.Textbox(
                            show_label=False,
                            value=self.rag_system.generator.PROMPT_TEMPLATE,
                            lines=10,
                            placeholder="System Prompt..."
                        )
                        reset_prompt_btn = gr.Button("恢复默认", size="sm")

                    gr.Markdown("---")
                    
                    # 模型选择
                    gr.Markdown("#### 🤖 模型切换")
                    
                    # 获取当前模型作为默认值
                    current_llm = self.rag_system.config['models']['llm']
                    current_provider = current_llm.get('provider', 'huggingface')
                    if current_provider == 'ollama':
                         current_val = f"ollama:{current_llm.get('ollama', {}).get('model', '')}"
                    else:
                         current_val = f"huggingface:{current_llm.get('name', '')}"

                    model_dropdown = gr.Dropdown(
                        label="选择模型",
                        choices=self.get_available_models(),
                        value=current_val,
                        interactive=True,
                        container=False
                    )
                    model_status = gr.Textbox(show_label=False, placeholder="模型状态...", lines=1, interactive=False)

                    gr.Markdown("---")

                    # 系统信息
                    gr.Markdown("#### ℹ️ 系统状态")
                    info_output = gr.Markdown(elem_id="sys_info")
                    refresh_info_btn = gr.Button("刷新状态", size="sm")

            #事件绑定逻辑
            new_chat_btn.click(fn=self.handle_clear, outputs=[chatbot, question_input, chat_selector])

            submit_triggers = [question_input.submit, submit_btn.click]
            for trigger in submit_triggers:
                trigger(
                    fn=self.answer_question,
                    inputs=[question_input, chatbot, top_k_slider, prompt_input],
                    outputs=[chatbot, question_input, chat_selector]
                )

            reset_prompt_btn.click(fn=lambda: self.rag_system.generator.PROMPT_TEMPLATE, outputs=prompt_input)
            
            chat_selector.change(fn=self.load_chat, inputs=chat_selector, outputs=[chatbot, upload_status, chat_selector])
            refresh_chats_btn.click(fn=lambda: gr.update(choices=self.list_chats()), outputs=chat_selector)

            upload_btn.click(fn=self.upload_files, inputs=file_upload, outputs=upload_status).then(fn=self.refresh_kb_list, outputs=kb_table)
            build_btn.click(fn=self.build_kb_from_directory, outputs=upload_status).then(fn=self.refresh_kb_list, outputs=kb_table)
            refresh_kb_btn.click(fn=self.refresh_kb_list, outputs=kb_table)
            
            build_btn.click(fn=self.build_kb_from_directory, outputs=upload_status).then(fn=self.refresh_kb_list, outputs=kb_table)
            refresh_kb_btn.click(fn=self.refresh_kb_list, outputs=kb_table)
            
            model_dropdown.change(
                fn=self.handle_model_change,
                inputs=model_dropdown,
                outputs=[model_status, info_output]
            )
            
            demo.load(self.get_system_info, outputs=info_output)
            refresh_info_btn.click(fn=self.get_system_info, outputs=info_output)

        return demo


def main():
    """主函数"""
    # 初始化
    app = RAGWebApp()

    demo = app.create_interface()
    
    # 启动服务
    web_config = app.rag_system.config['web']
    demo.launch(
        server_name=web_config['host'],
        server_port=web_config['port'],
        share=web_config['share'],
        theme=gr.themes.Soft(),
        css=app.custom_css
    )


if __name__ == "__main__":
    main()
