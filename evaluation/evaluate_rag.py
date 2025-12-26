import os
import sys
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict

# 添加项目根目录到路径
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / "src"))

import yaml
from src.rag_system import RAGSystem

# Ragas 相关导入
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

def run_evaluation(config_path="config.yaml", dataset_path="evaluation/test_dataset.json"):
    """运行RAG系统并获取评测用的输出"""
    
    # 1. 初始化系统
    print("正在初始化 RAG 系统以进行评测...")
    rag = RAGSystem(config_path)
    rag.load_knowledge_base()
    
    # 2. 加载测试集
    with open(dataset_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    evaluation_results = []
    
    print(f"开始评测，共 {len(test_data)} 个问题...")
    
    for item in test_data:
        question = item['question']
        ground_truth = item['ground_truth']
        
        print(f"\n提问: {question}")
        
        # 3. 运行查询
        result = rag.query(question)
        
        # 4. 获取检索到的上下文
        # Ragas 需要的 context 是一个列表
        # 获取检索到的原始内容列表
        retrieved_docs = rag.retriever.retrieve(question)
        contexts = [doc.page_content for doc, score in retrieved_docs]
        
        evaluation_results.append({
            "question": question,
            "answer": result['answer'],
            "contexts": contexts,
            "ground_truth": ground_truth
        })
    
    # 保存结果为 JSON 以备 Ragas 使用
    output_path = "evaluation/eval_output.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        
    print(f"\n✓ 评测数据生成完成，已保存至: {output_path}")
    return evaluation_results

def run_ragas_metrics(config_path="config.yaml", eval_output_path="evaluation/eval_output.json"):
    """使用 Ragas 计算得分"""
    print("\n" + "="*50)
    print("开始 Ragas 打分阶段...")
    print("="*50)
    
    # 1. 加载配置获取 API 信息
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    api_config = config['evaluation']['api']
    
    # 2. 初始化用于评价的大模型 (DeepSeek via ChatOpenAI)
    # Ragas 如果不显式传入 llm，默认会寻找 OpenAI API Key
    llm = ChatOpenAI(
        model=api_config['model'],
        openai_api_key=api_config['api_key'],
        openai_api_base=api_config['base_url']
    )
    
    # 3. 初始化 Embedding 模型 (Ragas 文档相关性计算需要)
    # 直接复用项目自带的 BGE 模型
    embed_config = config['models']['embedding']
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        encode_kwargs={"normalize_embeddings": True},
        model_kwargs={"device": "cuda"}
    )

    # 4. 加载生成的评测数据
    with open(eval_output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换为 Ragas 需要的 Dataset 格式
    from datasets import Dataset
    dataset = Dataset.from_list(data)
    
    # 5. 执行评测
    print(f"正在通过 {api_config['model']} 进行全维度打分，请稍候...")
    metrics = [
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    ]
    
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings
    )
    
    # 6. 展示并保存结果
    print("\n" + "*"*50)
    print("RAGS 评测得分总览:")
    print(result)
    print("*"*50)
    
    # 保存结果到 CSV 方便查阅
    df = result.to_pandas()
    result_csv = "evaluation/evaluation_results.csv"
    df.to_csv(result_csv, index=False, encoding='utf_8_sig')
    print(f"\n✓ 详细得分已汇总至: {result_csv}")
    
    return result

if __name__ == "__main__":
    # 第一步：让系统跑一遍，收集答案和上下文
    run_evaluation()
    
    # 第二步：接入 Ragas 进行打分
    run_ragas_metrics()
