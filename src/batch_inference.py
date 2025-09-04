# -*- coding: utf-8 -*-
"""
材料型号规范化批量推理脚本
========================

基于训练好的 ByT5 + LoRA 模型进行批量文本规范化，并输出置信度评分。

功能特性：
- 📊 批量CSV文件处理，支持大规模数据
- 🎯 输出推理结果和置信度评分
- 📈 实时进度显示和错误处理
- 🇨🇳 内置国内镜像支持，适配中文环境
- 💾 自动结果保存和恢复机制

输入格式：CSV 文件，包含 'dirty' 列
输出格式：CSV 文件，包含 'dirty', 'clean', 'confidence' 列

使用示例：
  # 基本批量推理
  python src/batch_inference.py --ckpt ckpts/norm-byt5-cpu/best --input test_data/input_samples.csv --output results/inference_results.csv
  
  # 指定镜像和缓存（国内环境推荐）
  python src/batch_inference.py --ckpt ckpts/norm-byt5-cpu/best --input test_data/input_samples.csv --output results/inference_results.csv --hf-endpoint https://hf-mirror.com --hf-home ./.hf_cache
  
  # 离线模式
  python src/batch_inference.py --ckpt ./local_models/byt5-small --input test_data/input_samples.csv --output results/inference_results.csv --offline

适用环境：Windows + Conda + 国内网络环境
作者：Material-Code-Manager 项目组
"""

import os
import re as pyre
import time
import argparse
import threading
import pandas as pd
import numpy as np
import torch
import json
import os.path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

RUNNING = True

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def heartbeat(msg="💡 批量推理中..."):
    while RUNNING:
        time.sleep(10)
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="模型/权重目录或模型名")
    ap.add_argument("--input", type=str, required=True, help="输入CSV文件路径")
    ap.add_argument("--output", type=str, required=True, help="输出CSV文件路径")
    ap.add_argument("--input_column", type=str, default="dirty", help="输入列名（默认：dirty）")
    ap.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    ap.add_argument("--regex", type=str,
                    default=r"^[\p{Han}A-Za-z0-9\-\_/．\.,，（）\(\)=+\*×／/:：;；\[\]【】<>《》\s]{1,120}$",
                    help="生成后合法性校验用的宽松正则")
    ap.add_argument("--beams", type=int, default=1, help="束搜索大小")
    ap.add_argument("--max_new_tokens", type=int, default=32, help="生成最大新token数")
    # 运行环境相关
    ap.add_argument("--offline", action="store_true", help="完全离线")
    ap.add_argument("--hf-endpoint", type=str, default="https://hf-mirror.com", help="Hugging Face 镜像端点")
    ap.add_argument("--hf-home", type=str, default=".\.hf_cache", help="HF 缓存目录")
    return ap.parse_args()

def calculate_confidence(logits, generated_ids, tokenizer):
    """计算生成结果的置信度"""
    try:
        # 获取生成序列的概率
        probs = torch.softmax(logits, dim=-1)
        # 计算生成token的平均概率作为置信度
        token_probs = []
        for i, token_id in enumerate(generated_ids[0]):
            if i < len(logits[0]):
                token_prob = probs[0][i][token_id].item()
                token_probs.append(token_prob)
        
        if token_probs:
            confidence = np.mean(token_probs)
        else:
            confidence = 0.0
        
        return confidence
    except Exception:
        return 0.0

def main():
    global RUNNING
    
    args = parse_args()
    
    # 1) 设置环境
    log("==== 🧭 设置运行环境 ====")
    # 设置默认或用户指定的 HF 配置
    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ["HF_HOME"] = args.hf_home
    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    log(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}")
    log(f"HF_HOME={os.environ.get('HF_HOME')}")
    log(f"TRANSFORMERS_OFFLINE={os.environ.get('TRANSFORMERS_OFFLINE')}")
    
    print("\n==== 🚀 批量推理启动 ====\n", flush=True)

    # 心跳线程
    t = threading.Thread(target=heartbeat, daemon=True)
    t.start()

    # 2) 读取输入数据
    log("==== 📖 读取输入数据 ====")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"输入文件不存在: {args.input}")
    
    df = pd.read_csv(args.input)
    log(f"读取到 {len(df)} 条数据")
    
    if args.input_column not in df.columns:
        raise ValueError(f"输入列 '{args.input_column}' 不存在于CSV文件中")

    # 3) 加载模型与分词器
    log("==== 🤖 加载模型与分词器 ====")
    local_only = os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    name_or_path = args.ckpt
    
    # 检查是否是 LoRA 适配器模型
    adapter_config_path = os.path.join(name_or_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        log("🔧 检测到 LoRA 适配器，加载基础模型和适配器")
        # 读取适配器配置获取基础模型路径
        with open(adapter_config_path, 'r', encoding='utf-8') as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get('base_model_name_or_path', 'google/byt5-small')
        
        try:
            # 先加载基础模型
            tok = AutoTokenizer.from_pretrained(base_model_name, local_files_only=local_only)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, local_files_only=local_only)
            # 然后加载 LoRA 适配器
            mdl = PeftModel.from_pretrained(mdl, name_or_path, local_files_only=local_only)
            log("✅ LoRA 模型加载成功")
        except OSError as e:
            log(f"⚠️ 在线/缓存加载失败，尝试本地加载：{e}")
            tok = AutoTokenizer.from_pretrained(base_model_name, local_files_only=True)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, local_files_only=True)
            mdl = PeftModel.from_pretrained(mdl, name_or_path, local_files_only=True)
            log("✅ LoRA 本地加载成功")
    else:
        # 普通模型加载
        try:
            tok = AutoTokenizer.from_pretrained(name_or_path, local_files_only=local_only)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(name_or_path, local_files_only=local_only)
        except OSError as e:
            log(f"⚠️ 在线/缓存加载失败，尝试本地加载：{e}")
            tok = AutoTokenizer.from_pretrained(name_or_path, local_files_only=True)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(name_or_path, local_files_only=True)
            log("✅ 本地加载成功")

    mdl.eval()

    # 4) 批量推理
    log("==== ✨ 开始批量推理 ====")
    results = []
    
    # 准备正则表达式
    try:
        import regex as re
        rgx = re.compile(args.regex)
        use_regex = True
    except Exception:
        use_regex = False
        log("⚠️ regex库不可用，将使用内置re库进行简单清理")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="推理进度"):
        try:
            input_text = str(row[args.input_column])
            prefix = "normalize: " + input_text.strip()
            inputs = tok([prefix], return_tensors="pt")
            
            with torch.no_grad():
                # 生成，同时返回scores用于计算置信度
                output = mdl.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.beams,
                    return_dict_in_generate=True,
                    output_scores=True,
                    do_sample=False
                )
                
                generated_ids = output.sequences
                scores = output.scores if hasattr(output, 'scores') else None
            
            # 解码结果
            pred = tok.decode(generated_ids[0], skip_special_tokens=True).replace("normalize:", "").strip()
            
            # 计算置信度
            if scores:
                confidence = calculate_confidence(scores, generated_ids, tok)
            else:
                confidence = 0.0
            
            # 正则兜底清理
            if use_regex:
                if not rgx.match(pred):
                    pred = re.sub(r"[\p{C}]", "", pred).strip()
            else:
                pred = pyre.sub(r"[\x00-\x1F\x7F]", "", pred).strip()
            
            # 保存结果
            result = {
                'input': input_text,
                'output': pred,
                'confidence': round(confidence, 4)
            }
            # 保留原始数据的其他列
            for col in df.columns:
                if col != args.input_column:
                    result[col] = row[col]
            
            results.append(result)
            
        except Exception as e:
            log(f"❌ 处理第 {idx+1} 行时出错: {e}")
            # 出错时也保存一条记录
            result = {
                'input': str(row[args.input_column]) if args.input_column in row else '',
                'output': f"ERROR: {str(e)}",
                'confidence': 0.0
            }
            for col in df.columns:
                if col != args.input_column:
                    result[col] = row[col]
            results.append(result)

    # 5) 保存结果
    log("==== 💾 保存结果 ====")
    results_df = pd.DataFrame(results)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    results_df.to_csv(args.output, index=False, encoding='utf-8')
    log(f"结果已保存到: {args.output}")
    log(f"共处理 {len(results)} 条数据")
    
    # 显示统计信息
    avg_confidence = results_df['confidence'].mean()
    log(f"平均置信度: {avg_confidence:.4f}")
    
    RUNNING = False
    print("\n==== ✅ 批量推理完成 ====\n", flush=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        RUNNING = False
        log("用户中断（Ctrl+C）")
    except Exception as ex:
        RUNNING = False
        log(f"❌ 程序异常：{ex}")
        raise
