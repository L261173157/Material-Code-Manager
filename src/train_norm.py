# -*- coding: utf-8 -*-
"""
材料型号规范化模型训练脚本
=========================

基于 ByT5 + LoRA 的序列到序列文本规范化模型训练程序。

特性：
- 🚀 字符级序列建模，无需分词
- 💡 LoRA 低秩适应，大幅降低训练参数
- 🇨🇳 支持国内 HuggingFace 镜像
- 🔄 自动模型保存和断点恢复
- 📊 详细训练日志和进度监控

适用环境：Windows + Conda + 国内网络环境

作者：Material-Code-Manager 项目组
"""ing: utf-8 -*-
"""
train_norm.py  —— 简洁版（Transformers 4.56.0+）
特性：
- 先读 config 写入 HF_ENDPOINT/HF_HOME（镜像/缓存），再导入 transformers
- --offline 支持完全离线(local_files_only + TRANSFORMERS_OFFLINE=1)
- 心跳日志：每 10 秒打印“程序仍在运行”
- 清晰阶段标志：加载数据 / 加载模型 / 训练 / 评估 / 保存
"""

import os
import sys
import time
import yaml
import argparse
import threading
import numpy as np

# ============ 工具函数 ============

RUNNING = True  # 全局运行标志

def log(msg: str):
    """输出带时间戳的日志信息"""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def heartbeat(msg="💡 训练进行中，请稍候..."):
    """后台心跳线程，每10秒输出一次状态"""
    while RUNNING:
        time.sleep(10)
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def parse_args():
    """解析命令行参数"""
    p = argparse.ArgumentParser(
        description="材料型号规范化模型训练程序",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python src/train_norm.py --config configs/config.yaml --csv data/train_pairs.csv --epochs 10
  python src/train_norm.py --config configs/config.yaml --offline  # 离线模式
        """
    )
    p.add_argument("--config", type=str, default="configs/config.yaml",
                   help="配置文件路径 (默认: configs/config.yaml)")
    p.add_argument("--csv", type=str, default=None,
                   help="训练数据CSV文件路径，覆盖配置文件中的设置")
    p.add_argument("--epochs", type=int, default=None,
                   help="训练轮数，覆盖配置文件中的设置")
    p.add_argument("--offline", action="store_true",
                   help="完全离线运行模式（需要预先下载模型）")
    return p.parse_args()

def load_config_and_set_env(path):
    """
    加载配置文件并设置环境变量
    注意：必须在导入transformers之前调用，以正确设置HF镜像
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # 设置环境变量（HuggingFace镜像配置等）
    for k, v in (cfg.get("env") or {}).items():
        os.environ[str(k)] = str(v)
        log(f"环境变量设置: {k}={v}")
    
    return cfg

# ============ 主程序入口 ============

# 1. 解析参数并加载配置（在导入transformers之前完成）
args = parse_args()
log("🚀 开始初始化训练环境...")
cfg = load_config_and_set_env(args.config)

# 2. 参数覆盖
if args.csv:
    cfg["data"]["csv_path"] = args.csv
    log(f"📊 训练数据路径: {args.csv}")
if args.epochs:
    cfg["train"]["num_train_epochs"] = args.epochs
if args.offline:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

log(f"HF_ENDPOINT = {os.environ.get('HF_ENDPOINT')}")
log(f"HF_HOME     = {os.environ.get('HF_HOME')}")
log(f"TRANSFORMERS_OFFLINE = {os.environ.get('TRANSFORMERS_OFFLINE')}")

# 现在再导入依赖（确保 env 已生效）
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments, Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model
from rapidfuzz.distance import Levenshtein

# ============ 指标 ============
def compute_metrics_builder(tokenizer, prefix):
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
        label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
        pred_texts = [t.replace(prefix, "").strip() for t in pred_texts]
        exact = sum(int(p == y) for p, y in zip(pred_texts, label_texts))
        cer_sum = sum(Levenshtein.distance(p, y) / max(1, len(y)) for p, y in zip(pred_texts, label_texts))
        n = max(1, len(pred_texts))
        return {"exact_match": exact / n, "cer": cer_sum / n}
    return compute_metrics

# ============ 主流程 ============
def main():
    global RUNNING
    print("\n==== 🚀 程序启动 ====\n", flush=True)

    # 心跳
    t = threading.Thread(target=heartbeat, args=("💡 正在运行，请稍候...",), daemon=True)
    t.start()

    # 1) 数据
    csv_path = cfg["data"]["csv_path"]
    assert os.path.exists(csv_path), f"CSV not found: {csv_path}"
    log("==== 📊 加载数据集 ====")
    raw = load_dataset("csv", data_files=csv_path, split="train")
    raw = raw.train_test_split(test_size=cfg["data"]["test_size"], seed=cfg["train"]["seed"])
    train_valid = raw["train"].train_test_split(test_size=cfg["data"]["valid_size_from_train"], seed=cfg["train"]["seed"])
    train_ds, valid_ds, test_ds = train_valid["train"], train_valid["test"], raw["test"]
    log(f"数据切分：train={len(train_ds)}  valid={len(valid_ds)}  test={len(test_ds)}")

    # 2) 模型
    log("==== 🤖 加载模型与分词器 ====")
    local_only = bool(os.environ.get("TRANSFORMERS_OFFLINE") == "1")
    name_or_path = cfg["model"]["name"]
    try:
        tok = AutoTokenizer.from_pretrained(name_or_path, local_files_only=local_only)
        model = AutoModelForSeq2SeqLM.from_pretrained(name_or_path, local_files_only=local_only)
    except OSError as e:
        log(f"⚠️ 在线/缓存加载失败，尝试本地加载：{e}")
        tok = AutoTokenizer.from_pretrained(name_or_path, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(name_or_path, local_files_only=True)
        log("✅ 本地加载成功")

    if cfg["model"]["lora"]["enable"]:
        log("==== 🧩 挂载 LoRA 适配层 ====")
        lcfg = cfg["model"]["lora"]
        model = get_peft_model(model, LoraConfig(
            r=lcfg["r"], lora_alpha=lcfg["alpha"],
            target_modules=lcfg["target_modules"], lora_dropout=lcfg["dropout"],
            bias="none", task_type="SEQ_2_SEQ_LM"
        ))
        log("LoRA 挂载完成")

    # 3) 预处理映射
    log("==== 🧪 准备数据映射 ====")
    MAX_LEN = cfg["train"]["max_length"]; PREFIX = cfg["train"]["prefix"]
    def preprocess(ex):
        inp = (ex.get("dirty") or "").strip()
        tgt = (ex.get("clean") or "").strip()
        enc = tok(PREFIX + inp, max_length=MAX_LEN, truncation=True)
        with tok.as_target_tokenizer():
            labels = tok(tgt, max_length=MAX_LEN, truncation=True)
        enc["labels"] = labels["input_ids"]
        return enc
    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
    valid_ds = valid_ds.map(preprocess, remove_columns=valid_ds.column_names)
    test_ds_prep = test_ds.map(preprocess, remove_columns=test_ds.column_names)
    log("数据映射完成")

    # 4) 训练
    log("==== 🎯 开始训练 ====")
    ta = Seq2SeqTrainingArguments(
        output_dir=cfg["train"]["save_dir"],
        per_device_train_batch_size=cfg["train"]["batch_size"],
        per_device_eval_batch_size=cfg["train"]["eval_batch_size"],
        gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
        learning_rate=float(cfg["train"]["learning_rate"]),
        num_train_epochs=cfg["train"]["num_train_epochs"],
        weight_decay=float(cfg["train"]["weight_decay"]),
        logging_steps=cfg["train"]["logging_steps"],
        eval_strategy=cfg["train"]["eval_strategy"],  # 修改这里：evaluation_strategy -> eval_strategy
        save_strategy=cfg["train"]["save_strategy"],
        load_best_model_at_end=True,
        metric_for_best_model=cfg["train"]["metric_for_best"],
        greater_is_better=cfg["train"]["greater_is_better"],
        predict_with_generate=True,
        fp16=cfg["train"]["fp16"],
        label_smoothing_factor=float(cfg["train"].get("label_smoothing", 0.0)),
        seed=cfg["train"]["seed"],
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=ta,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
        tokenizer=tok,
        compute_metrics=compute_metrics_builder(tok, PREFIX),
    )

    trainer.train()
    log("✅ 训练完成")

    # 5) 评估
    log("==== 📈 测试集评估 ====")
    test_metrics = trainer.evaluate(test_ds_prep)
    print("TEST:", test_metrics, flush=True)

    # 6) 保存
    log("==== 💾 保存最佳模型 ====")
    best_dir = os.path.join(cfg["train"]["save_dir"], "best")
    os.makedirs(best_dir, exist_ok=True)
    trainer.save_model(best_dir)
    tok.save_pretrained(best_dir)
    log(f"模型已保存：{best_dir}")

    RUNNING = False
    print("\n==== ✅ 程序结束 ====\n", flush=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        RUNNING = False
        log("用户中断（Ctrl+C）")
        sys.exit(130)
    except Exception as ex:
        RUNNING = False
        log(f"❌ 程序异常：{ex}")
        raise
