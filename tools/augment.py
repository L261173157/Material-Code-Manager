# -*- coding: utf-8 -*-
"""
数据增强工具
============

通过字符替换和空格扰动来增强训练数据，提升模型泛化能力。

主要功能：
- 🔄 字符级随机替换（全角/半角转换）
- 📝 随机空格插入和删除
- 📊 可配置的增强倍数
- 🎯 专门针对材料型号文本优化

使用示例：
  # 基本用法（2倍增强）
  python tools/augment.py data/train_pairs.csv data/train_pairs_augmented.csv 2
  
  # 5倍增强
  python tools/augment.py data/train_pairs.csv data/train_pairs_5x.csv 5

适用环境：Windows + 中文材料型号数据
作者：Material-Code-Manager 项目组
"""

import csv, random, sys

# 字符替换映射表（全角/半角，常见符号）
subs = [
    ("x", "×"), ("×", "x"),                    # 乘号替换
    ("/", "／"), ("／", "/"),                   # 斜杠替换
    ("-", "－"), ("－", "-"), ("-", "–"), ("–", "-"),  # 连字符替换
    (".", "．"), ("．", "."),                   # 句号替换
    ("(", "（"), ("（", "("), (")", "）"), ("）", ")"),  # 括号替换
    (",", "，"), ("，", ","),                   # 逗号替换
    ("GB-T", "GB／T"), ("GB／T", "GB-T"),      # 标准号格式替换
]

# 设置随机种子确保可重现
random.seed(42)

def perturb(s: str, p=0.3):
    """
    对输入字符串进行随机扰动
    
    参数:
        s: 输入字符串
        p: 扰动概率 (0.0-1.0)
    
    返回:
        扰动后的字符串
    """
    t = s
    
    # 字符替换扰动
    for a, b in subs:
        if random.random() < p:
            t = t.replace(a, b)
    
    # 随机插入空格
    if random.random() < p:
        i = random.randrange(0, len(t)+1)
        t = t[:i] + " " + t[i:]
    
    # 随机删除空格
    if random.random() < p and " " in t:
        t = t.replace("  ", " ").replace(" ", "", 1)
    
    return t

# ============ 主程序 ============

# 解析命令行参数
inp = sys.argv[1] if len(sys.argv) > 1 else "data/train_pairs.csv"
outp = sys.argv[2] if len(sys.argv) > 2 else "data/train_pairs_augmented.csv"
mult = int(sys.argv[3]) if len(sys.argv) > 3 else 2  # 每条clean数据的增强倍数

print(f"📊 输入文件: {inp}")
print(f"📝 输出文件: {outp}")
print(f"🔄 增强倍数: {mult}")

# 读取原始数据并进行增强
rows = []
with open(inp, "r", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        clean = row["clean"].strip()
        
        # 保留原始数据
        rows.append({"dirty": row["dirty"], "clean": clean})
        
        # 生成增强数据
        for _ in range(mult):
            augmented_dirty = perturb(clean)
            rows.append({"dirty": augmented_dirty, "clean": clean})

# 保存增强后的数据
with open(outp, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["dirty","clean"])
    w.writeheader()
    w.writerows(rows)

print(f"✅ 数据增强完成！")
print(f"📈 总行数: {len(rows)}")
print(f"💾 已保存到: {outp}")
