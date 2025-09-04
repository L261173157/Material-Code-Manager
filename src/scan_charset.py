# -*- coding: utf-8 -*-
"""
数据集字符分析工具
=================

分析CSV数据集中的字符分布，帮助优化模型配置和数据预处理。

主要功能：
- 📊 统计所有唯一字符及其频次
- 🔍 显示Unicode字符名称和编码
- 📈 按频次排序输出字符分析
- 🎯 帮助调试字符编码问题

使用示例：
  # 分析训练数据
  python src/scan_charset.py data/train_pairs.csv
  
  # 分析测试数据
  python src/scan_charset.py test_data/input_samples.csv

适用环境：Windows + CSV数据文件
作者：Material-Code-Manager 项目组
"""

import csv
import sys
import collections
import unicodedata

# 解析命令行参数
path = sys.argv[1] if len(sys.argv) > 1 else "data/train_pairs.csv"
print(f"📊 分析文件: {path}")

# 字符频次统计
chars = collections.Counter()

# 读取CSV文件并统计字符
print("🔍 正在扫描字符...")
with open(path, "r", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        # 分析dirty和clean两列的字符
        for col in ("dirty", "clean"):
            if col in row:
                for ch in row[col]:
                    if not ch.strip():  # 跳过纯空白字符的统计显示
                        continue
                    chars[ch] += 1

# 输出统计结果
print(f"\n✅ 字符分析完成！")
print(f"📈 唯一字符数: {len(chars)}")
print(f"📋 字符详细分布:\n")

print(f"{'字符':<8} {'Unicode':<10} {'字符名称':<45} {'频次':<8}")
print("-" * 75)

# 按频次降序输出字符信息
for ch, cnt in chars.most_common():
    # 获取Unicode字符信息
    name = unicodedata.name(ch, "N/A")
    code = f"U+{ord(ch):04X}"
    
    # 格式化输出
    print(f"{repr(ch):<8} {code:<10} {name:<45} {cnt:<8}")

print(f"\n🎯 分析建议:")
print(f"   - 检查是否有异常字符（如控制字符、特殊符号）")
print(f"   - 确认全角/半角字符的一致性")
print(f"   - 验证中文字符编码是否正确")
