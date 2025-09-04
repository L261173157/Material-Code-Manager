# -*- coding: utf-8 -*-
"""
词典吸附后处理工具
=================

将模型预测结果吸附到最相近的标准词典条目，提升输出规范性。

主要功能：
- 🎯 基于相似度的词典匹配
- 📊 可配置的相似度阈值
- 🔍 使用 RapidFuzz 快速模糊匹配
- 📝 支持标准型号词典文件

使用示例：
  # 基本用法
  python src/dict_snap.py --pred "GB／T3452.1-1×1.8" --dict data/standard_vocab.txt --cut 90.0
  
  # 降低匹配阈值
  python src/dict_snap.py --pred "预测结果" --dict data/standard_vocab.txt --cut 80.0

适用环境：Windows + 标准型号词典
作者：Material-Code-Manager 项目组
"""

import argparse
from rapidfuzz import process

def snap_to_dict(pred: str, dict_path: str, score_cut: float = 90.0) -> str:
    """
    将预测结果吸附到最相近的标准型号库
    
    参数:
        pred: 模型预测结果
        dict_path: 标准词典文件路径
        score_cut: 相似度阈值（0-100）
    
    返回:
        吸附后的标准型号（如未匹配则返回原预测）
    """
    # 读取标准词典
    with open(dict_path, "r", encoding="utf-8") as f:
        items = [line.strip() for line in f if line.strip()]
    
    # 模糊匹配最相似条目
    match = process.extractOne(pred, items, score_cutoff=score_cut)
    
    if match:
        best, score, _ = match
        print(f"✅ 匹配成功: '{pred}' -> '{best}' (相似度: {score:.1f})")
        return best
    else:
        print(f"❌ 未找到匹配: '{pred}' (阈值: {score_cut})")
        return pred

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="词典吸附后处理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python src/dict_snap.py --pred "GB／T3452.1-1×1.8" --dict data/standard_vocab.txt --cut 90.0
  python src/dict_snap.py --pred "预测结果" --dict data/standard_vocab.txt --cut 85.0
        """
    )
    ap.add_argument("--pred", type=str, required=True,
                   help="模型预测结果")
    ap.add_argument("--dict", type=str, required=True,
                   help="标准词典文件路径")
    ap.add_argument("--cut", type=float, default=90.0,
                   help="相似度阈值 (0-100，默认90.0)")
    args = ap.parse_args()
    
    result = snap_to_dict(args.pred, args.dict, args.cut)
    print(f"🎯 最终结果: {result}")
