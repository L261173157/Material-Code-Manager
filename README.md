# 材料型号规范化系统

基于 **ByT5 + LoRA** 的材料型号文本规范化系统，用于将不规范的型号文本自动转换为标准格式。

## 项目简介

本系统采用字符级序列到序列模型（ByT5）结合低秩适应（LoRA）技术，实现高效的文本规范化：
- 🎯 **高精度**：基于Transformer架构的字符级建模
- 💡 **低成本**：LoRA技术显著降低训练参数量
- 🚀 **易部署**：支持单个推理和批量处理
- 🇨🇳 **本地化**：专为中文环境优化，内置国内镜像支持

## 项目结构

```
Material-Code-Manager/
├── .conda/                          # Conda虚拟环境
├── .hf_cache/                       # HuggingFace模型缓存
├── configs/
│   └── config.yaml                  # 训练配置文件
├── data/
│   └── train_pairs.csv             # 训练数据（不规范→标准）
├── test_data/                       # 测试数据目录
│   ├── input_samples.csv           # 测试输入样本
│   └── expected_results.csv        # 预期结果
├── results/                         # 推理结果目录
├── ckpts/norm-byt5-cpu/best/       # 训练好的模型
├── src/                            # 核心代码
│   ├── train_norm.py              # 模型训练脚本
│   ├── constrained_generate.py    # 单个文本推理
│   ├── batch_inference.py         # 批量推理（带置信度）
│   ├── dict_snap.py              # 结果后处理（词典吸附）
│   ├── scan_charset.py           # 数据集字符分析
│   └── utils.py                  # 工具函数
├── tools/
│   └── augment.py                 # 数据增强工具
├── scripts/
│   └── train.sh                   # 训练脚本
├── requirements.txt               # Python依赖
└── README.md                      # 项目说明
```

## 快速开始

### 1. 环境准备（Windows）

```powershell
# 创建Conda环境
conda create -p .conda python=3.11 -y
conda activate .\.conda

# 安装依赖（使用清华镜像）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

### 2. 数据准备

训练数据格式（CSV，UTF-8编码）：
```csv
dirty,clean
"1×1.8  GB-T 3452.1-1992","GB／T3452.1-1×1.8"
"8X5-10米","8×5-10米"
"ACQ25x25SB","ACQ25×25SB"
```

### 3. 模型训练

```powershell
# 激活环境
conda activate .\.conda

# 开始训练（约15-30分钟，CPU可用）
python src\train_norm.py --config configs\config.yaml --csv data\train_pairs.csv --epochs 10
```

### 4. 模型推理

#### 单个文本推理
```powershell
python src\constrained_generate.py --ckpt ckpts\norm-byt5-cpu\best --text "1×1.8  GB-T 3452.1-1992"
```

#### 批量推理（带置信度）
```powershell
python src\batch_inference.py --ckpt ckpts\norm-byt5-cpu\best --input test_data\input_samples.csv --output results\inference_results.csv
```

## 详细使用说明

### 数据增强
```powershell
# 对训练数据进行增强（2倍扩充）
python tools\augment.py data\train_pairs.csv data\train_pairs_augmented.csv 2
```

### 字符集分析
```powershell
# 分析数据集字符分布
python src\scan_charset.py data\train_pairs.csv
```

### 结果后处理
```powershell
# 将预测结果吸附到标准词典
python src\dict_snap.py --pred "GB／T3452.1-1×1.8" --dict data\standard_vocab.txt --cut 90.0
```

## 配置说明

主要配置文件 `configs/config.yaml`：
```yaml
model_name: "google/byt5-small"  # 基础模型
max_length: 128                  # 最大序列长度
learning_rate: 5e-4             # 学习率
batch_size: 8                   # 批次大小
epochs: 10                      # 训练轮数
lora_r: 8                       # LoRA秩
lora_alpha: 32                  # LoRA缩放参数
```

## 性能指标

- **训练时间**：约20分钟（CPU，62样本×10轮）
- **模型大小**：~300MB（包含LoRA权重）
- **推理速度**：~1-2秒/样本（CPU）
- **内存占用**：~2GB（训练时）

## 技术特点

1. **字符级建模**：直接处理字符序列，避免分词问题
2. **LoRA微调**：只训练0.1%参数，显著降低计算成本
3. **国内优化**：默认使用HuggingFace国内镜像
4. **置信度评估**：提供预测置信度分数
5. **批量处理**：支持大规模数据处理

## 故障排除

### 常见问题

1. **网络连接问题**
   ```powershell
   # 设置HuggingFace镜像
   set HF_ENDPOINT=https://hf-mirror.com
   set HF_HOME=.\.hf_cache
   ```

2. **内存不足**
   - 降低batch_size到4或2
   - 使用梯度累积

3. **训练中断**
   - 模型自动保存在 `ckpts/norm-byt5-cpu/best/`
   - 可以从断点继续训练

### 环境变量
```powershell
# 设置国内镜像（可选）
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HOME = ".\.hf_cache"
$env:PIP_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple/"
```

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。
