# Causal Diffusion PoC: 扩散模型中的因果解耦研究

本项目是一个概念验证（Proof of Concept），旨在探索**因果扩散模型（Structural Causal Diffusion, SCD）**如何处理数据中的伪相关（Spurious Correlations）。

通过对比 **Baseline（基线模型）** 和 **SCD（结构因果扩散模型）** 在构建的 "Causal-MNIST" 数据集上的表现，本项目验证了显式建模因果变量（数字）和环境变量（颜色）能否提升生成的鲁棒性和反事实推理能力。

## 📂 项目结构

```text
causal-diffusion-poc/
├── config.yaml             # 全局配置文件 (超参数、路径设置)
├── environment.yml         # Conda 环境依赖文件
├── .gitignore              # Git 忽略配置
├── data/                   # 存放生成的训练集和测试集 (.pt 文件)
├── saved_models/           # 存放训练好的模型权重 (.pt 文件)
├── results/                # 存放生成的图像样本和评估结果
│   ├── data_samples/       # 数据集可视化样本
│   ├── h1_robustness/      # H1 实验：OOD 鲁棒性生成结果
│   └── h2_counterfactuals/ # H2 实验：反事实推理结果
└── src/                    # 源代码目录
    ├── create_dataset.py   # 数据生成脚本 (Causal-MNIST)
    ├── diffusion.py        # 扩散过程实现 (Forward/Sampling)
    ├── models.py           # Conditional U-Net 模型架构 (含 Self-Attention)
    ├── train.py            # 模型训练脚本 (Baseline & SCD)
    └── evaluate.py         # 评估脚本 (H1 & H2 假设验证)

```

## 🧪 实验设计

### 数据集：Causal-MNIST

我们在 MNIST 基础上构建了一个存在强伪相关的数据集：

* **数字 (Causal Label, )**: `3` 和 `7`
* **颜色 (Spurious Label, )**: `红色` 和 `绿色`
* **训练集 (IID)**: 存在强相关性 (Correlation = 0.9)。
* 数字 `3` 有 90% 概率是 **红色**。
* 数字 `7` 有 90% 概率是 **绿色**。


* **测试集 (OOD)**: 相关性反转。
* 数字 `3` 主要为绿色，数字 `7` 主要为红色。



### 模型对比

1. **Baseline Model**: 标准条件扩散模型 。仅以数字类别作为条件，容易学习到“3就是红色”的错误关联。
2. **SCD Model**: 结构因果模型 。同时以数字类别和颜色作为独立条件输入，通过 `Embedding` 相加的方式注入 U-Net。

### 验证假设

* **H1 (Robustness)**: 在分布外 (OOD) 场景下，SCD 模型能比 Baseline 生成更符合预期的数字（即便颜色组合在训练集中很少见）。
* **H2 (Counterfactuals)**: SCD 模型能够执行反事实推理（例如：将一个“红色的3”变成“红色的7”），而 Baseline 往往会在改变数字的同时错误地改变颜色。

## 🚀 快速开始

### 1. 环境配置

本项目基于 PyTorch 和 Apple Silicon (MPS) 优化（也可运行于 CUDA）。

```bash
# 创建 Conda 环境
conda env create -f environment.yml

# 激活环境
conda activate causal_diff_env

```

### 2. 生成数据

首先生成带有颜色偏差的 MNIST 数据集。

```bash
python src/create_dataset.py

```

* 输出：`data/train.pt`, `data/test.pt` 以及 `results/data_samples/` 中的预览图。

### 3. 训练模型

该脚本会依次训练 `baseline` 和 `scd` 两个模型。

```bash
python src/train.py

```

* 配置：默认 500 Epochs，Batch Size 64（可在 `config.yaml` 中修改）。
* 输出：模型保存在 `saved_models/`。

### 4. 评估结果

执行两个主要的假设测试（H1 和 H2）。

```bash
python src/evaluate.py

```

* **H1 结果**: 查看 `results/h1_robustness/`。
* **H2 结果**: 查看 `results/h2_counterfactuals/`，包含原始图与反事实生成图的对比。

## ⚙️ 技术细节

* **架构**: Conditional U-Net。
* 包含 **Self-Attention (自注意力)** 模块以捕捉全局结构。
* 使用正弦位置嵌入 (Sinusoidal Position Embeddings) 处理时间步。


* **扩散策略**: DDPM (Denoising Diffusion Probabilistic Models)，1000 步。
* **图像规格**: 32x32 像素，RGB 3通道。
* **条件机制**:
* Baseline: `Time_Emb + Class_Emb(Digit)`
* SCD: `Time_Emb + Class_Emb(Digit) + Color_Emb(Color)`



## 📊 配置文件说明 (`config.yaml`)

```yaml
IMG_SIZE: 32            # 图像尺寸
DIGITS: [3, 7]          # 使用的数字类别
EPOCHS: 500             # 训练轮次 (建议保持较高以获得清晰图像)
BATCH_SIZE: 64          # 批次大小
LEARNING_RATE: 0.0001   # 学习率
TIMESTEPS: 1000         # 扩散步数
BASE_CHANNELS: 64       # U-Net 基础通道数
DEVICE: 'mps'           # 'mps' (Mac), 'cuda' (NVIDIA), or 'cpu'

```

## 📝 引用与致谢

本项目基于 PyTorch 官方示例与因果表示学习相关文献构建。如有使用请注明出处。
