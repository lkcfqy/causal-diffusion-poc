# causal-diffusion-poc

因果条件扩散模型 POC。项目基于彩色 MNIST 构造一个带伪相关的数据集：训练集中数字 3 大多为红色、数字 7 大多为绿色，测试集反转颜色相关性，用来比较普通条件扩散模型和显式拆分因果/伪特征条件的 SCD 模型。

## 当前状态

仓库包含数据生成、训练、评估脚本，并已经提交了两个模型权重：

- `saved_models/baseline_model.pt`
- `saved_models/scd_model.pt`

`results/data_samples/` 中保留了若干训练样本图。H1 鲁棒性和 H2 反事实评估脚本已经实现，但对应生成结果需要运行 `src/evaluate.py` 后才会写入 `results/h1_robustness/` 和 `results/h2_counterfactuals/`。

## 实验设计

- 因果标签 `z_c`：数字类别，当前为 `3` 和 `7`。
- 伪相关标签 `z_s`：颜色，当前为红色和绿色。
- `baseline`：只以数字条件生成。
- `scd`：同时接收数字条件和颜色条件，支持更明确的反事实控制。

默认配置见 `config.yaml`：图像大小 `32`，训练 `500` epoch，扩散步数 `1000`，默认设备为 `mps`。

## 环境准备

```bash
conda env create -f environment.yml
conda activate causal_diff_env
```

如果不是 Apple Silicon，可把 `config.yaml` 中的 `DEVICE` 改为 `cuda` 或 `cpu`。

## 运行流程

生成 Causal-MNIST 数据：

```bash
python src/create_dataset.py
```

训练 baseline 和 SCD：

```bash
python src/train.py
```

评估并生成图片：

```bash
python src/evaluate.py
```

## 主要文件

- `src/create_dataset.py`：构造 IID/OOD 彩色 MNIST。
- `src/models.py`：条件 U-Net。
- `src/diffusion.py`：扩散过程。
- `src/train.py`：依次训练 baseline 和 SCD。
- `src/evaluate.py`：H1 颜色鲁棒性和 H2 反事实评估。
- `config.yaml`：数据、训练和模型配置。

## 许可证

当前仓库未包含独立 `LICENSE` 文件。如需公开复用或分发，请先补充明确的开源许可证。
