# CPT spectral transfer: AdamW / Full Muon / Pion

用于另一台服务器的独立代码快照。包含已验证的 Pythia-31M CPT、配对数据处理、梯度诊断、13 项语料距离与报告代码。原有 Python 文件保持逐字一致，见 `SOURCE_SNAPSHOT.json`。

**这不是已经完成的 2.8B/6.9B 多卡训练系统。** 当前训练器采用 GPT-NeoX 参数名、单 GPU、有效 batch 32，并从数据 manifest 读取模型 ID。仅修改模型名字、或套上 `torchrun`，不能自动获得正确的多卡训练；后续扩展需单独验证分片、优化器和学习率。

## 1. 新服务器安装

代码位于公开仓库 `sun-messi/Muon` 的独立分支，不需要 GitHub 登录即可下载。

```bash
git clone --branch cpt-spectral-transfer-20261020 --single-branch https://github.com/sun-messi/Muon.git muon-cpt
cd muon-cpt/cpt-spectral-transfer
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pytest -q
python train_openwebmath.py --help
```

`reference_environment.json` 是原服务器的依赖版本。CUDA wheel、驱动兼容性和集群模块需在目标服务器核对。数据源准备还可能需要 `bsdtar` 及 ORD 专用依赖，不能据此声称所有下载器在干净环境中均已实测。

## 2. 算法在哪里

- `spectral.py`：共享矩阵优化器 `MatchedMuon`、Muon 5NS、Pion promotion/suppression。
- `train_mid.py`：GPT-NeoX 参数分组；隐藏层二维矩阵走矩阵优化器，其他参数走辅助优化器。
- `train_smoltalk.py` / `train_openwebmath.py`：全位置 causal CE、调度、配对指纹、评估和保存。
- `prepare_real_task_*`：已有语料处理程序；不是一个已经覆盖所有来源的一键下载器。
- `analyze_corpus_distances.py` / `report_corpus_distances.py`：16 语料、两参照、13 指标与优化器差值的分析。
- `examples/observed_results/`：已有 48 条训练路线的小型数值表，不含文本数据或模型权重。

可直接在自己的训练循环中使用：

```python
from spectral import MatchedMuon, parse_schedule

# matrix_parameters 必须是经核对的隐藏层矩阵参数；bias、embedding 等另行处理。
opt = MatchedMuon(matrix_parameters, lr=0.003, momentum=0.95,
                  schedule=parse_schedule("0:pion:3:2:.45"))
# Full Muon 则用 "0:muon5ns:0:0:0"。这里学习率只对应原 31M 实验。
```

## 3. 数据另传，不放 GitHub

在**原服务器**执行，第二个参数换成新服务器的 SSH 地址，第三个参数换成新服务器的项目绝对目录：

```bash
bash scripts/sync_reference_data.sh \
  /home/sunj11/muon_project/pythia31m-midtraining-pion \
  USER@NEW_SERVER \
  /ABS/PATH/muon-cpt/cpt-spectral-transfer
```

只传 run2/3/4 的 16 组缓存和必要 endpoint CSV。会**解引用旧机器上的绝对符号链接**，新服务器得到实际缓存文件；不传训练 checkpoint 或大型原始语料。可重复执行 rsync 续传。模型权重由 Hugging Face 另行下载，或使用集群已有缓存。

在新服务器校验：

```bash
python scripts/verify_reference_caches.py
```

若不使用旧缓存，可参考 `prepare_real_task_*` 重建；重建所需原始数据、版本及下载依赖需要另行准备。

## 4. 先做原设置短步测试

确保模型可以通过 Hugging Face 下载；需要离线运行时，先在可联网节点下载 manifest 指定的精确 revision。

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_reference.sh \
  runs/real_task_eight_theme_cpt/run4/cache/pile_cc \
  runs/transfer_smoke/run1/pion pion smoke
```

三个方法名：`adam`、`muon`、`pion`；正式复现将末尾 `smoke` 改为 `full`。每次使用新的输出目录，不覆盖已有结果。Slurm 单卡示例见 `scripts/reference.slurm`；分区、账户、GPU 类型及 walltime 需按目标集群填写。

## 5. 重算已有语料距离

传完缓存后，可直接运行：

```bash
CUDA_VISIBLE_DEVICES=0 python analyze_corpus_distances.py \
  --out runs/corpus_distribution_distance/run1
python report_corpus_distances.py runs/corpus_distribution_distance/run1
```

第一次会下载固定 revision 的 `sentence-transformers/all-MiniLM-L6-v2`。代码分析的是现有 16 领域；新模型、新领域需后续参数化扩展，不能只给旧结果换标签。

结果边界：单训练 seed；五次子采样不是五个训练 seed；Pile held-out 只是预训练分布的近似。距离与方法优势的相关不等于因果机制。
