# COTN光伏预测框架 - 整理版

## 🎯 项目概述

本项目是基于COTN (Continuous Oscillator Time Networks) 的光伏发电预测系统，已完成代码整理和优化。

## ✅ 已修复的关键问题

### 1. **数据分割逻辑修复**
- **问题**: 原始代码使用ETT数据集的固定日期分割，导致train loss和vali loss下降但test loss不下降
- **修复**: 创建专门的`Dataset_PV`类，使用80%/10%/10%的比例分割光伏数据
- **影响**: 确保了train/validation/test数据的时序连续性和合理性

### 2. **多数据集支持**
- 支持所有8个光伏电站数据集的自动训练
- 智能处理不同Excel文件的sheet名称差异 (sheet1 vs Sheet1)
- 自动处理缺失的温度列等数据格式问题

### 3. **预测长度优化**
- 默认预测长度设置为100个时间步 (25小时)
- 支持命令行自定义预测长度

## 📁 整理后的文件结构

```
COTN/
├── train_multi_datasets.py    # 主训练脚本 (新)
├── train_all.sh              # Shell快捷脚本 (新)
├── data/
│   └── data_loader.py        # 修复数据分割逻辑
├── exp/
│   ├── exp_config.py         # 配置文件 (更新pred_len=100)
│   └── exp_informer.py       # 实验类
├── models/                   # COTN模型定义
├── utils/                    # 工具函数
├── ETT-small/               # 数据文件 (8个站点CSV)
└── checkpoints/             # 模型保存目录
```

## 🚀 快速开始

### 方法1: 使用Shell脚本 (推荐)

```bash
# 训练所有数据集 (Lee振荡器)
./train_all.sh

# 训练所有数据集 (ReLU激活函数)
./train_all.sh --activation relu

# 仅训练指定数据集
./train_all.sh --datasets "Site_1_50MW Site_2_130MW"

# 仅训练模式 (服务器端使用)
./train_all.sh --train-only

# 自定义预测长度和训练轮数
./train_all.sh --pred-len 50 --epochs 10

# 查看所有选项
./train_all.sh --help
```

### 方法2: 直接使用Python

```bash
# 激活环境 (如果需要)
conda activate torch

# 训练所有数据集
python train_multi_datasets.py --datasets Site_1_50MW Site_2_130MW Site_3_30MW Site_4_130MW Site_5_110MW Site_6_35MW Site_7_30MW Site_8_30MW

# 使用ReLU激活函数
python train_multi_datasets.py --activation relu --datasets Site_1_50MW

# 仅训练模式
python train_multi_datasets.py --train_only --datasets Site_1_50MW
```

## 🎛️ 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--datasets` | 数据集列表 | 所有8个数据集 |
| `--activation` | 激活函数 (lee/relu) | lee |
| `--pred_len` | 预测长度 | 100 |
| `--epochs` | 训练轮数 | 6 |
| `--train_only` | 仅训练模式 | False |
| `--test_only` | 仅测试模式 | False |
| `--batch_size` | 批大小 | 32 |
| `--learning_rate` | 学习率 | 1e-4 |

## 📊 数据集信息

| 数据集 | 装机容量 | 数据点数 | 状态 |
|--------|----------|----------|------|
| Site_1_50MW | 50MW | 70,176 | ✅ |
| Site_2_130MW | 130MW | 70,176 | ✅ |
| Site_3_30MW | 30MW | 20,352 | ✅ |
| Site_4_130MW | 130MW | 70,176 | ✅ |
| Site_5_110MW | 110MW | 70,176 | ✅ |
| Site_6_35MW | 35MW | 70,176 | ✅ |
| Site_7_30MW | 30MW | 70,176 | ✅ |
| Site_8_30MW | 30MW | 69,408 | ✅ |

**总计**: 545MW装机容量，510,816个数据点

## 📈 模型配置

- **输入序列**: 96个时间步 (24小时历史)
- **预测长度**: 100个时间步 (25小时预测)
- **数据分割**: 80% 训练 / 10% 验证 / 10% 测试
- **激活函数**: Lee振荡器 (默认) 或 ReLU
- **时间频率**: 15分钟间隔

## 🔧 技术特点

1. **修复的数据分割**: 按时序比例分割，避免train/vali/test不一致问题
2. **智能数据处理**: 自动处理不同格式的Excel文件
3. **灵活的激活函数**: 支持Lee振荡器和ReLU对比实验
4. **批量训练**: 一次性训练多个数据集
5. **详细日志**: 显示训练进度和数据分割信息

## 📝 输出文件

训练完成后会生成：

- `checkpoints/informer_{dataset}_{activation}_pred{length}/checkpoint.pth` - 模型权重
- `training_results_{activation}_pred{length}.json` - 训练结果摘要 (如果使用--save_results)

## ⚡ 性能优化

- 自动检测并使用GPU加速
- 支持早停机制防止过拟合
- 内存优化的数据加载
- 并行数据处理

## 🛠️ 故障排除

1. **CUDA错误**: 脚本会自动检测CUDA环境，如果有问题会切换到CPU
2. **数据文件不存在**: 确保ETT-small/目录下有所需的CSV文件
3. **内存不足**: 减少batch_size参数
4. **训练不收敛**: 尝试调整学习率或使用不同的激活函数

---

## 🎯 准备就绪!

所有问题已修复，代码已优化整理。您现在可以：

1. **服务器端训练**: `./train_all.sh --train-only`
2. **本地分析**: `./train_all.sh --test-only` 
3. **完整流程**: `./train_all.sh`

祝您训练顺利！🚀