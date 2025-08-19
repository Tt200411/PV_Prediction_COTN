# COTN光伏预测完整测试框架

本框架实现了COTN (Continuous Oscillator Time Networks) 模型在光伏发电预测任务上的完整测试体系，包括单数据集测试、多数据集对比、滚动回测等功能。

## 📁 文件结构

```
COTN/
├── prepare_pv_data.py          # 光伏数据预处理脚本
├── train_pv_prediction.py     # 基础COTN训练脚本
├── backtest_framework.py      # 滚动回测框架
├── multi_dataset_tester.py    # 多数据集测试框架  
├── cotn_complete_tester.py    # 完整测试套件集成
├── debug_loader.py            # 数据加载器调试工具
├── exp/
│   ├── exp_config.py          # COTN模型配置 (已适配光伏数据)
│   └── exp_informer.py        # COTN实验类 (已添加光伏数据映射)
├── data/
│   └── data_loader.py         # 数据加载器 (已修改为时间序列分割)
└── ETT-small/
    ├── PV_Solar_Station_1.csv # 已转换的光伏数据
    └── ...                    # 其他数据集将动态生成
```

## 🚀 快速开始

### 1. 环境准备
```bash
# 激活torch环境
conda activate torch

# 确保在COTN目录下
cd /Users/tangbao/Desktop/PV_prediction/COTN
```

### 2. 基础训练
```bash
# 训练Site 1数据集上的COTN模型
python train_pv_prediction.py
```

### 3. 完整测试套件
```bash
# 运行完整的测试套件
python cotn_complete_tester.py
```

## 📊 测试框架详解

### 1. 单数据集测试 (`train_pv_prediction.py`)

**功能**: 在单个光伏电站数据上训练和评估COTN模型

**配置**:
- 输入序列长度: 96个时间步 (24小时)
- 预测序列长度: 24个时间步 (6小时)  
- 时间序列分割: 前80%训练，后20%测试
- 单变量预测: 只预测发电功率

**使用示例**:
```python
from cotn_complete_tester import run_cotn_on_single_dataset

# 测试单个数据集
result = run_cotn_on_single_dataset("Site_1_50MW")
print(f"R²: {result['pred_result']['r2']:.3f}")
```

### 2. 滚动回测框架 (`backtest_framework.py`)

**功能**: 模拟真实部署环境的滚动预测测试

**特点**:
- **扩展窗口**: 使用所有历史数据训练
- **固定预测窗口**: 每次预测6小时
- **定期重训练**: 避免模型漂移
- **步进式评估**: 每6小时评估一次性能

**关键参数**:
```python
WalkForwardBacktest(
    model=cotn_model,
    data=pv_data,
    train_window=None,      # 扩展窗口
    test_window=24,         # 预测6小时  
    step_size=24,           # 每次前进6小时
    retrain_freq=96         # 每24小时重训练
)
```

**使用示例**:
```python
from cotn_complete_tester import run_cotn_backtest

# 运行滚动回测
backtest, summary, results = run_cotn_backtest()
print(f"平均MAE: {summary['avg_mae']:.3f} MW")
```

### 3. 多数据集测试框架 (`multi_dataset_tester.py`)

**功能**: 在所有8个光伏电站数据上评估模型泛化性能

**电站信息**:
| 电站 | 装机容量 | 数据特点 |
|------|----------|----------|
| Site 1 | 50MW | 中等规模 |
| Site 2 | 130MW | 大规模 |
| Site 3 | 30MW | 小规模 |
| Site 4 | 130MW | 大规模 |
| Site 5 | 110MW | 大规模 |
| Site 6 | 35MW | 小规模 |
| Site 7 | 30MW | 小规模 |
| Site 8 | 30MW | 小规模 |

**自动化流程**:
1. 自动加载所有数据集
2. 转换为COTN兼容格式
3. 依次训练和测试
4. 生成性能对比报告
5. 可视化结果对比

**使用示例**:
```python
from cotn_complete_tester import run_cotn_multi_dataset_test

# 运行多数据集测试
tester = run_cotn_multi_dataset_test()
```

## 🔧 配置说明

### COTN模型配置 (`exp/exp_config.py`)

已针对光伏预测任务优化的关键配置:

```python
class InformerConfig:
    def __init__(self):
        # 时间序列参数
        self.seq_len = 96      # 输入: 24小时历史
        self.label_len = 48    # 解码器起始: 12小时
        self.pred_len = 24     # 预测: 6小时未来
        
        # 模型参数  
        self.enc_in = 1        # 单变量输入
        self.dec_in = 1        # 单变量解码
        self.c_out = 1         # 单变量输出
        
        # 数据参数
        self.features = 'S'    # 单变量预测模式
        self.target = 'Power'  # 目标: 发电功率
        self.freq = 't'        # 分钟级时间特征
```

### 时间序列分割策略

**严格时间分割**:
- 训练集: 前80%数据（时间顺序）
- 测试集: 后20%数据（未来数据）
- 无数据泄露: 测试集完全不可见

**滑动窗口生成**:
- 样本1: 时间步[0:95] → 预测[96:119]  
- 样本2: 时间步[1:96] → 预测[97:120]
- ...
- 样本N: 时间步[N:N+95] → 预测[N+96:N+119]

## 📈 性能基准

### 与随机森林对比

**随机森林基准** (现有项目结果):
- 映射式预测: R² = 0.967, MAE = 0.944 MW
- 时间序列预测: R² = 0.871, MAE = 1.988 MW

**COTN目标**:
- 时间序列建模: 期望R² > 0.871
- 长期依赖: 更好的多步预测能力
- 模式识别: 自动学习复杂天气-发电关系

## 🛠️ 故障排除

### 常见问题

1. **训练数据长度错误**
   ```
   错误: __len__() should return >= 0
   解决: 检查seq_len + pred_len是否小于数据集大小
   ```

2. **GPU内存不足**
   ```
   解决: 设置config.use_gpu = False使用CPU训练
   ```

3. **时间特征解析错误**
   ```
   错误: '15min' not supported
   解决: 使用config.freq = 't'作为分钟级频率
   ```

4. **模型维度不匹配**
   ```
   错误: expected input[32, 1, 98] to have 6 channels
   解决: 确保config.enc_in与features模式匹配
   ```

### 性能优化

1. **训练加速**:
   - 使用GPU: `config.use_gpu = True`
   - 减少epochs: `config.train_epochs = 3`
   - 小批量: `config.batch_size = 16`

2. **回测优化**:
   - 增大step_size减少回测次数
   - 使用固定窗口而非扩展窗口
   - 降低重训练频率

## 📝 输出文件

### 自动生成的结果文件

1. **模型文件**:
   - `checkpoints/informer_PV_*.pth`: 训练好的模型权重

2. **可视化结果**:
   - `cotn_pv_prediction_results.png`: 单数据集预测结果
   - `cotn_backtest_results.png`: 滚动回测结果
   - `multi_dataset_comparison.png`: 多数据集性能对比

3. **报告文件**:
   - `multi_dataset_report.csv`: 详细性能数据
   - `comprehensive_report.txt`: 综合测试报告

## 🔄 扩展使用

### 添加新数据集

1. 将Excel文件放入`/Users/tangbao/Desktop/PV_prediction/Solar Station/`
2. 运行多数据集测试器会自动识别和处理

### 自定义配置

```python
# 创建自定义配置
custom_config = {
    'seq_len': 192,     # 使用2天历史
    'pred_len': 48,     # 预测12小时
    'train_epochs': 10, # 更多训练轮数
}

# 应用到模型
model = COTNModelWrapper("Site_1_50MW", config_overrides=custom_config)
```

### 集成其他模型

框架设计为模块化，可以轻松替换COTN为其他时间序列模型，只需实现相同的接口:

```python
class YourModel:
    def fit(self, train_data): pass
    def predict(self, test_data): pass
```

---

**注**: 本框架为COTN在光伏预测任务上的完整实现，提供了从数据预处理到性能评估的全流程工具链。