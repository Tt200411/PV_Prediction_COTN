# 光伏发电预测项目

本项目对50MW光伏电站的发电数据进行分析和预测建模。

## 项目结构

```
PV_prediction/
├── prediction/                    # 预测模型相关文件
│   ├── pv_power_predictor.py     # 主要预测模型训练脚本
│   └── prediction_interface.py   # 预测接口和演示
├── analysis/                     # 数据分析和可视化
│   ├── pv_data_analysis.py      # 基础数据分析
│   ├── advanced_pv_analysis.py  # 高级数据分析
│   ├── visualization_analysis.py # 专门的可视化分析
│   ├── *.png                    # 生成的分析图表
├── Solar Station/               # 原始数据
│   └── Solar station site 1 (Nominal capacity-50MW).xlsx
└── README.md                   # 项目说明文档
```

## 数据概览

- **数据源**: 50MW光伏电站2019-2020年运行数据
- **数据量**: 70,176个数据点（15分钟间隔）
- **主要变量**:
  - 总辐照强度 (W/m²)
  - 直接辐照强度 (W/m²) 
  - 全球水平辐照强度 (W/m²)
  - 空气温度 (°C)
  - 大气压 (hpa)
  - 发电功率 (MW)

## 快速开始

### 1. 环境准备

确保已安装以下Python包：
```bash
conda activate torch  # 或其他包含所需依赖的环境
```

需要的包：pandas, numpy, matplotlib, seaborn, scikit-learn, openpyxl

### 2. 数据分析

运行可视化分析：
```bash
cd analysis/
python visualization_analysis.py
```

这将生成4个主要分析图表：
- 数据总览分析
- 相关性分析  
- 季节性分析
- 性能分析

### 3. 预测模型

训练预测模型：
```bash
cd prediction/
python pv_power_predictor.py
```

使用预测接口：
```bash
python prediction_interface.py
```

## 主要发现

### 电站性能
- **容量因子**: 19.3%
- **平均功率**: 9.67 MW
- **最大功率**: 48.32 MW  
- **总发电量**: 169.6 GWh (2年)

### 季节特性
- **夏季**表现最佳: 平均10.75MW
- **秋季**次之: 平均10.42MW
- **春季和冬季**相对较低

### 预测模型性能
- **准确度**: 96.9% (R² = 0.969)
- **平均绝对误差**: 1.019 MW
- **最重要特征**: 总辐照强度 (91.0%重要性)

## 文件说明

### prediction/ 文件夹

#### pv_power_predictor.py
- 完整的预测模型训练流程
- 包含数据预处理、特征工程、模型训练、评估
- 支持随机森林和线性回归模型
- 自动保存训练好的模型

#### prediction_interface.py  
- 易用的预测接口
- 支持单点预测、批量预测、日发电曲线预测
- 包含演示功能

### analysis/ 文件夹

#### visualization_analysis.py
- 专门的可视化分析脚本
- 生成4类分析图表
- 包含详细的性能指标计算

#### pv_data_analysis.py
- 基础数据探索和分析
- 简单的可视化功能

#### advanced_pv_analysis.py  
- 综合分析脚本
- 包含数据分析、可视化和预测建模

## 使用示例

### 预测示例

```python
from prediction.prediction_interface import PVPowerPredictionInterface

# 初始化预测器
predictor = PVPowerPredictionInterface()

# 单点预测
power = predictor.predict_single(
    total_irradiance=800,    # W/m²
    direct_irradiance=600,   # W/m²  
    global_irradiance=200,   # W/m²
    temperature=25,          # °C
    atmosphere=915,          # hpa
    hour=12,                 # 小时
    month=6,                 # 月份
    day_of_year=150          # 年内第几天
)
print(f"预测功率: {power:.2f} MW")

# 日发电曲线预测
daily_curve = predictor.predict_daily_curve(
    date_str='2024-06-15',
    avg_irradiance=600,
    avg_temperature=28
)
```

## 注意事项

1. 确保Excel数据文件路径正确
2. 首次运行预测脚本需要安装openpyxl包
3. 生成的图表和模型文件会保存在对应文件夹中
4. 预测模型基于历史数据训练，实际应用时需要考虑设备老化等因素

## 联系信息

如有问题或建议，请联系项目开发者。