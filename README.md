# PV_Prediction_COTN

## 光伏发电预测项目

本项目对50MW光伏电站的发电数据进行分析和预测建模。

## 项目结构

```
PV_prediction/
├── Solar Station/                    # 原始数据文件夹
│   └── Solar station site 1 (Nominal capacity-50MW).xlsx
├── prediction/                       # 预测模型文件夹
│   ├── pv_power_predictor.py        # 主预测模型训练脚本
│   ├── prediction_interface.py      # 预测接口和演示
│   └── time_series_analysis.py      # 时间序列分析
├── analysis/                         # 数据分析和可视化文件夹
│   ├── pv_data_analysis.py          # 基础数据分析
│   ├── advanced_pv_analysis.py      # 高级分析功能
│   └── visualization_analysis.py    # 可视化分析
└── README.md                         # 项目说明文档
```

## 数据概览

项目使用的数据来源于50MW光伏电站，包含：
- **时间跨度**: 2019-2020年，共70,176个15分钟间隔的数据点
- **主要变量**:
  - 总辐照强度 (W/m²)
  - 直接法向辐照强度 (W/m²)
  - 全球水平辐照强度 (W/m²)
  - 空气温度 (°C)
  - 大气压 (hpa)
  - 发电功率 (MW) - 目标变量

## 核心功能模块

### 1. 预测模型 (prediction/)

#### pv_power_predictor.py
主要的光伏发电功率预测模型，基于随机森林算法：

**特征工程**:
- 时间特征：小时、月份、年内第几天、季节、是否周末
- 辐照特征：辐照比率、辐照总和
- 标准化特征用于模型训练

**模型性能**:
- R²: 0.967 (96.7% 准确度)
- MAE: 0.944 MW
- RMSE: 1.953 MW
- 交叉验证得分: 0.966 ± 0.002

**主要功能**:
- 数据加载和预处理
- 特征工程和选择
- 模型训练和评估
- 特征重要性分析
- 模型保存和加载

#### prediction_interface.py
提供用户友好的预测接口：

**预测功能**:
- 单点预测：给定气象条件预测功率
- 批量预测：处理多个数据点
- 日发电曲线预测：预测全天24小时发电量

**演示案例**:
- 晴天 (800 W/m², 25°C): 26.29 MW
- 多云 (400 W/m², 22°C): 20.63 MW  
- 阴天 (150 W/m², 18°C): 5.17 MW

#### time_series_analysis.py
时间序列预测分析，比较不同预测方法：

**分析内容**:
- 时间序列分割：前80%训练，后20%测试
- 时间序列预测性能：R² = 0.871, MAE = 1.988 MW
- 预测误差分析和可视化
- 不同功率区间的预测性能评估

### 2. 数据分析 (analysis/)

#### visualization_analysis.py
综合数据可视化分析：

**分析模块**:
1. **概览分析**: 功率分布、时间趋势、统计摘要
2. **相关性分析**: 各变量间的关联关系热力图
3. **季节性分析**: 月度、小时变化模式，天气影响
4. **性能分析**: 效率指标、辐照利用率分析

#### advanced_pv_analysis.py
高级统计分析功能：

**分析功能**:
- 统计描述和分布分析
- 时间序列趋势分解
- 异常值检测和处理
- 性能基准分析

## 技术栈

- **数据处理**: pandas, numpy
- **机器学习**: scikit-learn (RandomForestRegressor)
- **可视化**: matplotlib, seaborn
- **数据读取**: openpyxl (Excel文件处理)
- **模型持久化**: joblib

## 模型特征重要性

基于随机森林模型的特征重要性排名：

1. **Total_Irradiance**: 0.264 - 总辐照强度 (最重要)
2. **Global_Irradiance**: 0.186 - 全球水平辐照强度
3. **Direct_Irradiance**: 0.157 - 直接法向辐照强度  
4. **Irradiance_Sum**: 0.127 - 辐照总和
5. **Hour**: 0.108 - 小时 (时间因素)
6. **Temperature**: 0.063 - 空气温度
7. **DayOfYear**: 0.045 - 年内第几天
8. **Month**: 0.025 - 月份

## 预测性能分析

### 映射式预测 vs 时间序列预测

**映射式预测** (随机分割数据):
- R²: 0.967 (96.7%)
- MAE: 0.944 MW
- 优点：准确度高，适合理想条件下的预测

**时间序列预测** (按时间顺序分割):
- R²: 0.871 (87.1%)  
- MAE: 1.988 MW
- 优点：更符合实际应用场景，避免使用未来数据

### 误差分析

**按功率区间的预测性能**:
- 0-5MW: MAE = 0.176 MW (低功率时段，主要是夜间)
- 5-15MW: MAE = 1.504 MW
- 15-25MW: MAE = 3.631 MW
- 25-35MW: MAE = 6.349 MW (高功率时段，误差较大)

**观察结论**:
- 夜间和低辐照时段预测准确
- 白天高功率时段预测误差较大
- 误差主要集中在快速功率变化期间

## 使用方法

### 1. 环境准备
```bash
# 激活conda环境 (建议使用torch环境)
conda activate torch

# 安装必要依赖
conda install pandas numpy matplotlib seaborn scikit-learn openpyxl joblib
```

### 2. 训练预测模型
```bash
cd prediction/
python pv_power_predictor.py
```

### 3. 使用预测接口
```bash
cd prediction/
python prediction_interface.py
```

### 4. 生成数据分析
```bash
cd analysis/
python visualization_analysis.py
```

### 5. 时间序列分析
```bash
cd prediction/
python time_series_analysis.py
```

## 输出文件

**预测模块输出**:
- `pv_power_model.joblib`: 训练好的预测模型
- `prediction_results.png`: 预测结果可视化
- `feature_importance.png`: 特征重要性图表
- `time_series_comparison_200pts.png`: 时间序列预测对比
- `time_series_error_analysis.png`: 误差分析图表

**分析模块输出**:
- `pv_overview_analysis.png`: 数据概览分析
- `pv_correlation_analysis.png`: 相关性分析热力图
- `pv_seasonal_analysis.png`: 季节性分析图表
- `pv_performance_analysis.png`: 性能分析图表

## 项目亮点

1. **完整的机器学习流程**: 从数据加载到模型部署
2. **多种预测方式**: 支持单点、批量、日曲线预测
3. **时间序列分析**: 区分映射问题和时间序列问题
4. **全面的可视化**: 多维度数据分析和结果展示
5. **模块化设计**: 预测和分析功能分离，便于扩展
6. **实用的接口**: 简单易用的预测API

## 应用场景

- **电网调度**: 提前预测光伏电站发电量，优化电网运行
- **能源管理**: 帮助电站运营商制定发电计划
- **投资决策**: 评估光伏项目的发电效率和经济效益
- **研究分析**: 为光伏发电相关研究提供数据支持

## 未来改进方向

1. **提升时间序列预测精度**: 当前87.1%的准确率还有提升空间
2. **引入深度学习模型**: 尝试LSTM、Transformer等模型
3. **多元时间序列**: 考虑多个电站间的关联预测
4. **实时预测功能**: 集成天气预报API，实现实时预测
5. **模型解释性**: 增加SHAP等模型解释工具
6. **部署优化**: 开发Web接口，便于实际应用

## 注意事项

1. 确保Excel数据文件路径正确
2. 首次运行预测脚本需要安装openpyxl包
3. 生成的图表和模型文件会保存在对应文件夹中
4. 预测模型基于历史数据训练，实际应用时需要考虑设备老化等因素

## 联系信息

如有问题或建议，请联系项目开发者。