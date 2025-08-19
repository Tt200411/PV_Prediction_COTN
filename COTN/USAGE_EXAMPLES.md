# COTN使用示例

## 快速开始

你现在可以通过修改 `cotn_complete_tester.py` 文件末尾的配置选项来控制运行模式：

```python
# 配置选项
TRAIN_ONLY = False     # 设置为True仅执行训练
TEST_ONLY = False      # 设置为True仅执行测试  
USE_RELU = False       # 设置为True使用ReLU替代Lee振荡器
```

## 使用场景

### 1. 服务器端训练模式
```python
# 在服务器上只进行训练，保存模型
TRAIN_ONLY = True
TEST_ONLY = False
USE_RELU = False  # 或 True 根据需要选择激活函数
```

### 2. 个人电脑分析模式
```python
# 在个人电脑上加载已训练模型进行测试和分析
TRAIN_ONLY = False
TEST_ONLY = True
USE_RELU = False  # 必须与训练时保持一致
```

### 3. 完整流程模式
```python
# 训练+测试一体化
TRAIN_ONLY = False
TEST_ONLY = False
USE_RELU = False  # 或 True
```

### 4. 激活函数对比实验
```python
# 运行Lee振荡器版本
USE_RELU = False

# 运行ReLU版本
USE_RELU = True
```

## 文件名规范

现在生成的模型和结果文件会包含更多信息：

- **Lee振荡器**: `informer_Site_1_50MW_lee2_pred24.pth`
- **ReLU激活**: `informer_Site_1_50MW_relu_pred24.pth`
- **不同预测步长**: `informer_Site_1_50MW_lee2_pred48.pth`

其中：
- `Site_1_50MW`: 数据集名称
- `lee2` 或 `relu`: 激活函数类型
- `pred24`: 预测步长（24个时间步 = 6小时）

## 编程接口

你也可以在代码中直接使用新功能：

```python
from cotn_complete_tester import run_cotn_on_single_dataset

# 仅训练模式
result = run_cotn_on_single_dataset(
    "Site_1_50MW", 
    train_only=True, 
    test_only=False, 
    use_relu=False
)

# 仅测试模式
result = run_cotn_on_single_dataset(
    "Site_1_50MW", 
    train_only=False, 
    test_only=True, 
    use_relu=False
)

# ReLU激活函数
result = run_cotn_on_single_dataset(
    "Site_1_50MW", 
    train_only=False, 
    test_only=False, 
    use_relu=True
)
```

## 实际工作流程建议

1. **服务器端**：
   ```bash
   # 修改配置为 TRAIN_ONLY = True
   python cotn_complete_tester.py
   # 下载生成的 .pth 文件到本地
   ```

2. **本地端**：
   ```bash
   # 修改配置为 TEST_ONLY = True  
   python cotn_complete_tester.py
   # 分析生成的结果和图表
   ```

这样可以充分利用服务器的计算资源进行训练，同时在本地进行灵活的分析和实验。