#!/bin/bash

# COTN多数据集训练脚本
# 使用方法: ./train_all.sh [选项]

echo "COTN光伏预测多数据集训练脚本"
echo "=================================="

# 检查Python环境
if command -v /opt/miniconda3/envs/torch/bin/python &> /dev/null; then
    PYTHON_CMD="/opt/miniconda3/envs/torch/bin/python"
    echo "✅ 使用torch环境Python"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    echo "⚠️  使用系统默认Python"
else
    echo "❌ 未找到Python，请检查环境"
    exit 1
fi

# 检查CUDA
if $PYTHON_CMD -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')" 2>/dev/null; then
    echo "✅ CUDA环境检查完成"
else
    echo "⚠️  CUDA环境异常，将使用CPU"
fi

echo "=================================="

# 默认参数
DATASETS="Site_1_50MW Site_2_130MW Site_3_30MW Site_4_130MW Site_5_110MW Site_6_35MW Site_7_30MW Site_8_30MW"
ACTIVATION="lee"
PRED_LEN=100
EPOCHS=6
MODE="train_test"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --activation)
            ACTIVATION="$2"
            shift 2
            ;;
        --pred-len)
            PRED_LEN="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --train-only)
            MODE="train_only"
            shift
            ;;
        --test-only)
            MODE="test_only"
            shift
            ;;
        --help)
            echo "使用方法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --datasets NAMES     指定数据集 (默认: 所有8个数据集)"
            echo "  --activation TYPE    激活函数 lee|relu (默认: lee)"
            echo "  --pred-len N         预测长度 (默认: 100)"
            echo "  --epochs N           训练轮数 (默认: 6)"
            echo "  --train-only         仅训练模式"
            echo "  --test-only          仅测试模式"
            echo "  --help               显示帮助"
            echo ""
            echo "示例:"
            echo "  $0                                    # 训练所有数据集"
            echo "  $0 --datasets Site_1_50MW            # 仅训练Site 1"
            echo "  $0 --activation relu --epochs 10     # 使用ReLU，训练10轮"
            echo "  $0 --train-only --pred-len 50        # 仅训练，预测长度50"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 构建Python命令
PYTHON_ARGS="--datasets $DATASETS --activation $ACTIVATION --pred_len $PRED_LEN --epochs $EPOCHS --verbose"

if [[ "$MODE" == "train_only" ]]; then
    PYTHON_ARGS="$PYTHON_ARGS --train_only"
elif [[ "$MODE" == "test_only" ]]; then
    PYTHON_ARGS="$PYTHON_ARGS --test_only"
fi

echo "训练配置:"
echo "  数据集: $DATASETS"
echo "  激活函数: $ACTIVATION"
echo "  预测长度: $PRED_LEN"
echo "  训练轮数: $EPOCHS"
echo "  模式: $MODE"
echo "=================================="

# 执行训练
echo "开始训练..."
$PYTHON_CMD train_multi_datasets.py $PYTHON_ARGS

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 训练完成！"
    echo ""
    echo "生成的模型文件位于: checkpoints/"
    echo "查看具体模型: ls -la checkpoints/"
else
    echo ""
    echo "❌ 训练过程中出现错误"
    exit 1
fi