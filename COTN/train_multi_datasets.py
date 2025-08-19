#!/usr/bin/env python3
"""
COTN多数据集训练脚本
修复了train/vali/test分割逻辑，支持命令行参数和批量训练
"""

import argparse
import os
import sys
import traceback
from exp.exp_config import InformerConfig
from exp.exp_informer import Exp_Informer
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='COTN光伏发电预测训练')
    
    # 基本参数
    parser.add_argument('--datasets', type=str, nargs='+', 
                       default=['Site_1_50MW'], 
                       help='要训练的数据集列表 (可多个)')
    parser.add_argument('--activation', type=str, default='lee', 
                       choices=['lee', 'relu'], 
                       help='激活函数类型')
    parser.add_argument('--lee_type', type=int, default=3, 
                       help='Lee振荡器类型 (1-8)')
    parser.add_argument('--pred_len', type=int, default=100, 
                       help='预测长度')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=6, 
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='批大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                       help='学习率')
    parser.add_argument('--patience', type=int, default=3, 
                       help='早停耐心值')
    
    # 模式选择
    parser.add_argument('--train_only', action='store_true', 
                       help='仅训练模式')
    parser.add_argument('--test_only', action='store_true', 
                       help='仅测试模式')
    parser.add_argument('--skip_test', action='store_true', 
                       help='跳过测试阶段')
    
    # GPU设置
    parser.add_argument('--use_gpu', type=bool, default=True, 
                       help='是否使用GPU')
    parser.add_argument('--gpu', type=int, default=0, 
                       help='GPU设备ID')
    
    # 其他设置
    parser.add_argument('--save_results', action='store_true', 
                       help='保存详细结果')
    parser.add_argument('--verbose', action='store_true', 
                       help='详细输出')
    
    return parser.parse_args()


def create_config(dataset_name, args):
    """为指定数据集创建配置"""
    config = InformerConfig()
    
    # 数据集配置
    config.data = dataset_name
    config.data_path = f'{dataset_name}.csv'
    
    # 模型配置
    config.pred_len = args.pred_len
    config.train_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.patience = args.patience
    
    # 激活函数配置
    if args.activation == 'relu':
        config.activation = 'relu'
        config.lee_oscillator = False
        config.use_relu = True
    else:
        config.activation = 'Lee'
        config.lee_oscillator = True
        config.use_relu = False
        config.lee_type = args.lee_type
    
    # GPU配置
    config.use_gpu = torch.cuda.is_available() and args.use_gpu
    config.gpu = args.gpu
    config.device = f'cuda:{args.gpu}'
    
    # 文件名配置
    config.include_pred_len_in_filename = True
    
    if args.verbose:
        print(f"配置详情 - {dataset_name}:")
        print(f"  预测长度: {config.pred_len}")
        print(f"  激活函数: {config.activation}")
        print(f"  训练轮数: {config.train_epochs}")
        print(f"  设备: {'GPU' if config.use_gpu else 'CPU'}")
    
    return config


def train_single_dataset(dataset_name, args):
    """训练单个数据集"""
    print(f"\n{'='*80}")
    print(f"开始训练数据集: {dataset_name}")
    print(f"{'='*80}")
    
    try:
        # 创建配置
        config = create_config(dataset_name, args)
        
        # 检查数据文件是否存在
        data_file = os.path.join(config.root_path, config.data_path)
        if not os.path.exists(data_file):
            print(f"❌ 数据文件不存在: {data_file}")
            return None
        
        # 创建实验实例
        exp = Exp_Informer(config)
        
        # 生成设置标识
        activation_suffix = 'relu' if getattr(config, 'use_relu', False) else f'lee{config.lee_type}'
        setting = f'informer_{dataset_name}_{activation_suffix}_pred{config.pred_len}'
        
        print(f"训练设置: {setting}")
        
        result = {'dataset': dataset_name, 'setting': setting}
        
        # 训练阶段
        if not args.test_only:
            print(f"🚀 开始训练...")
            import time
            start_time = time.time()
            
            train_result = exp.train(setting)
            
            end_time = time.time()
            training_time = end_time - start_time
            print(f"✅ 训练完成，耗时: {training_time:.2f}秒")
            
            result['train_result'] = train_result
        
        # 测试阶段
        if not args.train_only and not args.skip_test:
            print(f"🧪 开始测试...")
            test_result = exp.test(setting)
            print(f"✅ 测试完成")
            result['test_result'] = test_result
            
            # 预测评估
            print(f"📊 生成预测结果...")
            pred_result = exp.predict(setting, load=True)
            result['pred_result'] = pred_result
        
        return result
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        if args.verbose:
            traceback.print_exc()
        return None


def main():
    args = parse_args()
    
    print(f"COTN多数据集训练工具")
    print(f"{'='*80}")
    print(f"数据集: {args.datasets}")
    print(f"激活函数: {args.activation}")
    print(f"预测长度: {args.pred_len}")
    print(f"训练模式: {'仅训练' if args.train_only else '仅测试' if args.test_only else '训练+测试'}")
    print(f"GPU加速: {torch.cuda.is_available()}")
    print(f"{'='*80}")
    
    # 检查是否有可用的数据集
    available_datasets = []
    data_dir = "ETT-small"
    
    for dataset in args.datasets:
        csv_file = os.path.join(data_dir, f"{dataset}.csv")
        if os.path.exists(csv_file):
            available_datasets.append(dataset)
        else:
            print(f"⚠️  数据集 {dataset} 的CSV文件不存在: {csv_file}")
    
    if not available_datasets:
        print("❌ 没有可用的数据集，请检查数据文件")
        return
    
    print(f"✅ 找到 {len(available_datasets)} 个可用数据集")
    
    # 训练结果统计
    successful_trainings = 0
    failed_trainings = 0
    results = {}
    
    # 开始批量训练
    for i, dataset in enumerate(available_datasets, 1):
        print(f"\n进度: [{i}/{len(available_datasets)}]")
        
        result = train_single_dataset(dataset, args)
        
        if result:
            results[dataset] = result
            successful_trainings += 1
            print(f"✅ {dataset} 训练成功")
        else:
            failed_trainings += 1
            print(f"❌ {dataset} 训练失败")
    
    # 总结报告
    print(f"\n{'='*80}")
    print(f"训练完成总结")
    print(f"{'='*80}")
    print(f"总数据集: {len(available_datasets)}")
    print(f"成功训练: {successful_trainings}")
    print(f"失败训练: {failed_trainings}")
    print(f"成功率: {successful_trainings/len(available_datasets)*100:.1f}%")
    
    if args.save_results and results:
        # 保存详细结果
        import json
        result_file = f"training_results_{args.activation}_pred{args.pred_len}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            # 简化results以便JSON序列化
            simple_results = {}
            for dataset, result in results.items():
                simple_results[dataset] = {
                    'dataset': result['dataset'],
                    'setting': result['setting'],
                    'success': True
                }
            json.dump(simple_results, f, indent=2, ensure_ascii=False)
        print(f"📄 详细结果已保存: {result_file}")
    
    print(f"{'='*80}")


if __name__ == "__main__":
    main()