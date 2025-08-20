#!/usr/bin/env python3
"""
独立的可视化工具
用于为已有的预测结果生成拟合图像和保存文件
"""

import os
import sys
import argparse
import numpy as np
from utils.visualization import (plot_prediction_results, plot_metrics_comparison, 
                               save_results_to_files, generate_comprehensive_report)

def parse_args():
    parser = argparse.ArgumentParser(description='COTN结果可视化工具')
    
    parser.add_argument('--results_dir', type=str, required=True,
                       help='结果目录路径 (包含pred.npy和true.npy文件)')
    parser.add_argument('--dataset_name', type=str, required=True,
                       help='数据集名称')
    parser.add_argument('--setting', type=str, required=True,
                       help='实验设置标识')
    parser.add_argument('--show_samples', type=int, default=5,
                       help='显示的样本数量')
    parser.add_argument('--pred_len', type=int, default=100,
                       help='预测长度')
    parser.add_argument('--output_plots', type=str, default='./plots',
                       help='图像输出目录')
    parser.add_argument('--output_export', type=str, default='./results_export',
                       help='结果导出目录')
    
    return parser.parse_args()


def load_results(results_dir):
    """加载预测结果文件"""
    pred_file = os.path.join(results_dir, 'pred.npy')
    true_file = os.path.join(results_dir, 'true.npy')
    
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"预测结果文件不存在: {pred_file}")
    if not os.path.exists(true_file):
        raise FileNotFoundError(f"真实标签文件不存在: {true_file}")
    
    pred_data = np.load(pred_file)
    true_data = np.load(true_file)
    
    print(f"✅ 已加载结果数据:")
    print(f"   预测数据形状: {pred_data.shape}")
    print(f"   真实数据形状: {true_data.shape}")
    
    return pred_data, true_data


def main():
    args = parse_args()
    
    print(f"COTN结果可视化工具")
    print(f"{'='*50}")
    print(f"结果目录: {args.results_dir}")
    print(f"数据集: {args.dataset_name}")
    print(f"设置: {args.setting}")
    print(f"{'='*50}")
    
    try:
        # 加载结果数据
        pred_data, true_data = load_results(args.results_dir)
        
        # 生成拟合图像
        print(f"📊 生成拟合图像...")
        plot_prediction_results(pred_data, true_data, args.dataset_name, args.setting, 
                              output_dir=args.output_plots, show_samples=args.show_samples, 
                              pred_len=args.pred_len)
        
        # 生成指标分析图
        print(f"📊 生成指标分析图...")
        plot_metrics_comparison(pred_data, true_data, args.dataset_name, args.setting, 
                              output_dir=args.output_plots)
        
        # 保存结果到多种格式
        print(f"💾 保存结果文件...")
        save_results_to_files(pred_data, true_data, args.dataset_name, args.setting, 
                            output_dir=args.output_export)
        
        # 生成综合报告
        print(f"📄 生成综合报告...")
        generate_comprehensive_report(args.dataset_name, args.setting, 
                                    args.results_dir, args.output_plots, args.output_export)
        
        print(f"\n🎉 可视化和结果保存完成！")
        print(f"   图像文件: {args.output_plots}/")
        print(f"   导出文件: {args.output_export}/")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()