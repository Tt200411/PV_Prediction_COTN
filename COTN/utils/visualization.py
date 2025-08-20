import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from utils.metrics import metric

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_prediction_results(pred_data, true_data, dataset_name, setting, 
                          output_dir="./plots", show_samples=5, pred_len=100):
    """
    绘制预测结果拟合图像
    
    参数:
    - pred_data: 预测结果数据 (shape: [n_samples, pred_len, features])
    - true_data: 真实标签数据 (shape: [n_samples, pred_len, features])
    - dataset_name: 数据集名称
    - setting: 实验设置标识
    - output_dir: 输出目录
    - show_samples: 显示的样本数量
    - pred_len: 预测长度
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算指标
    mae, mse, rmse, mape, mspe = metric(pred_data, true_data)
    
    # 选择要显示的样本
    total_samples = pred_data.shape[0]
    sample_indices = np.linspace(0, total_samples-1, show_samples, dtype=int)
    
    # 创建子图
    fig, axes = plt.subplots(show_samples, 1, figsize=(15, 3*show_samples))
    if show_samples == 1:
        axes = [axes]
    
    fig.suptitle(f'{dataset_name} - 预测结果拟合图\n{setting}\nMAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}', 
                 fontsize=14, fontweight='bold')
    
    for i, sample_idx in enumerate(sample_indices):
        ax = axes[i]
        
        # 获取当前样本的预测和真实值
        pred_sample = pred_data[sample_idx, :, 0]  # 假设只有一个特征
        true_sample = true_data[sample_idx, :, 0]
        
        # 创建时间轴
        x_axis = range(pred_len)
        
        # 绘制预测和真实值
        ax.plot(x_axis, true_sample, label='真实值', color='blue', linewidth=2, alpha=0.8)
        ax.plot(x_axis, pred_sample, label='预测值', color='red', linewidth=2, alpha=0.8, linestyle='--')
        
        # 计算当前样本的误差
        sample_mae = np.mean(np.abs(pred_sample - true_sample))
        
        # 设置图形属性
        ax.set_title(f'样本 {sample_idx+1} (MAE: {sample_mae:.4f})', fontsize=12)
        ax.set_xlabel('时间步长')
        ax.set_ylabel('功率值')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加填充区域显示误差
        ax.fill_between(x_axis, true_sample, pred_sample, alpha=0.2, color='gray', label='误差区域')
    
    plt.tight_layout()
    
    # 保存图像
    plot_filename = f"{dataset_name}_{setting}_prediction_fit.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 拟合图像已保存: {plot_path}")
    return plot_path


def plot_metrics_comparison(pred_data, true_data, dataset_name, setting, output_dir="./plots"):
    """
    绘制各种评价指标的对比图
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算指标
    mae, mse, rmse, mape, mspe = metric(pred_data, true_data)
    
    # 创建指标对比图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # MAE和MSE
    metrics = ['MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE']
    values = [mae, mse, rmse, mape, mspe]
    
    # 条形图
    ax1.bar(metrics[:3], values[:3], color=['skyblue', 'lightcoral', 'lightgreen'])
    ax1.set_title('损失指标 (MAE, MSE, RMSE)')
    ax1.set_ylabel('数值')
    
    # MAPE和MSPE
    ax2.bar(metrics[3:], values[3:], color=['orange', 'purple'])
    ax2.set_title('百分比误差 (MAPE, MSPE)')
    ax2.set_ylabel('百分比')
    
    # 误差分布直方图
    errors = (pred_data - true_data).flatten()
    ax3.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax3.set_title('预测误差分布')
    ax3.set_xlabel('误差值')
    ax3.set_ylabel('频次')
    ax3.axvline(0, color='red', linestyle='--', linewidth=2)
    
    # 散点图 (预测 vs 真实)
    pred_flat = pred_data.flatten()[:10000]  # 取样显示
    true_flat = true_data.flatten()[:10000]
    ax4.scatter(true_flat, pred_flat, alpha=0.5, s=1)
    ax4.plot([true_flat.min(), true_flat.max()], [true_flat.min(), true_flat.max()], 
             'r--', linewidth=2, label='理想拟合线')
    ax4.set_xlabel('真实值')
    ax4.set_ylabel('预测值')
    ax4.set_title('预测值 vs 真实值')
    ax4.legend()
    
    plt.suptitle(f'{dataset_name} - {setting} 评价指标汇总', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存图像
    metrics_filename = f"{dataset_name}_{setting}_metrics_analysis.png"
    metrics_path = os.path.join(output_dir, metrics_filename)
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 指标分析图已保存: {metrics_path}")
    return metrics_path


def save_results_to_files(pred_data, true_data, dataset_name, setting, output_dir="./results_export"):
    """
    将预测结果保存为多种格式
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算指标
    mae, mse, rmse, mape, mspe = metric(pred_data, true_data)
    
    # 准备数据
    results = {
        'dataset_info': {
            'dataset_name': dataset_name,
            'setting': setting,
            'total_samples': int(pred_data.shape[0]),
            'prediction_length': int(pred_data.shape[1]),
            'features': int(pred_data.shape[2])
        },
        'metrics': {
            'MAE': float(mae),
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAPE': float(mape),
            'MSPE': float(mspe)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # 保存JSON格式的结果摘要
    json_filename = f"{dataset_name}_{setting}_summary.json"
    json_path = os.path.join(output_dir, json_filename)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON摘要已保存: {json_path}")
    
    # 保存CSV格式的详细数据
    # 展平数据用于CSV保存
    n_samples, pred_len, n_features = pred_data.shape
    
    csv_data = []
    for sample_idx in range(n_samples):
        for time_step in range(pred_len):
            for feature_idx in range(n_features):
                csv_data.append({
                    'dataset': dataset_name,
                    'setting': setting,
                    'sample_id': sample_idx,
                    'time_step': time_step,
                    'feature_id': feature_idx,
                    'predicted_value': float(pred_data[sample_idx, time_step, feature_idx]),
                    'true_value': float(true_data[sample_idx, time_step, feature_idx]),
                    'absolute_error': float(abs(pred_data[sample_idx, time_step, feature_idx] - 
                                               true_data[sample_idx, time_step, feature_idx]))
                })
    
    df = pd.DataFrame(csv_data)
    csv_filename = f"{dataset_name}_{setting}_detailed_results.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"✅ CSV详细结果已保存: {csv_path}")
    
    # 保存精简版CSV (样本级统计)
    sample_stats = []
    for sample_idx in range(n_samples):
        pred_sample = pred_data[sample_idx]
        true_sample = true_data[sample_idx]
        
        sample_mae = np.mean(np.abs(pred_sample - true_sample))
        sample_mse = np.mean((pred_sample - true_sample)**2)
        sample_rmse = np.sqrt(sample_mse)
        
        sample_stats.append({
            'dataset': dataset_name,
            'setting': setting,
            'sample_id': sample_idx,
            'sample_mae': sample_mae,
            'sample_mse': sample_mse,
            'sample_rmse': sample_rmse,
            'pred_mean': float(np.mean(pred_sample)),
            'pred_std': float(np.std(pred_sample)),
            'true_mean': float(np.mean(true_sample)),
            'true_std': float(np.std(true_sample))
        })
    
    stats_df = pd.DataFrame(sample_stats)
    stats_filename = f"{dataset_name}_{setting}_sample_statistics.csv"
    stats_path = os.path.join(output_dir, stats_filename)
    stats_df.to_csv(stats_path, index=False, encoding='utf-8')
    print(f"✅ 样本统计CSV已保存: {stats_path}")
    
    return {
        'json_path': json_path,
        'csv_path': csv_path,
        'stats_path': stats_path,
        'results': results
    }


def generate_comprehensive_report(dataset_name, setting, results_dir, plots_dir, export_dir):
    """
    生成综合报告
    """
    report_content = f"""
# {dataset_name} 预测结果报告

## 实验设置
- 数据集: {dataset_name}
- 配置: {setting}
- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 文件结构
- 结果数据: `{results_dir}/`
- 可视化图像: `{plots_dir}/`
- 导出文件: `{export_dir}/`

## 生成的文件
1. 拟合图像: `{dataset_name}_{setting}_prediction_fit.png`
2. 指标分析图: `{dataset_name}_{setting}_metrics_analysis.png`
3. 结果摘要: `{dataset_name}_{setting}_summary.json`
4. 详细数据: `{dataset_name}_{setting}_detailed_results.csv`
5. 样本统计: `{dataset_name}_{setting}_sample_statistics.csv`

## 使用说明
- 查看拟合效果请参考PNG图像文件
- 详细指标数据请查看JSON和CSV文件
- 可以使用CSV文件进行进一步的数据分析

---
由COTN光伏预测框架自动生成
"""
    
    report_filename = f"{dataset_name}_{setting}_report.md"
    report_path = os.path.join(export_dir, report_filename)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✅ 综合报告已保存: {report_path}")
    return report_path