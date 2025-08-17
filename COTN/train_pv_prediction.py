from exp.exp_config import InformerConfig
from exp.exp_informer import Exp_Informer
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

def train_pv_prediction():
    """
    训练COTN模型进行光伏发电功率预测
    使用严格的时间序列分割（前80%训练，后20%测试）
    """
    print("="*60)
    print("COTN光伏发电功率预测模型训练")
    print("="*60)
    
    # 创建配置
    config = InformerConfig()
    
    # 根据是否有GPU调整设备设置
    config.use_gpu = True if torch.cuda.is_available() and config.use_gpu else False
    print(f"使用设备: {'GPU' if config.use_gpu else 'CPU'}")
    
    # 显示关键配置
    print(f"\n配置信息:")
    print(f"  数据集: {config.data}")
    print(f"  输入序列长度: {config.seq_len} (1天)")
    print(f"  预测序列长度: {config.pred_len} (6小时)")
    print(f"  特征数量: {config.enc_in} (单变量)")
    print(f"  预测模式: {config.features} (单变量)")
    print(f"  目标变量: {config.target}")
    print(f"  时间频率: {config.freq}")
    
    # 创建实验
    exp = Exp_Informer(config)
    
    # 设置当前的lee_type
    config.lee_type = 2
    setting = f'informer_PV_{config.data}_lee{config.lee_type}'
    
    print(f"\n开始训练模型: {setting}")
    print("-" * 40)
    
    # 训练模型
    print('>>>>>>>开始训练 : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    try:
        exp.train(setting)
        print("✓ 训练完成")
    except Exception as e:
        print(f"✗ 训练失败: {e}")
        return None
    
    # 测试模型
    print('>>>>>>>开始测试 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    try:
        test_results = exp.test(setting)
        print("✓ 测试完成")
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return None
    
    # 进行预测并生成详细评估
    print('>>>>>>>开始预测评估 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    try:
        # 获取测试数据和预测结果
        predictions, ground_truth = exp.predict(setting, load=True)
        
        # 计算评估指标
        if predictions is not None and ground_truth is not None:
            predictions = predictions.flatten() if len(predictions.shape) > 1 else predictions
            ground_truth = ground_truth.flatten() if len(ground_truth.shape) > 1 else ground_truth
            
            mae = mean_absolute_error(ground_truth, predictions)
            mse = mean_squared_error(ground_truth, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(ground_truth, predictions)
            
            print(f"\n" + "="*50)
            print("COTN光伏预测模型性能评估")
            print("="*50)
            print(f"MAE (平均绝对误差): {mae:.3f} MW")
            print(f"RMSE (均方根误差): {rmse:.3f} MW")
            print(f"R² (决定系数): {r2:.3f}")
            print(f"准确度: {r2*100:.1f}%")
            
            # 与现有随机森林模型对比
            print(f"\n对比基准（随机森林时间序列预测）:")
            print(f"  随机森林 R²: 0.871 (87.1%)")
            print(f"  随机森林 MAE: 1.988 MW")
            print(f"  COTN R²: {r2:.3f} ({r2*100:.1f}%)")
            print(f"  COTN MAE: {mae:.3f} MW")
            
            if r2 > 0.871:
                print(f"✓ COTN模型性能优于随机森林 (+{(r2-0.871)*100:.1f}%)")
            else:
                print(f"⚠ COTN模型性能低于随机森林 ({(r2-0.871)*100:.1f}%)")
            
            # 保存结果
            results = {
                'model': 'COTN',
                'setting': setting,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'accuracy': r2*100
            }
            
            # 可视化预测结果
            plot_prediction_results(ground_truth, predictions, setting, r2, mae)
            
            return results
            
    except Exception as e:
        print(f"✗ 预测评估失败: {e}")
        return None

def plot_prediction_results(ground_truth, predictions, setting, r2, mae, n_points=500):
    """
    绘制预测结果对比图
    """
    print(f"\n生成预测结果可视化...")
    
    # 取最后n个点进行可视化
    if len(ground_truth) > n_points:
        gt_plot = ground_truth[-n_points:]
        pred_plot = predictions[-n_points:]
    else:
        gt_plot = ground_truth
        pred_plot = predictions
    
    # 创建时间索引
    time_idx = range(len(gt_plot))
    
    # 创建图形
    plt.figure(figsize=(15, 10))
    
    # 子图1: 预测vs实际对比
    plt.subplot(2, 1, 1)
    plt.plot(time_idx, gt_plot, label='实际功率', alpha=0.8, linewidth=1.2, color='blue')
    plt.plot(time_idx, pred_plot, label='COTN预测', alpha=0.8, linewidth=1.2, color='red')
    plt.title(f'COTN光伏功率预测结果对比 (R²={r2:.3f}, MAE={mae:.3f}MW)')
    plt.xlabel('时间步 (15分钟间隔)')
    plt.ylabel('发电功率 (MW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 误差分析
    plt.subplot(2, 1, 2)
    errors = pred_plot - gt_plot
    plt.plot(time_idx, errors, alpha=0.6, color='green', linewidth=1)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title(f'预测误差分析 (预测值 - 实际值)')
    plt.xlabel('时间步 (15分钟间隔)')
    plt.ylabel('误差 (MW)')
    plt.grid(True, alpha=0.3)
    
    # 添加误差统计信息
    plt.text(0.02, 0.98, f'误差统计:\n平均误差: {np.mean(errors):.3f} MW\n误差标准差: {np.std(errors):.3f} MW', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图片
    output_path = f'/Users/tangbao/Desktop/PV_prediction/COTN/cotn_pv_prediction_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 预测结果图已保存: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    # 执行训练和评估
    results = train_pv_prediction()
    
    if results:
        print(f"\n" + "="*60)
        print("COTN光伏预测模型训练完成！")
        print(f"最终性能: R²={results['r2']:.3f}, MAE={results['mae']:.3f} MW")
        print("="*60)
    else:
        print("\n训练过程中出现错误，请检查配置和数据。")