from exp.exp_config import InformerConfig
from exp.exp_informer import Exp_Informer
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

class COTNModelWrapper:
    """
    COTN模型包装器，适配回测和多数据集测试框架
    """
    
    def __init__(self, dataset_name, config_overrides=None, train_only=False, test_only=False, use_relu=False):
        self.dataset_name = dataset_name
        self.config = InformerConfig()
        self.exp = None
        self.setting = None
        self.trained = False
        
        # 设置训练/测试模式
        self.train_only = train_only
        self.test_only = test_only
        self.train_and_test = not (train_only or test_only)
        
        # 设置激活函数
        if use_relu:
            self.config.activation = 'relu'
            self.config.lee_oscillator = False
            self.config.use_relu = True
        
        # 应用配置覆盖
        if config_overrides:
            for key, value in config_overrides.items():
                setattr(self.config, key, value)
        
        # 更新数据集相关配置
        self.config.data = dataset_name
        self.config.data_path = f'{dataset_name}.csv'
        
        # 设置设备
        self.config.use_gpu = True if torch.cuda.is_available() and self.config.use_gpu else False
        
        print(f"初始化COTN模型: {dataset_name}")
        print(f"使用设备: {'GPU' if self.config.use_gpu else 'CPU'}")
        print(f"激活函数: {self.config.activation}")
        print(f"运行模式: {'仅训练' if train_only else '仅测试' if test_only else '训练+测试'}")
    
    def train(self):
        """训练COTN模型"""
        try:
            print(f"开始训练COTN模型: {self.dataset_name}")
            
            # 创建实验实例
            self.exp = Exp_Informer(self.config)
            
            # 设置训练标识和文件名
            self.config.lee_type = 2
            activation_suffix = 'relu' if getattr(self.config, 'use_relu', False) else f'lee{self.config.lee_type}'
            
            if getattr(self.config, 'include_pred_len_in_filename', True):
                self.setting = f'informer_{self.dataset_name}_{activation_suffix}_pred{self.config.pred_len}'
            else:
                self.setting = f'informer_{self.dataset_name}_{activation_suffix}'
            
            print(f"训练设置: {self.setting}")
            
            # 执行训练
            self.exp.train(self.setting)
            self.trained = True
            
            print(f"✓ 训练完成: {self.setting}")
            return {'status': 'success', 'setting': self.setting}
            
        except Exception as e:
            print(f"✗ 训练失败: {e}")
            return None
    
    def test(self):
        """测试COTN模型"""
        if not self.trained or not self.exp:
            print("模型尚未训练，无法测试")
            return None
        
        try:
            print(f"开始测试COTN模型: {self.setting}")
            
            # 执行测试
            test_result = self.exp.test(self.setting)
            
            print(f"✓ 测试完成: {self.setting}")
            return {'status': 'success', 'test_result': test_result}
            
        except Exception as e:
            print(f"✗ 测试失败: {e}")
            return None
    
    def predict_and_evaluate(self):
        """预测并评估性能"""
        if not self.trained or not self.exp:
            print("模型尚未训练，无法预测")
            return None
        
        try:
            print(f"开始预测评估: {self.setting}")
            
            # 执行预测
            predictions, ground_truth = self.exp.predict(self.setting, load=True)
            
            if predictions is None or ground_truth is None:
                print("预测失败，返回结果为空")
                return None
            
            # 处理预测结果格式
            predictions = predictions.flatten() if len(predictions.shape) > 1 else predictions
            ground_truth = ground_truth.flatten() if len(ground_truth.shape) > 1 else ground_truth
            
            # 计算评估指标
            mae = mean_absolute_error(ground_truth, predictions)
            mse = mean_squared_error(ground_truth, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(ground_truth, predictions)
            
            result = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'predictions': predictions,
                'ground_truth': ground_truth
            }
            
            print(f"✓ 预测评估完成:")
            print(f"  MAE: {mae:.3f} MW")
            print(f"  RMSE: {rmse:.3f} MW") 
            print(f"  R²: {r2:.3f}")
            
            return result
            
        except Exception as e:
            print(f"✗ 预测评估失败: {e}")
            return None
    
    def run_full_pipeline(self):
        """运行完整的训练+测试流程"""
        results = {}
        
        # 根据模式选择执行步骤
        if self.train_only or self.train_and_test:
            train_result = self.train()
            results['train_result'] = train_result
            if not train_result:
                print("训练失败，终止流程")
                return results
        
        if self.test_only or self.train_and_test:
            if self.test_only and not self.trained:
                # 仅测试模式下，需要加载已训练的模型
                print("仅测试模式，尝试加载已训练模型...")
                self.exp = Exp_Informer(self.config)
                self.setting = f'informer_{self.dataset_name}_lee{self.config.lee_type}'
                self.trained = True
            
            test_result = self.test()
            results['test_result'] = test_result
            
            if test_result:
                pred_result = self.predict_and_evaluate()
                results['pred_result'] = pred_result
        
        return results
    
    def fit(self, train_data):
        """适配回测框架的训练接口"""
        # 对于回测，每次都重新训练
        # 注意：这可能会很慢，实际应用中可能需要增量学习
        return self.train()
    
    def predict(self, test_data):
        """适配回测框架的预测接口"""
        if not self.trained:
            raise ValueError("模型尚未训练")
        
        # 简化的预测逻辑，实际中需要根据COTN的预测接口调整
        # 这里返回一个简单的预测结果
        try:
            # 获取最新的预测结果
            result = self.predict_and_evaluate()
            if result and 'predictions' in result:
                # 返回预测长度对应的预测值
                pred_len = min(len(test_data), len(result['predictions']))
                return result['predictions'][:pred_len]
            else:
                # 如果预测失败，返回零值
                return np.zeros(len(test_data))
        except:
            return np.zeros(len(test_data))


def run_cotn_on_single_dataset(dataset_name, test_ratio=0.2, train_only=False, test_only=False, use_relu=False):
    """在单个数据集上运行完整的COTN测试"""
    print("="*60)
    print(f"COTN单数据集测试: {dataset_name}")
    print("="*60)
    
    # 创建COTN模型
    model = COTNModelWrapper(dataset_name, train_only=train_only, test_only=test_only, use_relu=use_relu)
    
    # 运行完整流程
    results = model.run_full_pipeline()
    
    if results:
        results.update({
            'dataset': dataset_name,
            'model': model
        })
    
    return results


def run_cotn_multi_dataset_test(train_only=False, test_only=False, use_relu=False):
    """运行COTN多数据集测试"""
    from multi_dataset_tester import MultiDatasetTester
    
    print("="*60)
    print("COTN多数据集测试")
    print(f"运行模式: {'仅训练' if train_only else '仅测试' if test_only else '训练+测试'}")
    print(f"激活函数: {'ReLU' if use_relu else 'Lee振荡器'}")
    print("="*60)
    
    # 创建多数据集测试器
    tester = MultiDatasetTester()
    
    # 创建带参数的模型构造函数
    def create_cotn_model(dataset_name):
        return COTNModelWrapper(dataset_name, train_only=train_only, test_only=test_only, use_relu=use_relu)
    
    # 使用真实的COTN模型进行测试
    tester.test_all_datasets(create_cotn_model, test_ratio=0.2)
    
    return tester


def run_cotn_backtest():
    """运行COTN滚动回测"""
    from backtest_framework import WalkForwardBacktest
    
    print("="*60)
    print("COTN滚动回测")
    print("="*60)
    
    # 加载测试数据（使用Site 1）
    dataset_name = "Site_1_50MW"
    
    # 准备数据
    file_path = "/Users/tangbao/Desktop/PV_prediction/Solar Station/Solar station site 1 (Nominal capacity-50MW).xlsx"
    df = pd.read_excel(file_path, sheet_name='sheet1')
    
    # 数据预处理
    df['Time'] = pd.to_datetime(df['Time(year-month-day h:m:s)'])
    df = df.rename(columns={'Power (MW)': 'Power'})
    df = df.set_index('Time').sort_index()
    
    # 创建COTN模型
    model = COTNModelWrapper(dataset_name)
    
    # 由于COTN训练时间较长，使用较小的回测窗口
    backtest = WalkForwardBacktest(
        model=model,
        data=df,
        train_window=None,      # 使用扩展窗口
        test_window=24,         # 预测6小时
        step_size=96,           # 每次向前24小时（减少回测次数）
        retrain_freq=288        # 每72小时重新训练一次
    )
    
    # 运行回测（使用较小的范围）
    summary, results_df = backtest.run_backtest(start_ratio=0.6, end_ratio=0.8)
    
    if summary:
        print(f"\n回测结果:")
        print(f"平均MAE: {summary['avg_mae']:.3f}")
        print(f"平均R²: {summary['avg_r2']:.3f}")
        
        # 绘制结果
        backtest.plot_results(
            n_points=200, 
            save_path='/Users/tangbao/Desktop/PV_prediction/COTN/cotn_backtest_results.png'
        )
    
    return backtest, summary, results_df


def create_comprehensive_report(single_results=None, multi_results=None, backtest_results=None):
    """创建综合测试报告"""
    print("\n" + "="*80)
    print("COTN光伏预测模型综合测试报告")
    print("="*80)
    
    report_sections = []
    
    # 1. 单数据集测试结果
    if single_results:
        report_sections.append(f"""
1. 单数据集测试结果 ({single_results['dataset']})
   - 训练状态: {single_results['train_result']['status']}
   - 测试状态: {single_results['test_result']['status']}
   - MAE: {single_results['pred_result']['mae']:.3f} MW
   - RMSE: {single_results['pred_result']['rmse']:.3f} MW
   - R²: {single_results['pred_result']['r2']:.3f}
   - 准确度: {single_results['pred_result']['r2']*100:.1f}%
        """)
    
    # 2. 多数据集测试结果
    if multi_results and hasattr(multi_results, 'results'):
        successful_tests = len([r for r in multi_results.results.values() if r])
        total_tests = len(multi_results.datasets)
        
        if successful_tests > 0:
            avg_r2 = np.mean([r['pred_result']['r2'] for r in multi_results.results.values() 
                             if r and r['pred_result']])
            
            report_sections.append(f"""
2. 多数据集测试结果
   - 测试数据集数量: {total_tests}
   - 成功测试数量: {successful_tests}
   - 成功率: {successful_tests/total_tests*100:.1f}%
   - 平均R²: {avg_r2:.3f}
            """)
    
    # 3. 滚动回测结果
    if backtest_results:
        summary, results_df, backtest = backtest_results
        if summary:
            report_sections.append(f"""
3. 滚动回测结果
   - 回测步数: {summary['total_steps']}
   - 平均MAE: {summary['avg_mae']:.3f} MW
   - 平均RMSE: {summary['avg_rmse']:.3f} MW  
   - 平均R²: {summary['avg_r2']:.3f}
   - R²标准差: {summary['std_r2']:.3f}
   - 重新训练频率: 每{summary['retraining_freq']}步
            """)
    
    # 输出报告
    for section in report_sections:
        print(section)
    
    # 保存报告到文件
    report_content = "COTN光伏预测模型综合测试报告\n" + "="*50 + "\n" + "\n".join(report_sections)
    
    report_path = "/Users/tangbao/Desktop/PV_prediction/COTN/comprehensive_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n综合报告已保存: {report_path}")
    print("="*80)


if __name__ == "__main__":
    print("COTN完整测试套件")
    print("选择测试模式:")
    print("1. 单数据集测试")
    print("2. 多数据集测试") 
    print("3. 滚动回测")
    print("4. 综合测试（全部）")
    print("\n运行选项:")
    print("- 训练+测试模式 (默认)")
    print("- 仅训练模式 (train_only=True)")
    print("- 仅测试模式 (test_only=True)")
    print("- 激活函数选择: Lee振荡器 (默认) 或 ReLU (use_relu=True)")
    
    # 配置选项
    TRAIN_ONLY = False     # 设置为True仅执行训练
    TEST_ONLY = False      # 设置为True仅执行测试  
    USE_RELU = False       # 设置为True使用ReLU替代Lee振荡器
    TEST_MODE = "single"   # "single", "multi", "backtest", "all"
    
    print(f"\n当前配置:")
    print(f"- 运行模式: {'仅训练' if TRAIN_ONLY else '仅测试' if TEST_ONLY else '训练+测试'}")
    print(f"- 激活函数: {'ReLU' if USE_RELU else 'Lee振荡器'}")
    print(f"- 测试模式: {TEST_MODE}")
    
    if TEST_MODE in ["single", "all"]:
        # 准备Site 1数据
        from multi_dataset_tester import MultiDatasetTester
        tester = MultiDatasetTester()
        tester.prepare_dataset_for_cotn("Site_1_50MW")
        
        # 运行单数据集测试
        print(f"\n{'='*60}")
        print("运行单数据集测试")
        print(f"{'='*60}")
        
        single_result = run_cotn_on_single_dataset(
            "Site_1_50MW", 
            train_only=TRAIN_ONLY, 
            test_only=TEST_ONLY, 
            use_relu=USE_RELU
        )
        
        if single_result:
            create_comprehensive_report(single_results=single_result)
        else:
            print("单数据集测试失败")
    
    if TEST_MODE in ["multi", "all"]:
        # 运行多数据集测试
        print(f"\n{'='*60}")
        print("运行多数据集测试")
        print(f"{'='*60}")
        
        multi_result = run_cotn_multi_dataset_test(
            train_only=TRAIN_ONLY,
            test_only=TEST_ONLY, 
            use_relu=USE_RELU
        )
        
        if multi_result:
            print("多数据集测试完成")
        else:
            print("多数据集测试失败")
    
    print(f"\n{'='*60}")
    print("所有测试完成！")
    print(f"{'='*60}")