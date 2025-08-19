import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class WalkForwardBacktest:
    """
    滚动回测框架 - 用于时间序列预测模型的性能评估
    模拟真实部署环境中的预测场景
    """
    
    def __init__(self, model, data, 
                 train_window=None, 
                 test_window=24,  # 预测6小时(24个15分钟)
                 step_size=24,    # 每次向前滚动6小时
                 retrain_freq=96): # 每24小时重新训练一次
        """
        初始化滚动回测
        
        Args:
            model: 预测模型（COTN或其他）
            data: 时间序列数据 DataFrame
            train_window: 训练窗口大小（None表示扩展窗口）
            test_window: 测试窗口大小
            step_size: 滚动步长
            retrain_freq: 重新训练频率
        """
        self.model = model
        self.data = data
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.retrain_freq = retrain_freq
        
        # 结果存储
        self.results = []
        self.predictions = []
        self.actuals = []
        self.timestamps = []
        
    def run_backtest(self, start_ratio=0.3, end_ratio=0.95):
        """
        执行滚动回测
        
        Args:
            start_ratio: 开始回测的数据比例（前30%作为初始训练集）
            end_ratio: 结束回测的数据比例
        """
        print("="*60)
        print("开始滚动回测...")
        print("="*60)
        
        data_length = len(self.data)
        start_idx = int(data_length * start_ratio)
        end_idx = int(data_length * end_ratio)
        
        print(f"数据总长度: {data_length}")
        print(f"回测起始点: {start_idx} ({start_ratio*100:.1f}%)")
        print(f"回测结束点: {end_idx} ({end_ratio*100:.1f}%)")
        print(f"回测窗口数: {(end_idx - start_idx) // self.step_size}")
        
        current_pos = start_idx
        step_count = 0
        retrain_count = 0
        
        while current_pos + self.test_window <= end_idx:
            step_count += 1
            
            # 确定训练集范围
            if self.train_window is None:
                # 扩展窗口：使用所有历史数据
                train_start = 0
                train_end = current_pos
            else:
                # 固定窗口：使用最近的训练窗口
                train_start = max(0, current_pos - self.train_window)
                train_end = current_pos
            
            # 确定测试集范围
            test_start = current_pos
            test_end = current_pos + self.test_window
            
            print(f"\nStep {step_count}:")
            print(f"  训练集: [{train_start}:{train_end}] ({train_end-train_start}个点)")
            print(f"  测试集: [{test_start}:{test_end}] ({test_end-test_start}个点)")
            
            # 准备训练和测试数据
            train_data = self.data.iloc[train_start:train_end]
            test_data = self.data.iloc[test_start:test_end]
            
            # 是否需要重新训练模型
            should_retrain = (step_count == 1 or 
                            (step_count - 1) % (self.retrain_freq // self.step_size) == 0)
            
            if should_retrain:
                retrain_count += 1
                print(f"  → 重新训练模型 (第{retrain_count}次)")
                try:
                    self.model.fit(train_data)
                    print(f"  ✓ 训练完成")
                except Exception as e:
                    print(f"  ✗ 训练失败: {e}")
                    current_pos += self.step_size
                    continue
            
            # 进行预测
            try:
                predictions = self.model.predict(test_data)
                actuals = test_data['Power'].values
                
                # 计算评估指标
                mae = mean_absolute_error(actuals, predictions)
                rmse = np.sqrt(mean_squared_error(actuals, predictions))
                r2 = r2_score(actuals, predictions)
                
                # 存储结果
                result = {
                    'step': step_count,
                    'timestamp': test_data.index[0] if hasattr(test_data.index, '__getitem__') else test_start,
                    'train_size': train_end - train_start,
                    'test_size': test_end - test_start,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'retrained': should_retrain
                }
                
                self.results.append(result)
                self.predictions.extend(predictions.tolist() if hasattr(predictions, 'tolist') else predictions)
                self.actuals.extend(actuals.tolist())
                self.timestamps.extend(test_data.index.tolist() if hasattr(test_data.index, 'tolist') else range(test_start, test_end))
                
                print(f"  ✓ 预测完成: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")
                
            except Exception as e:
                print(f"  ✗ 预测失败: {e}")
            
            # 移动到下一个位置
            current_pos += self.step_size
        
        print(f"\n滚动回测完成！")
        print(f"总步数: {len(self.results)}")
        print(f"重新训练次数: {retrain_count}")
        
        return self.get_summary()
    
    def get_summary(self):
        """获取回测结果汇总"""
        if not self.results:
            return None
        
        df_results = pd.DataFrame(self.results)
        
        summary = {
            'total_steps': len(self.results),
            'avg_mae': df_results['mae'].mean(),
            'avg_rmse': df_results['rmse'].mean(),
            'avg_r2': df_results['r2'].mean(),
            'std_mae': df_results['mae'].std(),
            'std_rmse': df_results['rmse'].std(),
            'std_r2': df_results['r2'].std(),
            'min_r2': df_results['r2'].min(),
            'max_r2': df_results['r2'].max(),
            'retraining_freq': self.retrain_freq // self.step_size
        }
        
        return summary, df_results
    
    def plot_results(self, n_points=500, save_path=None):
        """绘制回测结果"""
        if not self.predictions:
            print("没有预测结果可以绘制")
            return
        
        # 选择最后n个点进行绘制
        if len(self.predictions) > n_points:
            plot_pred = self.predictions[-n_points:]
            plot_actual = self.actuals[-n_points:]
            plot_time = self.timestamps[-n_points:]
        else:
            plot_pred = self.predictions
            plot_actual = self.actuals
            plot_time = self.timestamps
        
        # 创建图形
        plt.figure(figsize=(15, 10))
        
        # 子图1: 预测vs实际
        plt.subplot(3, 1, 1)
        plt.plot(plot_time, plot_actual, label='实际值', alpha=0.8, linewidth=1.2, color='blue')
        plt.plot(plot_time, plot_pred, label='预测值', alpha=0.8, linewidth=1.2, color='red')
        plt.title('滚动回测结果：预测vs实际')
        plt.ylabel('发电功率 (MW)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2: 预测误差
        plt.subplot(3, 1, 2)
        errors = np.array(plot_pred) - np.array(plot_actual)
        plt.plot(plot_time, errors, alpha=0.6, color='green', linewidth=1)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.title('预测误差 (预测值 - 实际值)')
        plt.ylabel('误差 (MW)')
        plt.grid(True, alpha=0.3)
        
        # 子图3: 滚动MAE
        if hasattr(self, 'results') and self.results:
            plt.subplot(3, 1, 3)
            df_results = pd.DataFrame(self.results)
            plt.plot(df_results['step'], df_results['mae'], marker='o', linewidth=1.5, markersize=3)
            plt.title('滚动MAE变化')
            plt.xlabel('回测步数')
            plt.ylabel('MAE (MW)')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"回测结果图已保存: {save_path}")
        
        plt.show()
        
        return plt.gcf()


class MockCOTNModel:
    """
    COTN模型的模拟接口，用于测试回测框架
    实际使用时替换为真正的COTN模型
    """
    
    def __init__(self):
        self.is_trained = False
    
    def fit(self, train_data):
        """训练模型"""
        # 模拟训练过程
        self.is_trained = True
        
    def predict(self, test_data):
        """预测"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        # 简单的移动平均预测（仅用于测试）
        power_values = test_data['Power'].values
        if len(power_values) > 0:
            # 使用最后一个值作为预测（持续预测）
            last_value = power_values[0] if len(power_values) > 0 else 0
            return np.full(len(test_data), last_value)
        else:
            return np.zeros(len(test_data))

def test_backtest_framework():
    """测试滚动回测框架"""
    print("测试滚动回测框架...")
    
    # 加载测试数据
    from prepare_pv_data import prepare_pv_data_for_cotn
    
    # 读取数据
    file_path = "/Users/tangbao/Desktop/PV_prediction/Solar Station/Solar station site 1 (Nominal capacity-50MW).xlsx"
    df = pd.read_excel(file_path, sheet_name='sheet1')
    
    # 数据预处理（简化版）
    df['Time'] = pd.to_datetime(df['Time(year-month-day h:m:s)'])
    df = df.rename(columns={'Power (MW)': 'Power'})
    df = df.set_index('Time').sort_index()
    
    # 创建模拟模型
    model = MockCOTNModel()
    
    # 创建回测器
    backtest = WalkForwardBacktest(
        model=model,
        data=df,
        train_window=None,  # 使用扩展窗口
        test_window=24,     # 预测6小时
        step_size=24,       # 每次向前6小时
        retrain_freq=96     # 每24小时重新训练
    )
    
    # 运行回测
    summary, results_df = backtest.run_backtest(start_ratio=0.5, end_ratio=0.7)
    
    print("\n回测结果汇总:")
    print(f"总步数: {summary['total_steps']}")
    print(f"平均MAE: {summary['avg_mae']:.3f} ± {summary['std_mae']:.3f}")
    print(f"平均R²: {summary['avg_r2']:.3f} ± {summary['std_r2']:.3f}")
    
    # 绘制结果
    backtest.plot_results(n_points=200, save_path='/Users/tangbao/Desktop/PV_prediction/COTN/backtest_test_results.png')
    
    return backtest, summary, results_df

if __name__ == "__main__":
    # 运行测试
    backtest, summary, results = test_backtest_framework()
    print("滚动回测框架测试完成！")