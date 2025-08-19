import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class MultiDatasetTester:
    """
    多数据集测试器 - 在所有光伏电站数据上评估COTN模型性能
    """
    
    def __init__(self, data_dir="/Users/tangbao/Desktop/PV_prediction/Solar Station"):
        self.data_dir = data_dir
        self.datasets = {}
        self.results = {}
        self.load_all_datasets()
    
    def load_all_datasets(self):
        """加载所有数据集"""
        print("加载所有光伏电站数据集...")
        
        # 获取所有Excel文件
        excel_files = [f for f in os.listdir(self.data_dir) if f.endswith('.xlsx')]
        
        for file in excel_files:
            try:
                # 提取电站信息
                if 'site' in file.lower():
                    site_num = file.split('site')[1].split('(')[0].strip()
                    capacity = file.split('capacity-')[1].split('MW')[0]
                    dataset_name = f"Site_{site_num}_{capacity}MW"
                else:
                    dataset_name = file.replace('.xlsx', '')
                
                # 读取数据 - 自动检测sheet名称
                file_path = os.path.join(self.data_dir, file)
                
                # 尝试不同的sheet名称
                sheet_names_to_try = ['sheet1', 'Sheet1', 0]  # 也包括索引0作为后备
                df = None
                used_sheet = None
                
                for sheet_name in sheet_names_to_try:
                    try:
                        df = pd.read_excel(file_path, sheet_name=sheet_name)
                        used_sheet = sheet_name
                        break
                    except:
                        continue
                
                if df is None:
                    print(f"✗ 无法读取 {file} 的任何sheet")
                    continue
                
                # 数据预处理 - 灵活处理列名
                df['Time'] = pd.to_datetime(df['Time(year-month-day h:m:s)'])
                
                # 基础列名映射
                column_mapping = {
                    'Time(year-month-day h:m:s)': 'Time_Original',
                    'Total solar irradiance (W/m2)': 'Total_Irradiance',
                    'Direct normal irradiance (W/m2)': 'Direct_Irradiance', 
                    'Global horizontal irradiance (W/m2)': 'Global_Irradiance',
                    'Atmosphere (hpa)': 'Atmosphere',
                    'Power (MW)': 'Power'
                }
                
                # 处理温度列 - 可能有不同的名称或缺失
                temp_column = None
                possible_temp_columns = [
                    'Air temperature  (°C) ',
                    'Air temperature (°C)',
                    'Temperature (°C)',
                    'Relative humidity (%)'  # 某些数据集可能用湿度代替温度
                ]
                
                for col in possible_temp_columns:
                    if col in df.columns:
                        temp_column = col
                        break
                
                if temp_column:
                    column_mapping[temp_column] = 'Temperature'
                else:
                    # 如果没有温度列，创建一个默认值
                    df['Temperature'] = 25.0  # 默认25度
                    print(f"  警告: {dataset_name} 缺少温度列，使用默认值25°C")
                
                df = df.rename(columns=column_mapping)
                
                # 按时间排序
                df = df.sort_values('Time').reset_index(drop=True)
                
                # 基本统计信息
                stats = {
                    'capacity': int(capacity),
                    'data_points': len(df),
                    'time_range': (df['Time'].min(), df['Time'].max()),
                    'power_stats': {
                        'mean': df['Power'].mean(),
                        'max': df['Power'].max(),
                        'min': df['Power'].min(),
                        'std': df['Power'].std()
                    }
                }
                
                self.datasets[dataset_name] = {
                    'data': df,
                    'stats': stats,
                    'file_path': file_path,
                    'sheet_used': used_sheet
                }
                
                print(f"✓ {dataset_name}: {len(df)} 数据点, 装机容量 {capacity}MW (sheet: {used_sheet})")
                
            except Exception as e:
                print(f"✗ 加载 {file} 失败: {e}")
        
        print(f"\n总共加载了 {len(self.datasets)} 个数据集")
        self.print_dataset_summary()
    
    def print_dataset_summary(self):
        """打印数据集摘要"""
        print("\n" + "="*60)
        print("数据集摘要")
        print("="*60)
        
        total_capacity = 0
        total_points = 0
        
        for name, dataset in self.datasets.items():
            stats = dataset['stats']
            capacity = stats['capacity']
            points = stats['data_points']
            power_max = stats['power_stats']['max']
            
            total_capacity += capacity
            total_points += points
            
            print(f"{name:20} | 装机容量: {capacity:3d}MW | 数据点: {points:6d} | 最大功率: {power_max:6.2f}MW")
        
        print("-" * 60)
        print(f"{'总计':20} | 装机容量: {total_capacity:3d}MW | 数据点: {total_points:6d}")
        print("="*60)
    
    def prepare_dataset_for_cotn(self, dataset_name):
        """为指定数据集准备COTN格式的CSV文件"""
        if dataset_name not in self.datasets:
            raise ValueError(f"数据集 {dataset_name} 不存在")
        
        df = self.datasets[dataset_name]['data'].copy()
        
        # 创建COTN需要的格式
        cotn_df = pd.DataFrame()
        cotn_df['date'] = df['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        cotn_df['Total_Irradiance'] = df['Total_Irradiance']
        cotn_df['Global_Irradiance'] = df['Global_Irradiance']
        cotn_df['Direct_Irradiance'] = df['Direct_Irradiance']
        cotn_df['Temperature'] = df['Temperature']
        cotn_df['Atmosphere'] = df['Atmosphere']
        cotn_df['Power'] = df['Power']
        
        # 数据清理
        for col in ['Total_Irradiance', 'Global_Irradiance', 'Direct_Irradiance']:
            cotn_df[col] = cotn_df[col].clip(lower=0)
        cotn_df['Power'] = cotn_df['Power'].clip(lower=0)
        
        # 保存CSV文件
        output_path = f"/Users/tangbao/Desktop/PV_prediction/COTN/ETT-small/{dataset_name}.csv"
        cotn_df.to_csv(output_path, index=False)
        
        print(f"✓ {dataset_name} 数据已保存到: {output_path}")
        return output_path
    
    def prepare_all_datasets_for_cotn(self):
        """为所有数据集准备COTN格式的CSV文件"""
        print("为所有数据集准备COTN格式文件...")
        prepared_count = 0
        
        for dataset_name in self.datasets.keys():
            try:
                csv_path = self.prepare_dataset_for_cotn(dataset_name)
                if csv_path:
                    prepared_count += 1
            except Exception as e:
                print(f"✗ 准备 {dataset_name} 失败: {e}")
        
        print(f"成功准备了 {prepared_count}/{len(self.datasets)} 个数据集")
        return prepared_count
    
    def test_single_dataset(self, dataset_name, model_class, test_ratio=0.2):
        """在单个数据集上测试模型"""
        print(f"\n测试数据集: {dataset_name}")
        print("-" * 40)
        
        # 检查CSV文件是否存在，不存在则准备
        csv_path = f"/Users/tangbao/Desktop/PV_prediction/COTN/ETT-small/{dataset_name}.csv"
        if not os.path.exists(csv_path):
            print(f"CSV文件不存在，正在准备: {dataset_name}")
            csv_path = self.prepare_dataset_for_cotn(dataset_name)
        else:
            print(f"使用已存在的CSV文件: {csv_path}")
        
        try:
            # 创建并训练模型
            if callable(model_class):
                # 如果传入的是函数（新的方式）
                model = model_class(dataset_name)
            else:
                # 如果传入的是类（旧的方式）
                model = model_class(dataset_name)
            
            # 获取数据集信息
            dataset = self.datasets[dataset_name]
            data_length = len(dataset['data'])
            split_point = int(data_length * (1 - test_ratio))
            
            print(f"数据点总数: {data_length}")
            print(f"训练集: 0 -> {split_point} ({(1-test_ratio)*100:.0f}%)")
            print(f"测试集: {split_point} -> {data_length} ({test_ratio*100:.0f}%)")
            
            # 运行模型流程
            if hasattr(model, 'run_full_pipeline'):
                # 新版本的COTNModelWrapper
                print("使用新版本的模型接口...")
                result_dict = model.run_full_pipeline()
                
                # 提取结果
                train_result = result_dict.get('train_result')
                test_result = result_dict.get('test_result')
                pred_result = result_dict.get('pred_result')
                
            else:
                # 旧版本或模拟模型
                print("使用传统的训练-测试流程...")
                
                # 训练模型
                print("开始训练...")
                train_result = model.train()
                
                if train_result is None:
                    print("✗ 训练失败")
                    return None
                
                # 测试模型
                print("开始测试...")
                test_result = model.test()
                
                if test_result is None:
                    print("✗ 测试失败")
                    return None
                
                # 进行预测评估
                print("生成预测...")
                pred_result = model.predict_and_evaluate()
            
            # 汇总结果
            result = {
                'dataset': dataset_name,
                'capacity': dataset['stats']['capacity'],
                'data_points': data_length,
                'train_size': split_point,
                'test_size': data_length - split_point,
                'train_result': train_result,
                'test_result': test_result,
                'pred_result': pred_result
            }
            
            self.results[dataset_name] = result
            
            # 打印结果
            if pred_result:
                print(f"✓ 测试完成:")
                print(f"  MAE: {pred_result['mae']:.3f} MW")
                print(f"  RMSE: {pred_result['rmse']:.3f} MW")
                print(f"  R²: {pred_result['r2']:.3f}")
                print(f"  准确度: {pred_result['r2']*100:.1f}%")
            
            return result
            
        except Exception as e:
            print(f"✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_all_datasets(self, model_class, test_ratio=0.2):
        """在所有数据集上测试模型"""
        print("\n" + "="*60)
        print("开始多数据集测试")
        print("="*60)
        
        # 首先准备所有数据集的CSV文件
        print("准备所有数据集的CSV文件...")
        self.prepare_all_datasets_for_cotn()
        
        successful_tests = 0
        failed_tests = 0
        
        for dataset_name in self.datasets.keys():
            try:
                result = self.test_single_dataset(dataset_name, model_class, test_ratio)
                if result:
                    successful_tests += 1
                else:
                    failed_tests += 1
            except Exception as e:
                print(f"✗ {dataset_name} 测试异常: {e}")
                import traceback
                traceback.print_exc()
                failed_tests += 1
        
        print(f"\n多数据集测试完成!")
        print(f"成功: {successful_tests}, 失败: {failed_tests}")
        
        if successful_tests > 0:
            self.generate_comparison_report()
    
    def generate_comparison_report(self):
        """生成对比报告"""
        if not self.results:
            print("没有测试结果可以生成报告")
            return
        
        print("\n" + "="*80)
        print("多数据集性能对比报告")
        print("="*80)
        
        # 创建结果表格
        report_data = []
        for dataset_name, result in self.results.items():
            if result and result['pred_result']:
                pred = result['pred_result']
                report_data.append({
                    '数据集': dataset_name,
                    '装机容量(MW)': result['capacity'],
                    '数据点数': result['data_points'],
                    'MAE(MW)': pred['mae'],
                    'RMSE(MW)': pred['rmse'],
                    'R²': pred['r2'],
                    '准确度(%)': pred['r2'] * 100
                })
        
        if not report_data:
            print("没有有效的测试结果")
            return
        
        df_report = pd.DataFrame(report_data)
        
        # 打印详细表格
        print(f"{'数据集':20} | {'容量':>6} | {'数据点':>8} | {'MAE':>8} | {'RMSE':>8} | {'R²':>8} | {'准确度':>8}")
        print("-" * 80)
        
        for _, row in df_report.iterrows():
            print(f"{row['数据集']:20} | {row['装机容量(MW)']:6.0f} | {row['数据点数']:8.0f} | "
                  f"{row['MAE(MW)']:8.3f} | {row['RMSE(MW)']:8.3f} | {row['R²']:8.3f} | {row['准确度(%)']:7.1f}%")
        
        # 统计摘要
        print("-" * 80)
        print(f"{'平均':20} | {df_report['装机容量(MW)'].mean():6.1f} | {df_report['数据点数'].mean():8.0f} | "
              f"{df_report['MAE(MW)'].mean():8.3f} | {df_report['RMSE(MW)'].mean():8.3f} | "
              f"{df_report['R²'].mean():8.3f} | {df_report['准确度(%)'].mean():7.1f}%")
        
        print(f"{'标准差':20} | {df_report['装机容量(MW)'].std():6.1f} | {df_report['数据点数'].std():8.0f} | "
              f"{df_report['MAE(MW)'].std():8.3f} | {df_report['RMSE(MW)'].std():8.3f} | "
              f"{df_report['R²'].std():8.3f} | {df_report['准确度(%)'].std():7.1f}%")
        
        # 最佳和最差性能
        best_idx = df_report['R²'].idxmax()
        worst_idx = df_report['R²'].idxmin()
        
        print(f"\n最佳性能: {df_report.loc[best_idx, '数据集']} (R²={df_report.loc[best_idx, 'R²']:.3f})")
        print(f"最差性能: {df_report.loc[worst_idx, '数据集']} (R²={df_report.loc[worst_idx, 'R²']:.3f})")
        
        # 保存报告
        report_path = "/Users/tangbao/Desktop/PV_prediction/COTN/multi_dataset_report.csv"
        df_report.to_csv(report_path, index=False)
        print(f"\n详细报告已保存: {report_path}")
        
        # 可视化对比
        self.plot_comparison_results(df_report)
        
        return df_report
    
    def plot_comparison_results(self, df_report):
        """绘制对比结果图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. R²对比
        axes[0,0].bar(range(len(df_report)), df_report['R²'], color='skyblue', alpha=0.7)
        axes[0,0].set_title('各数据集R²对比')
        axes[0,0].set_ylabel('R²')
        axes[0,0].set_xticks(range(len(df_report)))
        axes[0,0].set_xticklabels([name.split('_')[1] for name in df_report['数据集']], rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. MAE对比
        axes[0,1].bar(range(len(df_report)), df_report['MAE(MW)'], color='lightcoral', alpha=0.7)
        axes[0,1].set_title('各数据集MAE对比')
        axes[0,1].set_ylabel('MAE (MW)')
        axes[0,1].set_xticks(range(len(df_report)))
        axes[0,1].set_xticklabels([name.split('_')[1] for name in df_report['数据集']], rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. 装机容量vs性能
        axes[1,0].scatter(df_report['装机容量(MW)'], df_report['R²'], s=100, alpha=0.7, color='green')
        axes[1,0].set_xlabel('装机容量 (MW)')
        axes[1,0].set_ylabel('R²')
        axes[1,0].set_title('装机容量 vs 预测性能')
        axes[1,0].grid(True, alpha=0.3)
        
        # 添加电站标签
        for i, row in df_report.iterrows():
            axes[1,0].annotate(row['数据集'].split('_')[1], 
                             (row['装机容量(MW)'], row['R²']), 
                             xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 4. 数据量vs性能
        axes[1,1].scatter(df_report['数据点数'], df_report['R²'], s=100, alpha=0.7, color='orange')
        axes[1,1].set_xlabel('数据点数')
        axes[1,1].set_ylabel('R²')
        axes[1,1].set_title('数据量 vs 预测性能')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = "/Users/tangbao/Desktop/PV_prediction/COTN/multi_dataset_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"对比图表已保存: {plot_path}")
        
        plt.show()


class MockCOTNModelForDataset:
    """
    用于多数据集测试的COTN模型模拟器
    实际使用时替换为真正的COTN训练和预测逻辑
    """
    
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.trained = False
    
    def train(self):
        """训练模型"""
        print(f"  训练COTN模型 on {self.dataset_name}...")
        # 这里应该调用实际的COTN训练代码
        # 模拟训练结果
        self.trained = True
        return {'status': 'success', 'epochs': 6, 'final_loss': 0.123}
    
    def test(self):
        """测试模型"""
        print(f"  测试COTN模型 on {self.dataset_name}...")
        # 模拟测试结果
        return {'status': 'success', 'test_loss': 0.156}
    
    def predict_and_evaluate(self):
        """预测并评估"""
        print(f"  生成预测结果 for {self.dataset_name}...")
        
        # 模拟不同数据集的不同性能
        # 实际使用时替换为真正的预测和评估代码
        np.random.seed(hash(self.dataset_name) % 1000)  # 确保可重复的随机结果
        
        r2 = 0.85 + np.random.normal(0, 0.05)  # 模拟R²在0.8-0.9之间
        mae = 1.5 + np.random.normal(0, 0.3)   # 模拟MAE
        rmse = mae * 1.3 + np.random.normal(0, 0.1)  # 模拟RMSE
        
        return {
            'mae': max(0.5, mae),
            'rmse': max(0.7, rmse), 
            'r2': min(0.95, max(0.7, r2))
        }


def test_multi_dataset_framework():
    """测试多数据集框架"""
    print("测试多数据集框架...")
    
    # 创建测试器
    tester = MultiDatasetTester()
    
    # 在所有数据集上测试（使用模拟模型）
    tester.test_all_datasets(MockCOTNModelForDataset, test_ratio=0.2)
    
    return tester

if __name__ == "__main__":
    # 运行测试
    tester = test_multi_dataset_framework()
    print("多数据集测试框架完成！")