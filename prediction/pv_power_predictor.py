import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PVPowerPredictor:
    """光伏发电功率预测器"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        
    def load_data(self, file_path):
        """加载光伏数据"""
        df = pd.read_excel(file_path, sheet_name='sheet1')
        
        # 数据预处理
        df['Time'] = pd.to_datetime(df['Time(year-month-day h:m:s)'])
        
        # 重命名列名
        df = df.rename(columns={
            'Time(year-month-day h:m:s)': 'Time_Original',
            'Total solar irradiance (W/m2)': 'Total_Irradiance',
            'Direct normal irradiance (W/m2)': 'Direct_Irradiance', 
            'Global horizontal irradiance (W/m2)': 'Global_Irradiance',
            'Air temperature  (°C) ': 'Temperature',
            'Atmosphere (hpa)': 'Atmosphere',
            'Power (MW)': 'Power'
        })
        
        # 添加时间特征
        df['Hour'] = df['Time'].dt.hour
        df['Day'] = df['Time'].dt.day
        df['Month'] = df['Time'].dt.month
        df['DayOfYear'] = df['Time'].dt.dayofyear
        df['Season'] = df['Month'].apply(lambda x: (x%12 + 3)//3)
        df['IsWeekend'] = df['Time'].dt.weekday >= 5
        
        # 添加辐照相关特征
        df['Irradiance_Ratio'] = df['Direct_Irradiance'] / (df['Total_Irradiance'] + 1e-6)
        df['Irradiance_Sum'] = df['Total_Irradiance'] + df['Global_Irradiance']
        
        return df
    
    def prepare_features(self, df):
        """准备训练特征"""
        feature_cols = [
            'Total_Irradiance', 'Direct_Irradiance', 'Global_Irradiance', 
            'Temperature', 'Atmosphere', 'Hour', 'Month', 'DayOfYear', 
            'Season', 'IsWeekend', 'Irradiance_Ratio', 'Irradiance_Sum'
        ]
        
        X = df[feature_cols]
        y = df['Power']
        
        self.feature_names = feature_cols
        
        return X, y
    
    def train_model(self, X, y, model_type='random_forest'):
        """训练预测模型"""
        print(f"开始训练{model_type}模型...")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        # 特征标准化
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 选择模型
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=200, 
                max_depth=20, 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42, 
                n_jobs=-1
            )
        elif model_type == 'linear':
            self.model = LinearRegression()
        
        # 训练模型
        self.model.fit(X_train_scaled, y_train)
        
        # 预测
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # 评估模型
        train_metrics = self._calculate_metrics(y_train, y_pred_train)
        test_metrics = self._calculate_metrics(y_test, y_pred_test)
        
        # 交叉验证
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='r2')
        
        self.is_trained = True
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_scores': cv_scores,
            'y_test': y_test,
            'y_pred_test': y_pred_test,
            'X_test': X_test
        }
    
    def _calculate_metrics(self, y_true, y_pred):
        """计算评估指标"""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
        }
    
    def predict(self, X):
        """进行预测"""
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train_model方法")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def get_feature_importance(self):
        """获取特征重要性"""
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath):
        """保存模型"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """加载模型"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_trained = True
        print(f"模型已从 {filepath} 加载")

def visualize_predictions(results, save_path='prediction_results.png'):
    """可视化预测结果"""
    y_test = results['y_test']
    y_pred_test = results['y_pred_test']
    test_metrics = results['test_metrics']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('光伏发电功率预测模型结果', fontsize=16, fontweight='bold')
    
    # 1. 预测 vs 实际值散点图
    axes[0, 0].scatter(y_test, y_pred_test, alpha=0.6, s=10, color='blue')
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('实际功率 (MW)')
    axes[0, 0].set_ylabel('预测功率 (MW)')
    axes[0, 0].set_title(f'预测 vs 实际值\nR² = {test_metrics["r2"]:.3f}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 预测误差分布
    residuals = y_test - y_pred_test
    axes[0, 1].hist(residuals, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].set_xlabel('预测误差 (MW)')
    axes[0, 1].set_ylabel('频次')
    axes[0, 1].set_title(f'预测误差分布\nMAE = {test_metrics["mae"]:.3f} MW')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 时间序列预测结果 (部分数据)
    sample_size = min(1000, len(y_test))
    sample_idx = np.random.choice(len(y_test), sample_size, replace=False)
    sample_idx_sorted = np.sort(sample_idx)
    
    axes[1, 0].plot(range(sample_size), y_test.iloc[sample_idx_sorted], 'b-', 
                   label='实际值', alpha=0.8, linewidth=1)
    axes[1, 0].plot(range(sample_size), y_pred_test[sample_idx_sorted], 'r-', 
                   label='预测值', alpha=0.8, linewidth=1)
    axes[1, 0].set_xlabel('时间点')
    axes[1, 0].set_ylabel('功率 (MW)')
    axes[1, 0].set_title('时间序列预测对比 (随机采样)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 误差统计箱线图
    error_percentiles = np.percentile(np.abs(residuals), [25, 50, 75, 90, 95])
    axes[1, 1].boxplot([np.abs(residuals)], labels=['预测误差'])
    axes[1, 1].set_ylabel('绝对误差 (MW)')
    axes[1, 1].set_title('预测误差统计分布')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 添加统计信息
    textstr = f'''模型性能指标:
RMSE: {test_metrics["rmse"]:.3f} MW
MAE: {test_metrics["mae"]:.3f} MW
R²: {test_metrics["r2"]:.3f}
误差中位数: {np.median(np.abs(residuals)):.3f} MW'''
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    fig.text(0.02, 0.02, textstr, fontsize=10, verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_feature_importance_plot(predictor, save_path='feature_importance.png'):
    """创建特征重要性图"""
    importance_df = predictor.get_feature_importance()
    
    if importance_df is None:
        print("无法获取特征重要性信息")
        return
    
    plt.figure(figsize=(10, 8))
    bars = plt.barh(importance_df['feature'], importance_df['importance'])
    
    # 设置颜色
    colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.xlabel('重要性')
    plt.title('特征重要性排序', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, v in enumerate(importance_df['importance']):
        plt.text(v + 0.001, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数 - 完整的预测模型流程"""
    print("=== 光伏发电功率预测模型 ===\n")
    
    # 初始化预测器
    predictor = PVPowerPredictor()
    
    # 加载数据
    print("1. 加载数据...")
    file_path = "/Users/tangbao/Desktop/PV_prediction/Solar Station/Solar station site 1 (Nominal capacity-50MW).xlsx"
    df = predictor.load_data(file_path)
    print(f"   数据形状: {df.shape}")
    
    # 准备特征
    print("\n2. 准备特征...")
    X, y = predictor.prepare_features(df)
    print(f"   特征数量: {X.shape[1]}")
    print(f"   特征列表: {list(X.columns)}")
    
    # 训练模型
    print("\n3. 训练随机森林模型...")
    results = predictor.train_model(X, y, model_type='random_forest')
    
    # 输出结果
    print("\n4. 模型性能评估:")
    test_metrics = results['test_metrics']
    cv_scores = results['cv_scores']
    
    print(f"   测试集性能:")
    print(f"     - RMSE: {test_metrics['rmse']:.3f} MW")
    print(f"     - MAE: {test_metrics['mae']:.3f} MW")
    print(f"     - R²: {test_metrics['r2']:.3f}")
    print(f"   交叉验证R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # 特征重要性
    print("\n5. 特征重要性:")
    importance_df = predictor.get_feature_importance()
    for _, row in importance_df.head(8).iterrows():
        print(f"     {row['feature']}: {row['importance']:.3f}")
    
    # 保存模型
    print("\n6. 保存模型...")
    predictor.save_model('pv_power_model.joblib')
    
    # 创建可视化
    print("\n7. 生成预测结果可视化...")
    visualize_predictions(results)
    create_feature_importance_plot(predictor)
    
    print("\n=== 预测模型训练完成 ===")
    print("模型文件和图表已保存到 prediction/ 文件夹")
    
    return predictor

if __name__ == "__main__":
    predictor = main()