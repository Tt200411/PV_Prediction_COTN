import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_prepare_data():
    """加载并准备时间序列数据"""
    file_path = "/Users/tangbao/Desktop/PV_prediction/Solar Station/Solar station site 1 (Nominal capacity-50MW).xlsx"
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
    
    # 按时间排序
    df = df.sort_values('Time').reset_index(drop=True)
    
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

def time_series_evaluation(df):
    """进行时间序列评估，用前面的数据预测后面的数据"""
    print("=== 时间序列预测评估 ===\n")
    
    # 特征列
    feature_cols = [
        'Total_Irradiance', 'Direct_Irradiance', 'Global_Irradiance', 
        'Temperature', 'Atmosphere', 'Hour', 'Month', 'DayOfYear', 
        'Season', 'IsWeekend', 'Irradiance_Ratio', 'Irradiance_Sum'
    ]
    
    # 按时间序列分割：前80%用于训练，后20%用于测试
    split_point = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_point].copy()
    test_df = df.iloc[split_point:].copy()
    
    print(f"训练集时间范围: {train_df['Time'].min()} 到 {train_df['Time'].max()}")
    print(f"测试集时间范围: {test_df['Time'].min()} 到 {test_df['Time'].max()}")
    print(f"训练集大小: {len(train_df):,} 个点")
    print(f"测试集大小: {len(test_df):,} 个点")
    
    # 准备特征和目标
    X_train = train_df[feature_cols]
    y_train = train_df['Power']
    X_test = test_df[feature_cols]
    y_test = test_df['Power']
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练模型
    print("\n正在训练随机森林模型...")
    model = RandomForestRegressor(
        n_estimators=200, 
        max_depth=20, 
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # 预测
    y_pred = model.predict(X_test_scaled)
    
    # 计算评估指标
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n时间序列预测性能:")
    print(f"  MAE: {mae:.3f} MW")
    print(f"  RMSE: {rmse:.3f} MW")
    print(f"  R²: {r2:.3f}")
    print(f"  准确度: {r2*100:.1f}%")
    
    return test_df, y_pred, model, scaler

def plot_time_series_comparison(test_df, y_pred, n_points=200):
    """绘制最后n个时间点的预测vs实际对比"""
    
    # 取最后n个点
    if len(test_df) > n_points:
        plot_df = test_df.iloc[-n_points:].copy()
        plot_pred = y_pred[-n_points:]
    else:
        plot_df = test_df.copy()
        plot_pred = y_pred
    
    # 创建时间索引
    time_points = plot_df['Time'].values
    actual_power = plot_df['Power'].values
    
    # 创建图形
    plt.figure(figsize=(15, 8))
    
    # 绘制实际值和预测值
    plt.plot(time_points, actual_power, 'b-', linewidth=1.5, label='实际发电功率', alpha=0.8)
    plt.plot(time_points, plot_pred, 'r-', linewidth=1.5, label='预测发电功率', alpha=0.8)
    
    # 设置图形属性
    plt.title(f'时间序列预测对比 - 最后 {len(plot_df)} 个时间点', fontsize=16, fontweight='bold')
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('发电功率 (MW)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 旋转x轴标签以避免重叠
    plt.xticks(rotation=45)
    
    # 计算该段的评估指标
    mae_segment = mean_absolute_error(actual_power, plot_pred)
    rmse_segment = np.sqrt(mean_squared_error(actual_power, plot_pred))
    r2_segment = r2_score(actual_power, plot_pred)
    
    # 添加性能指标文本
    textstr = f'''最后{len(plot_df)}个点的预测性能:
MAE: {mae_segment:.3f} MW
RMSE: {rmse_segment:.3f} MW
R²: {r2_segment:.3f}
时间范围: {pd.to_datetime(time_points[0]).strftime("%Y-%m-%d %H:%M")} 
        到 {pd.to_datetime(time_points[-1]).strftime("%Y-%m-%d %H:%M")}'''
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('time_series_comparison_200pts.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 创建误差分析图
    plt.figure(figsize=(15, 5))
    
    # 子图1: 误差时间序列
    plt.subplot(1, 3, 1)
    errors = actual_power - plot_pred
    plt.plot(time_points, errors, 'g-', linewidth=1, alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('预测误差时间序列')
    plt.xlabel('时间')
    plt.ylabel('误差 (MW)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 子图2: 误差分布直方图
    plt.subplot(1, 3, 2)
    plt.hist(errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.title('误差分布')
    plt.xlabel('误差 (MW)')
    plt.ylabel('频次')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3)
    
    # 子图3: 散点图
    plt.subplot(1, 3, 3)
    plt.scatter(actual_power, plot_pred, alpha=0.6, s=10)
    min_val = min(actual_power.min(), plot_pred.min())
    max_val = max(actual_power.max(), plot_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('实际功率 (MW)')
    plt.ylabel('预测功率 (MW)')
    plt.title('预测 vs 实际散点图')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('time_series_error_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_prediction_patterns(test_df, y_pred):
    """分析预测模式"""
    print("\n=== 预测模式分析 ===")
    
    # 计算各个时段的预测性能
    test_df_copy = test_df.copy()
    test_df_copy['Predicted_Power'] = y_pred
    test_df_copy['Error'] = test_df_copy['Power'] - test_df_copy['Predicted_Power']
    test_df_copy['Abs_Error'] = np.abs(test_df_copy['Error'])
    
    # 按小时分析
    hourly_performance = test_df_copy.groupby('Hour').agg({
        'Error': ['mean', 'std'],
        'Abs_Error': 'mean',
        'Power': 'mean',
        'Predicted_Power': 'mean'
    }).round(3)
    
    print("\n按小时的预测性能 (前10小时):")
    print("小时  平均误差  误差标准差  平均绝对误差  实际均值  预测均值")
    print("-" * 60)
    for hour in range(10):
        if hour in hourly_performance.index:
            mean_err = hourly_performance.loc[hour, ('Error', 'mean')]
            std_err = hourly_performance.loc[hour, ('Error', 'std')]
            mae = hourly_performance.loc[hour, ('Abs_Error', 'mean')]
            actual_mean = hourly_performance.loc[hour, ('Power', 'mean')]
            pred_mean = hourly_performance.loc[hour, ('Predicted_Power', 'mean')]
            print(f"{hour:2d}    {mean_err:7.3f}    {std_err:8.3f}     {mae:10.3f}   {actual_mean:7.3f}   {pred_mean:7.3f}")
    
    # 分析高低功率时段的预测性能
    print(f"\n功率区间预测性能:")
    power_bins = [0, 5, 15, 25, 35, 50]
    power_labels = ['0-5MW', '5-15MW', '15-25MW', '25-35MW', '35-50MW']
    
    for i in range(len(power_bins)-1):
        mask = (test_df_copy['Power'] >= power_bins[i]) & (test_df_copy['Power'] < power_bins[i+1])
        if mask.sum() > 0:
            segment_mae = test_df_copy.loc[mask, 'Abs_Error'].mean()
            segment_count = mask.sum()
            print(f"  {power_labels[i]}: MAE={segment_mae:.3f} MW, 样本数={segment_count}")

def main():
    """主函数"""
    print("开始时间序列预测分析...")
    
    # 加载数据
    df = load_and_prepare_data()
    print(f"数据加载完成，总共 {len(df):,} 个时间点")
    
    # 进行时间序列评估
    test_df, y_pred, model, scaler = time_series_evaluation(df)
    
    # 绘制时间序列对比图
    print(f"\n正在绘制最后200个时间点的预测对比图...")
    plot_time_series_comparison(test_df, y_pred, n_points=200)
    
    # 分析预测模式
    analyze_prediction_patterns(test_df, y_pred)
    
    print("\n=== 分析完成 ===")
    print("图表已保存到 prediction/ 文件夹:")
    print("- time_series_comparison_200pts.png: 最后200点预测对比")
    print("- time_series_error_analysis.png: 误差分析图")

if __name__ == "__main__":
    main()