import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_pv_data():
    """加载光伏数据"""
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
    
    # 添加时间特征
    df['Hour'] = df['Time'].dt.hour
    df['Day'] = df['Time'].dt.day
    df['Month'] = df['Time'].dt.month
    df['DayOfYear'] = df['Time'].dt.dayofyear
    df['Season'] = df['Month'].apply(lambda x: (x%12 + 3)//3)
    
    return df

def create_overview_analysis(df):
    """创建总览分析图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('光伏电站数据总览分析', fontsize=16, fontweight='bold')
    
    # 1. 功率时间序列
    sample_df = df.sample(n=min(10000, len(df))).sort_values('Time')
    axes[0, 0].plot(sample_df['Time'], sample_df['Power'], alpha=0.6, linewidth=0.5)
    axes[0, 0].set_title('发电功率时间序列')
    axes[0, 0].set_ylabel('功率 (MW)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 日均发电模式
    hourly_avg = df.groupby('Hour')['Power'].mean()
    axes[0, 1].plot(hourly_avg.index, hourly_avg.values, 'o-', linewidth=2, markersize=6)
    axes[0, 1].set_title('日均发电模式')
    axes[0, 1].set_xlabel('小时')
    axes[0, 1].set_ylabel('平均功率 (MW)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 月度发电统计
    monthly_stats = df.groupby('Month').agg({'Power': ['mean', 'max']})
    months = monthly_stats.index
    axes[0, 2].bar(months, monthly_stats[('Power', 'mean')], alpha=0.7, label='平均功率')
    axes[0, 2].plot(months, monthly_stats[('Power', 'max')], 'ro-', label='最大功率', linewidth=2)
    axes[0, 2].set_title('月度发电统计')
    axes[0, 2].set_xlabel('月份')
    axes[0, 2].set_ylabel('功率 (MW)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 功率分布直方图
    axes[1, 0].hist(df['Power'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 0].set_title('发电功率分布')
    axes[1, 0].set_xlabel('功率 (MW)')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 辐照强度分布
    axes[1, 1].hist(df['Total_Irradiance'], bins=50, alpha=0.7, color='gold', edgecolor='black')
    axes[1, 1].set_title('总辐照强度分布')
    axes[1, 1].set_xlabel('辐照强度 (W/m²)')
    axes[1, 1].set_ylabel('频次')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 温度分布
    axes[1, 2].hist(df['Temperature'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1, 2].set_title('温度分布')
    axes[1, 2].set_xlabel('温度 (°C)')
    axes[1, 2].set_ylabel('频次')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis/overview_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_correlation_analysis(df):
    """创建相关性分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('光伏发电相关性分析', fontsize=16, fontweight='bold')
    
    # 1. 功率vs辐照强度散点图
    sample_idx = np.random.choice(len(df), 5000, replace=False)
    sample_df = df.iloc[sample_idx]
    
    scatter = axes[0, 0].scatter(sample_df['Total_Irradiance'], sample_df['Power'], 
                               c=sample_df['Temperature'], cmap='viridis', alpha=0.6, s=10)
    axes[0, 0].set_xlabel('总辐照强度 (W/m²)')
    axes[0, 0].set_ylabel('功率 (MW)')
    axes[0, 0].set_title('功率 vs 辐照强度 (彩色编码: 温度)')
    plt.colorbar(scatter, ax=axes[0, 0], label='温度 (°C)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 相关性热图
    corr_vars = ['Power', 'Total_Irradiance', 'Direct_Irradiance', 'Global_Irradiance', 'Temperature', 'Atmosphere']
    corr_matrix = df[corr_vars].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=axes[0, 1], fmt='.2f')
    axes[0, 1].set_title('变量相关性热图')
    
    # 3. 温度对发电效率的影响
    df_temp = df[df['Total_Irradiance'] > 100].copy()
    df_temp['Efficiency'] = df_temp['Power'] / df_temp['Total_Irradiance'] * 1000
    df_temp = df_temp[(df_temp['Efficiency'] > 0) & (df_temp['Efficiency'] < 1)]
    
    temp_bins = np.arange(-20, 45, 5)
    temp_centers = (temp_bins[:-1] + temp_bins[1:]) / 2
    efficiency_by_temp = []
    
    for i in range(len(temp_bins)-1):
        mask = (df_temp['Temperature'] >= temp_bins[i]) & (df_temp['Temperature'] < temp_bins[i+1])
        if mask.sum() > 0:
            efficiency_by_temp.append(df_temp.loc[mask, 'Efficiency'].median())
        else:
            efficiency_by_temp.append(np.nan)
    
    axes[1, 0].plot(temp_centers, efficiency_by_temp, 'ro-', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('温度 (°C)')
    axes[1, 0].set_ylabel('发电效率')
    axes[1, 0].set_title('温度对发电效率的影响')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 功率与多个变量的关系
    # 使用子采样数据创建3D效果的2D图
    sample_df_small = sample_df.sample(n=1000)
    bubble_sizes = (sample_df_small['Atmosphere'] - sample_df_small['Atmosphere'].min()) * 50 + 10
    
    scatter2 = axes[1, 1].scatter(sample_df_small['Total_Irradiance'], sample_df_small['Power'],
                                 s=bubble_sizes, c=sample_df_small['Temperature'], 
                                 cmap='coolwarm', alpha=0.6)
    axes[1, 1].set_xlabel('总辐照强度 (W/m²)')
    axes[1, 1].set_ylabel('功率 (MW)')
    axes[1, 1].set_title('功率关系图 (气泡大小: 大气压)')
    plt.colorbar(scatter2, ax=axes[1, 1], label='温度 (°C)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis/correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_seasonal_analysis(df):
    """创建季节性分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('光伏发电季节性分析', fontsize=16, fontweight='bold')
    
    # 1. 季节性功率箱线图
    season_labels = ['春季', '夏季', '秋季', '冬季']
    season_data = [df[df['Season'] == i]['Power'].values for i in range(1, 5)]
    bp1 = axes[0, 0].boxplot(season_data, labels=season_labels, patch_artist=True)
    colors = ['lightgreen', 'yellow', 'orange', 'lightblue']
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
    axes[0, 0].set_title('季节性发电功率分布')
    axes[0, 0].set_ylabel('功率 (MW)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 月度平均指标对比
    monthly_stats = df.groupby('Month').agg({
        'Power': 'mean',
        'Total_Irradiance': 'mean',
        'Temperature': 'mean'
    })
    
    x = monthly_stats.index
    ax2_1 = axes[0, 1]
    ax2_2 = ax2_1.twinx()
    
    line1 = ax2_1.plot(x, monthly_stats['Power'], 'b-o', label='平均功率')
    line2 = ax2_2.plot(x, monthly_stats['Total_Irradiance'], 'r-s', label='平均辐照')
    
    ax2_1.set_xlabel('月份')
    ax2_1.set_ylabel('功率 (MW)', color='b')
    ax2_2.set_ylabel('辐照强度 (W/m²)', color='r')
    ax2_1.set_title('月度功率与辐照趋势')
    ax2_1.grid(True, alpha=0.3)
    
    # 合并图例
    lines1, labels1 = ax2_1.get_legend_handles_labels()
    lines2, labels2 = ax2_2.get_legend_handles_labels()
    ax2_1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 3. 日内发电模式对比
    for season in range(1, 5):
        season_data = df[df['Season'] == season]
        hourly_power = season_data.groupby('Hour')['Power'].mean()
        axes[1, 0].plot(hourly_power.index, hourly_power.values, 
                       label=season_labels[season-1], linewidth=2, marker='o', markersize=4)
    
    axes[1, 0].set_xlabel('小时')
    axes[1, 0].set_ylabel('平均功率 (MW)')
    axes[1, 0].set_title('不同季节的日内发电模式')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 容量因子月度变化
    monthly_cf = df.groupby('Month')['Power'].mean() / 50 * 100  # 转换为百分比
    axes[1, 1].bar(monthly_cf.index, monthly_cf.values, 
                  color=['skyblue' if cf < 20 else 'lightgreen' for cf in monthly_cf.values])
    axes[1, 1].set_xlabel('月份')
    axes[1, 1].set_ylabel('容量因子 (%)')
    axes[1, 1].set_title('月度容量因子变化')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 添加平均线
    avg_cf = monthly_cf.mean()
    axes[1, 1].axhline(y=avg_cf, color='red', linestyle='--', label=f'年平均: {avg_cf:.1f}%')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('analysis/seasonal_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_performance_analysis(df):
    """创建性能分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('光伏电站性能分析', fontsize=16, fontweight='bold')
    
    # 1. 发电效率分析
    df_eff = df[df['Total_Irradiance'] > 100].copy()
    df_eff['Efficiency'] = df_eff['Power'] / df_eff['Total_Irradiance'] * 1000
    df_eff = df_eff[(df_eff['Efficiency'] > 0) & (df_eff['Efficiency'] < 1)]
    
    axes[0, 0].hist(df_eff['Efficiency'], bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[0, 0].set_xlabel('发电效率')
    axes[0, 0].set_ylabel('频次')
    axes[0, 0].set_title('发电效率分布')
    axes[0, 0].axvline(df_eff['Efficiency'].mean(), color='red', linestyle='--', 
                      label=f'平均值: {df_eff["Efficiency"].mean():.3f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 容量因子时间变化
    df['Date'] = df['Time'].dt.date
    daily_cf = df.groupby('Date')['Power'].mean() / 50 * 100
    
    # 取样显示（每10天一个点）
    sample_dates = daily_cf.iloc[::10]
    axes[0, 1].plot(range(len(sample_dates)), sample_dates.values, 'b-', alpha=0.7)
    axes[0, 1].set_xlabel('时间 (采样点)')
    axes[0, 1].set_ylabel('日平均容量因子 (%)')
    axes[0, 1].set_title('容量因子时间变化趋势')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 不同辐照强度下的发电表现
    irradiance_bins = [0, 200, 400, 600, 800, 1000, 1400]
    irradiance_labels = ['0-200', '200-400', '400-600', '600-800', '800-1000', '1000+']
    
    binned_power = []
    for i in range(len(irradiance_bins)-1):
        mask = (df['Total_Irradiance'] >= irradiance_bins[i]) & (df['Total_Irradiance'] < irradiance_bins[i+1])
        if mask.sum() > 0:
            binned_power.append(df.loc[mask, 'Power'].values)
        else:
            binned_power.append([])
    
    # 过滤空数据
    filtered_data = [data for data in binned_power if len(data) > 0]
    filtered_labels = [irradiance_labels[i] for i, data in enumerate(binned_power) if len(data) > 0]
    
    bp3 = axes[1, 0].boxplot(filtered_data, labels=filtered_labels, patch_artist=True)
    for patch in bp3['boxes']:
        patch.set_facecolor('lightblue')
    axes[1, 0].set_xlabel('辐照强度区间 (W/m²)')
    axes[1, 0].set_ylabel('功率 (MW)')
    axes[1, 0].set_title('不同辐照强度下的发电表现')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 关键性能指标总结
    axes[1, 1].axis('off')
    
    # 计算关键指标
    total_energy = df['Power'].sum() * 0.25 / 1000  # GWh
    avg_power = df['Power'].mean()
    max_power = df['Power'].max()
    capacity_factor = (avg_power / 50) * 100
    avg_irradiance = df['Total_Irradiance'].mean()
    avg_temp = df['Temperature'].mean()
    
    # 显示指标
    performance_text = f"""关键性能指标:

装机容量: 50 MW
平均功率: {avg_power:.2f} MW
最大功率: {max_power:.2f} MW
容量因子: {capacity_factor:.1f}%
总发电量: {total_energy:.1f} GWh

环境条件:
平均辐照: {avg_irradiance:.1f} W/m²
平均温度: {avg_temp:.1f} °C

数据期间: {df['Time'].min().strftime('%Y-%m-%d')} 
至 {df['Time'].max().strftime('%Y-%m-%d')}
数据点数: {len(df):,} 个"""
    
    axes[1, 1].text(0.1, 0.9, performance_text, transform=axes[1, 1].transAxes, 
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('analysis/performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_analysis_report():
    """生成完整的分析报告"""
    print("=== 光伏电站数据可视化分析报告 ===\n")
    
    # 加载数据
    print("正在加载数据...")
    df = load_pv_data()
    
    print(f"数据加载完成，共 {len(df):,} 条记录")
    print(f"时间范围: {df['Time'].min()} 至 {df['Time'].max()}")
    
    # 创建各种分析图表
    print("\n正在生成分析图表...")
    
    print("1. 创建总览分析图...")
    create_overview_analysis(df)
    
    print("2. 创建相关性分析图...")
    create_correlation_analysis(df)
    
    print("3. 创建季节性分析图...")
    create_seasonal_analysis(df)
    
    print("4. 创建性能分析图...")
    create_performance_analysis(df)
    
    print("\n=== 分析完成 ===")
    print("所有图表已保存到 analysis/ 文件夹")
    print("\n生成的文件:")
    print("- overview_analysis.png: 数据总览分析")
    print("- correlation_analysis.png: 相关性分析")
    print("- seasonal_analysis.png: 季节性分析")
    print("- performance_analysis.png: 性能分析")

if __name__ == "__main__":
    generate_analysis_report()