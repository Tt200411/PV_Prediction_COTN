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

def load_and_explore_data():
    """加载并探索Excel数据"""
    file_path = "/Users/tangbao/Desktop/PV_prediction/Solar Station/Solar station site 1 (Nominal capacity-50MW).xlsx"
    
    try:
        # 读取Excel文件，获取所有sheet名称
        xls = pd.ExcelFile(file_path)
        print(f"文件包含的工作表: {xls.sheet_names}")
        
        # 读取每个工作表的数据
        all_data = {}
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            all_data[sheet_name] = df
            print(f"\n=== 工作表: {sheet_name} ===")
            print(f"数据形状: {df.shape}")
            print(f"列名: {list(df.columns)}")
            print(f"前5行数据:")
            print(df.head())
            print(f"数据类型:")
            print(df.dtypes)
            print(f"缺失值统计:")
            print(df.isnull().sum())
            print("-" * 50)
        
        return all_data
    
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

def create_visualizations(data_dict):
    """创建数据可视化"""
    if not data_dict:
        print("没有数据可供可视化")
        return
    
    # 设置图形样式
    plt.style.use('seaborn-v0_8')
    
    for sheet_name, df in data_dict.items():
        print(f"\n正在为工作表 '{sheet_name}' 创建可视化...")
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'光伏电站数据分析 - {sheet_name}', fontsize=16, fontweight='bold')
        
        # 检查是否有时间列
        time_columns = []
        numeric_columns = []
        
        for col in df.columns:
            if df[col].dtype in ['datetime64[ns]', 'object']:
                # 尝试解析为时间
                try:
                    pd.to_datetime(df[col])
                    time_columns.append(col)
                except:
                    pass
            elif df[col].dtype in ['int64', 'float64']:
                numeric_columns.append(col)
        
        print(f"时间列: {time_columns}")
        print(f"数值列: {numeric_columns}")
        
        # 图1: 数据概览
        if numeric_columns:
            # 选择前几个数值列进行统计
            stats_data = df[numeric_columns[:5]].describe()
            sns.heatmap(stats_data, annot=True, fmt='.2f', ax=axes[0,0], cmap='YlOrRd')
            axes[0,0].set_title('数值列统计信息', fontweight='bold')
        
        # 图2: 时间序列图
        if time_columns and numeric_columns:
            time_col = time_columns[0]
            value_col = numeric_columns[0]
            
            # 转换时间列
            df_copy = df.copy()
            df_copy[time_col] = pd.to_datetime(df_copy[time_col])
            df_copy = df_copy.sort_values(time_col)
            
            axes[0,1].plot(df_copy[time_col], df_copy[value_col], 'b-', linewidth=1)
            axes[0,1].set_title(f'{value_col} 时间序列', fontweight='bold')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # 图3: 数值分布
        if numeric_columns:
            main_col = numeric_columns[0]
            axes[1,0].hist(df[main_col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1,0].set_title(f'{main_col} 分布直方图', fontweight='bold')
            axes[1,0].set_xlabel(main_col)
            axes[1,0].set_ylabel('频率')
        
        # 图4: 相关性热图
        if len(numeric_columns) >= 2:
            correlation_matrix = df[numeric_columns].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
            axes[1,1].set_title('数值列相关性热图', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'/Users/tangbao/Desktop/PV_prediction/pv_analysis_{sheet_name.replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

def analyze_pv_performance(data_dict):
    """分析光伏性能数据"""
    print("\n=== 光伏性能分析 ===")
    
    for sheet_name, df in data_dict.items():
        print(f"\n--- 工作表: {sheet_name} ---")
        
        # 查找可能的发电量、辐照度、温度等列
        power_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['power', 'generation', '发电', '功率'])]
        irradiance_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['irradiance', 'radiation', '辐照', '太阳'])]
        temp_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['temp', '温度'])]
        
        print(f"发电相关列: {power_cols}")
        print(f"辐照相关列: {irradiance_cols}")
        print(f"温度相关列: {temp_cols}")
        
        # 基本统计分析
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\n数值列基本统计:")
            print(df[numeric_cols].describe())
            
            # 计算一些关键指标
            for col in numeric_cols:
                values = df[col].dropna()
                if len(values) > 0:
                    print(f"\n{col}:")
                    print(f"  平均值: {values.mean():.2f}")
                    print(f"  最大值: {values.max():.2f}")
                    print(f"  最小值: {values.min():.2f}")
                    print(f"  标准差: {values.std():.2f}")

def main():
    """主函数"""
    print("开始分析光伏电站数据...")
    
    # 加载数据
    data = load_and_explore_data()
    
    if data:
        # 创建可视化
        create_visualizations(data)
        
        # 性能分析
        analyze_pv_performance(data)
        
        print("\n分析完成！图表已保存到项目目录。")
    else:
        print("数据加载失败，请检查文件路径和格式。")

if __name__ == "__main__":
    main()