import pandas as pd
import numpy as np
import os

def prepare_pv_data_for_cotn():
    """
    将光伏Excel数据转换为COTN兼容的CSV格式
    """
    print("开始准备光伏数据...")
    
    # 读取原始Excel数据
    file_path = "/Users/tangbao/Desktop/PV_prediction/Solar Station/Solar station site 1 (Nominal capacity-50MW).xlsx"
    df = pd.read_excel(file_path, sheet_name='sheet1')
    
    print(f"原始数据形状: {df.shape}")
    print(f"原始列名: {list(df.columns)}")
    
    # 数据预处理
    df['Time'] = pd.to_datetime(df['Time(year-month-day h:m:s)'])
    
    # 重命名列名，保持简洁
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
    
    # 创建COTN需要的格式：[date, features..., target]
    # 选择特征：5个气象特征 + 1个目标功率
    cotn_df = pd.DataFrame()
    
    # 时间列（COTN要求的格式）
    cotn_df['date'] = df['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # 特征列（按重要性排序）
    cotn_df['Total_Irradiance'] = df['Total_Irradiance']
    cotn_df['Global_Irradiance'] = df['Global_Irradiance']
    cotn_df['Direct_Irradiance'] = df['Direct_Irradiance']
    cotn_df['Temperature'] = df['Temperature']
    cotn_df['Atmosphere'] = df['Atmosphere']
    
    # 目标列（必须放在最后）
    cotn_df['Power'] = df['Power']
    
    # 检查数据质量
    print(f"\n数据质量检查:")
    print(f"转换后数据形状: {cotn_df.shape}")
    print(f"缺失值统计:\n{cotn_df.isnull().sum()}")
    print(f"数据范围:")
    print(cotn_df.describe())
    
    # 处理异常值（如果有）
    # 确保功率值不为负
    cotn_df['Power'] = cotn_df['Power'].clip(lower=0)
    
    # 确保辐照值不为负
    irradiance_cols = ['Total_Irradiance', 'Global_Irradiance', 'Direct_Irradiance']
    for col in irradiance_cols:
        cotn_df[col] = cotn_df[col].clip(lower=0)
    
    # 检查时间连续性
    time_diff = pd.to_datetime(cotn_df['date']).diff().dropna()
    expected_interval = pd.Timedelta(minutes=15)
    irregular_intervals = time_diff[time_diff != expected_interval]
    
    if len(irregular_intervals) > 0:
        print(f"\n警告: 发现 {len(irregular_intervals)} 个不规则时间间隔")
        print("前5个不规则间隔:")
        print(irregular_intervals.head())
    else:
        print(f"\n✓ 时间序列连续性检查通过，所有间隔均为15分钟")
    
    # 保存为CSV文件
    output_path = "/Users/tangbao/Desktop/PV_prediction/COTN/ETT-small/PV_Solar_Station_1.csv"
    cotn_df.to_csv(output_path, index=False)
    
    print(f"\n✓ 数据已保存到: {output_path}")
    print(f"最终数据形状: {cotn_df.shape}")
    print(f"时间范围: {cotn_df['date'].iloc[0]} 到 {cotn_df['date'].iloc[-1]}")
    
    # 显示前几行和后几行作为验证
    print(f"\n前5行数据:")
    print(cotn_df.head())
    print(f"\n后5行数据:")
    print(cotn_df.tail())
    
    return cotn_df

if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs("/Users/tangbao/Desktop/PV_prediction/COTN/ETT-small", exist_ok=True)
    
    # 执行数据准备
    pv_data = prepare_pv_data_for_cotn()
    
    print("\n" + "="*60)
    print("光伏数据预处理完成！")
    print("现在可以使用COTN模型进行训练。")
    print("="*60)