import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PVPowerPredictionInterface:
    """光伏发电功率预测接口"""
    
    def __init__(self, model_path='pv_power_model.joblib'):
        """初始化预测接口"""
        self.model_data = None
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        try:
            self.load_model(model_path)
            print("模型加载成功！")
        except FileNotFoundError:
            print(f"警告: 未找到模型文件 {model_path}")
            print("请先运行 pv_power_predictor.py 训练模型")
    
    def load_model(self, model_path):
        """加载训练好的模型"""
        self.model_data = joblib.load(model_path)
        self.model = self.model_data['model']
        self.scaler = self.model_data['scaler']
        self.feature_names = self.model_data['feature_names']
    
    def predict_single(self, total_irradiance, direct_irradiance, global_irradiance,
                      temperature, atmosphere, hour, month, day_of_year):
        """单点预测
        
        参数:
        - total_irradiance: 总辐照强度 (W/m²)
        - direct_irradiance: 直接辐照强度 (W/m²)
        - global_irradiance: 全球水平辐照强度 (W/m²)
        - temperature: 空气温度 (°C)
        - atmosphere: 大气压 (hpa)
        - hour: 小时 (0-23)
        - month: 月份 (1-12)
        - day_of_year: 年内第几天 (1-365)
        """
        if self.model is None:
            raise ValueError("模型未加载，请先加载模型")
        
        # 计算衍生特征
        season = (month % 12 + 3) // 3
        is_weekend = 0  # 默认工作日
        irradiance_ratio = direct_irradiance / (total_irradiance + 1e-6)
        irradiance_sum = total_irradiance + global_irradiance
        
        # 构建特征向量
        features = pd.DataFrame({
            'Total_Irradiance': [total_irradiance],
            'Direct_Irradiance': [direct_irradiance],
            'Global_Irradiance': [global_irradiance],
            'Temperature': [temperature],
            'Atmosphere': [atmosphere],
            'Hour': [hour],
            'Month': [month],
            'DayOfYear': [day_of_year],
            'Season': [season],
            'IsWeekend': [is_weekend],
            'Irradiance_Ratio': [irradiance_ratio],
            'Irradiance_Sum': [irradiance_sum]
        })
        
        # 确保特征顺序正确
        features = features[self.feature_names]
        
        # 标准化特征
        features_scaled = self.scaler.transform(features)
        
        # 预测
        prediction = self.model.predict(features_scaled)[0]
        
        return max(0, prediction)  # 确保预测值非负
    
    def predict_batch(self, input_data):
        """批量预测
        
        参数:
        - input_data: pandas DataFrame，包含所有必需的特征列
        """
        if self.model is None:
            raise ValueError("模型未加载，请先加载模型")
        
        # 复制数据避免修改原始数据
        df = input_data.copy()
        
        # 计算衍生特征（如果不存在）
        if 'Season' not in df.columns:
            df['Season'] = df['Month'].apply(lambda x: (x % 12 + 3) // 3)
        
        if 'IsWeekend' not in df.columns:
            df['IsWeekend'] = 0  # 默认工作日
        
        if 'Irradiance_Ratio' not in df.columns:
            df['Irradiance_Ratio'] = df['Direct_Irradiance'] / (df['Total_Irradiance'] + 1e-6)
        
        if 'Irradiance_Sum' not in df.columns:
            df['Irradiance_Sum'] = df['Total_Irradiance'] + df['Global_Irradiance']
        
        # 选择和排序特征
        features = df[self.feature_names]
        
        # 标准化特征
        features_scaled = self.scaler.transform(features)
        
        # 预测
        predictions = self.model.predict(features_scaled)
        
        # 确保预测值非负
        predictions = np.maximum(0, predictions)
        
        return predictions
    
    def predict_daily_curve(self, date_str, avg_irradiance, avg_temperature, atmosphere=915):
        """预测一天的发电曲线
        
        参数:
        - date_str: 日期字符串，格式: 'YYYY-MM-DD'
        - avg_irradiance: 当日平均辐照强度 (W/m²)
        - avg_temperature: 当日平均温度 (°C)
        - atmosphere: 大气压 (hpa)，默认915
        """
        date = datetime.strptime(date_str, '%Y-%m-%d')
        month = date.month
        day_of_year = date.timetuple().tm_yday
        
        # 生成24小时的发电预测
        hours = range(24)
        predictions = []
        
        for hour in hours:
            # 根据小时调整辐照强度（简化的日照模型）
            if 6 <= hour <= 18:  # 日间
                # 使用正弦函数模拟日照变化
                hour_factor = np.sin((hour - 6) * np.pi / 12)
                hourly_irradiance = avg_irradiance * hour_factor
                hourly_direct = hourly_irradiance * 0.7
                hourly_global = hourly_irradiance * 0.3
            else:  # 夜间
                hourly_irradiance = 0
                hourly_direct = 0
                hourly_global = 0
            
            # 预测该小时的发电量
            power = self.predict_single(
                total_irradiance=hourly_irradiance,
                direct_irradiance=hourly_direct,
                global_irradiance=hourly_global,
                temperature=avg_temperature,
                atmosphere=atmosphere,
                hour=hour,
                month=month,
                day_of_year=day_of_year
            )
            
            predictions.append({
                'hour': hour,
                'irradiance': hourly_irradiance,
                'predicted_power': power
            })
        
        return pd.DataFrame(predictions)

def demo_predictions():
    """演示预测功能"""
    print("=== 光伏发电功率预测演示 ===\n")
    
    # 初始化预测接口
    predictor = PVPowerPredictionInterface()
    
    if predictor.model is None:
        print("无法加载模型，演示结束")
        return
    
    # 演示1: 单点预测
    print("1. 单点预测演示:")
    power = predictor.predict_single(
        total_irradiance=800,    # 总辐照强度
        direct_irradiance=600,   # 直接辐照强度
        global_irradiance=200,   # 全球水平辐照强度
        temperature=25,          # 温度
        atmosphere=915,          # 大气压
        hour=12,                 # 中午12点
        month=6,                 # 6月
        day_of_year=150          # 第150天
    )
    print(f"   预测发电功率: {power:.2f} MW")
    
    # 演示2: 日发电曲线预测
    print("\n2. 日发电曲线预测:")
    daily_curve = predictor.predict_daily_curve(
        date_str='2024-06-15',
        avg_irradiance=600,
        avg_temperature=28
    )
    
    print("   小时   辐照强度   预测功率")
    print("   ----   --------   --------")
    for _, row in daily_curve.iterrows():
        print(f"   {int(row['hour']):2d}:00   {row['irradiance']:6.1f}    {row['predicted_power']:6.2f}")
    
    total_daily_energy = daily_curve['predicted_power'].sum() * 0.25  # 15分钟间隔转换为小时
    print(f"\n   预测日发电量: {total_daily_energy:.2f} MWh")
    
    # 演示3: 不同条件下的预测对比
    print("\n3. 不同天气条件预测对比:")
    conditions = [
        ("晴天", 800, 25),
        ("多云", 400, 22),
        ("阴天", 150, 18)
    ]
    
    print("   天气   辐照强度   温度   预测功率")
    print("   ----   --------   ----   --------")
    for condition, irradiance, temp in conditions:
        power = predictor.predict_single(
            total_irradiance=irradiance,
            direct_irradiance=irradiance * 0.7,
            global_irradiance=irradiance * 0.3,
            temperature=temp,
            atmosphere=915,
            hour=12,
            month=6,
            day_of_year=150
        )
        print(f"   {condition:4s}   {irradiance:6.0f}     {temp:2.0f}     {power:6.2f}")

if __name__ == "__main__":
    demo_predictions()