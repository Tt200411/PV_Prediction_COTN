import pandas as pd
import numpy as np
from data.data_loader import Dataset_Custom

# 测试数据加载器
def debug_data_loader():
    print("调试数据加载器...")
    
    for flag in ['train', 'val', 'test']:
        print(f"\n测试 {flag} 集:")
        try:
            dataset = Dataset_Custom(
                root_path='ETT-small',
                flag=flag,
                size=[96, 48, 24],  # seq_len, label_len, pred_len
                features='S',
                data_path='PV_Solar_Station_1.csv',
                target='Power',
                scale=True,
                timeenc=0,
                freq='t'
            )
            
            print(f"  数据集长度: {len(dataset)}")
            print(f"  data_x形状: {dataset.data_x.shape}")
            
        except Exception as e:
            print(f"  错误: {e}")
            
            # 手动计算边界来调试
            df = pd.read_csv('ETT-small/PV_Solar_Station_1.csv')
            num_train = int(len(df) * 0.8)
            
            if flag == 'train':
                border1, border2 = 0, num_train
            elif flag == 'val':
                border1, border2 = num_train-96, num_train
            else:  # test
                border1, border2 = num_train, len(df)
            
            data_length = border2 - border1
            usable_length = data_length - 96 - 24 + 1
            
            print(f"  边界: {border1} -> {border2}")
            print(f"  数据长度: {data_length}")
            print(f"  可用长度: {usable_length}")

if __name__ == "__main__":
    debug_data_loader()