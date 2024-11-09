# 将原始的数据转换为模型所需的数据格式
'''
TODO
1\ 将vehicle的基本信息转为模型所需的格式来编排
2\ 将agent_type转为由数字编码，排除行人，从1到6，分别是car/
3\ 将traffic_state转为由数字编码，从1到5，按南北、东西方向区分
4\ 组合上述信息至同一文件中
'''

import os
import pandas as pd
import numpy as np

# 实现功能：筛选出基本信息，顺序: frame_id,track_id,x,y,vx,vy,ax,ay,length,width,agent_type
def get_basic_info(file_path):
    df = pd.read_csv(file_path)
    # 指定需要的列，这里还保留了时间戳，因为要和交通灯状态关联，在关联后删除
    basic_info = df[['frame_id','timestamp_ms','track_id','x','y','vx','vy','ax','ay','length','width','agent_type']]
    # 去除frame_id = 0的情况
    basic_info = basic_info[basic_info['frame_id']!= 0]
    # 重排顺序
    basic_info_reranged = basic_info[['frame_id','timestamp_ms','track_id','x','y','vx','vy','ax','ay','length','width','agent_type']]
    # 将agent_type转为数字编码，排除行人，从1到6，分别是car/motorcycle/bicycle/tricycle/bus/truck
    agent_type_dict = {'car':1,'motorcycle':2, 'bicycle':3, 'tricycle':4, 'bus':5, 'truck':6}
    basic_info_reranged['agent_type'] = basic_info_reranged['agent_type'].apply(lambda x: agent_type_dict[x])
    
    # 后续需要根据file名自动命名，fix me!
    output_file = os.path.join(os.path.dirname(file_path), 'basic_info.csv')
    basic_info_reranged.to_csv(output_file, index=False)
    
# 实现功能：将traffic_state从对应的源文件中读取(还需要比较时间戳先后)，并转为对应的数字编码加入到basic_info中
def get_traffic_state(traffic_state_file_path, basic_info_file_path):
    df_traf = pd.read_csv(traffic_state_file_path)
    df_veh = pd.read_csv(basic_info_file_path)
    # 将时间戳转为datetime类型，进行比较以确定veh所对应的traffic_state
    df_traf['timestamp(ms)'] = pd.to_datetime(df_traf['timestamp(ms)'], unit='ms')
    df_veh['timestamp_ms'] = pd.to_datetime(df_veh['timestamp_ms'], unit='ms')
    # 在df_veh中的末尾增加一列，用于存储traffic_state
    df_veh['traffic_state'] = None
    
    
    def find_light_state(veh_timestamp, traffic_state_df):
        matching_states = traffic_state_df[traffic_state_df['timestamp(ms)'] <= veh_timestamp]
        if not matching_states.empty:
            # 取最近的变化状态,只取第三第四列，代表traffic_light_1和2，前者为南北通行、后者为东西通行
            last_state = matching_states.iloc[-1]
            traffic_light_1 = last_state[2]
            traffic_light_2 = last_state[3]
            # 映射为数字编码
            # 定义映射关系
            state_mapping = {
                "1,0":'1',
                "3,0":'2',
                "0,0":'3',
                "0,1":'4',
                "0,3":'5'
            }
            key = f"{traffic_light_1},{traffic_light_2}"
            return state_mapping.get(key, None)
        return None
    
    # 设定traffic_state的赋值条件：当veh中的timestamp大于traffic_state中的转换时刻，小于下一次的转换时刻，则veh的traffic_state为上一次转换时刻的traffic_state值
    for i, row in df_veh.iterrows():
        veh_timestamp = row['timestamp_ms']
        light_state = find_light_state(veh_timestamp, df_traf)
        if light_state is not None:
            df_veh.at[i, 'traffic_state'] = light_state
    
    output_file = os.path.join(os.path.dirname(basic_info_file_path), 'basic_info_traffic_state.csv')
    df_veh.to_csv(output_file, index=False)

# 实现功能：删除不需要的列，同时在保存时舍弃最上面的标题行[frame_id,timestamp_ms之类的
def delete_unnecessary_columns(file_path):
    df = pd.read_csv(file_path)
    # 指定删除的列
    column2delete = ['timestamp_ms']
    df = df.drop(columns=column2delete)
    df.to_csv(file_path, index=False, header=False) # 保存时舍弃最上面的标题行
    
    
if __name__ == '__main__':
    # test single file
    basic_file_path = '/root/trajectory_prediction/KI_GAN/scripts/dataset_example/original_data/Veh_smoothed_tracks.csv'
    get_basic_info(basic_file_path)
    
    basic_info_file_path = '/root/trajectory_prediction/KI_GAN/scripts/dataset_example/original_data/basic_info.csv'
    traffic_state_file_path = '/root/trajectory_prediction/KI_GAN/scripts/dataset_example/original_data/TrafficLight_8_07_1.csv'
    get_traffic_state(traffic_state_file_path,basic_info_file_path)
    
    basic_info_traffic_state_file_path = '/root/trajectory_prediction/KI_GAN/scripts/dataset_example/original_data/basic_info_traffic_state.csv'
    delete_unnecessary_columns(basic_info_traffic_state_file_path)
    
    # batch process 
    # 遍历文件夹下的所有文件，并调用相应的函数