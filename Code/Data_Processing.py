import timeit
start = timeit.default_timer()
import os
import shutil
import numpy as np
import pandas as pd
from make_dir import create_dir

# 读取关联数据ID
data_match_all = pd.read_csv('../Data/match(数据关联).csv', encoding='gbk')
df_match_all = pd.DataFrame(data_match_all, columns=['frame_num', 'lidar_track_id', 'dt_track_id'], dtype=int)
# 读取传感器数据
data_mea_all = pd.read_csv('../Data/match_dut(传感器观测).csv', encoding='gbk')
df_mea_all = pd.DataFrame(data_mea_all, columns=['stamp_sec', 'frame_num', 'track_id', 'nestCenter.x', 'nestCenter.y',
                                                 'relvelocity.x', 'relvelocity.y'], dtype=float)
# 读取真值数据
data_tru_all = pd.read_csv('../Data/match_gt(真值).csv', encoding='gbk')
df_tru_all = pd.DataFrame(data_tru_all, columns=['stamp_sec', 'frame_num', 'track_id', 'nestCenter.x', 'nestCenter.y',
                                                 'relvelocity.x', 'relvelocity.y'], dtype=float)

# 关联数据的帧数、真值ID、传感器ID
frame_match = df_match_all.iloc[:, 0]
tru_id_match = df_match_all.iloc[:, 1]
lidar_id_match = df_match_all.iloc[:, 2]

# 得到传感器所有的ID
sensor_id = []
for ele1 in lidar_id_match:
    if ele1 not in sensor_id:
        sensor_id.append(ele1)

# 创建文件夹
path1 = '../Objects_Data/'
name1 = 'ID'
'''
   以传感器目标ID创建文件夹，每一个传感器ID对应一个目标（一个传感器ID可能对应多个真值ID）
'''

create_dir(path1, name1, sensor_id)

# 创建文件夹，保存sensor_id
path2 = '../Objects_Data/'
name2 = 'sensor_id'
create_dir(path1, name2)


print('-------------------')
sens_id_data = pd.DataFrame({'sensor_id': sensor_id})
# 传感器ID保存为CSV文件，便于观察。
sens_id_data.to_csv('../Objects_Data/sensor_id/sensor_id.csv', index=False, sep=',')

# 遍历所有的传感器ID（目标ID）
for id in sensor_id:
    # 关联数据的长度
    n_1 = len(data_match_all)


    def read_id():

        data_id = []  # tru_id=0时lidar_id
        for i in range(0, n_1):
            if lidar_id_match[i] == id:  # 当传感器ID为id时，list添加对应索引的帧数、真值ID、传感器ID
                data_id.append([frame_match[i], tru_id_match[i], lidar_id_match[i]])
            else:
                continue
        # 将list转为数组
        data_id = np.array(data_id)
        # 将传感器ID为id的关联的帧数、真值ID存入对应id的文件夹
        file_id = pd.DataFrame({'frame_num': data_id[:, 0], 'tru_id': data_id[:, 1], 'lidar_id': data_id[:, 2]})
        #
        file_id.to_csv('../Objects_Data/ID' + str(id) + '/ID' + str(id) + '_match.csv', index=False, sep=',')
        return data_id, file_id


    # 单个传感器ID对应的帧数和真值ID
    data_id, file_id = read_id()
    # 读取传感器ID为id的目标数据的长度
    n_2 = len(data_id)
    # 对应的真值ID
    ele_id = data_id[:, 1]
    # 对应id的帧数
    frame_num_mat = data_id[:, 0]

    # 提取真值ID中的不同ID（一个传感器ID在不同时间段会有不同的真值ID）
    def func_ele():
        ele = []
        for e in ele_id:
            if e not in ele:
                ele.append(e)
        return ele

    ele = func_ele()  # 真值ID的不同ID

    # 第一模块：提取目标id不同真值ID对应的frame_num帧数
    # 提取传感器ID为id对应在不同时间段的真值ID
    def func_1():
        # 同一传感器ID对应真值ID最多为11个
        f0, f1, f2, f3 = [], [], [], []
        f4, f5, f6, f7 = [], [], [], []
        f8, f9, f10, f11 = [], [], [], []

        '''
            真值CSV文件中，在同一时间戳时，传感器ID对应的真值ID会出现两个或多个，
            但要按照关联数据来选择。所以需提取不同的真值ID对应的帧数。
        '''
        for i1 in range(0, n_2):  # 不同真值ID对应的frame_num帧数
            if ele_id[i1] == ele[0]:
                f0.append(frame_num_mat[i1])
            elif ele_id[i1] == ele[1]:
                f1.append(frame_num_mat[i1])
            elif ele_id[i1] == ele[2]:
                f2.append(frame_num_mat[i1])
            elif ele_id[i1] == ele[3]:
                f3.append(frame_num_mat[i1])
            elif ele_id[i1] == ele[4]:
                f4.append(frame_num_mat[i1])
            elif ele_id[i1] == ele[5]:
                f5.append(frame_num_mat[i1])
            elif ele_id[i1] == ele[6]:
                f6.append(frame_num_mat[i1])
            elif ele_id[i1] == ele[7]:
                f7.append(frame_num_mat[i1])
            elif ele_id[i1] == ele[8]:
                f8.append(frame_num_mat[i1])
            elif ele_id[i1] == ele[9]:
                f9.append(frame_num_mat[i1])
            elif ele_id[i1] == ele[10]:
                f10.append(frame_num_mat[i1])
            elif ele_id[i1] == ele[11]:
                f11.append(frame_num_mat[i1])

        return f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11


    f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11 = func_1()
    # 传感器ID为id对应的帧数
    frame_num = data_id[:, 0]

    # 第二模块：提取目标id的观测数据
    # 观测数据的总长度
    n_3 = len(data_mea_all)
    # 观测数据对应的物理量
    stamp_sec_mea = df_mea_all.iloc[:, 0]
    frame_num_mea = df_mea_all.iloc[:, 1]
    track_id_mea = df_mea_all.iloc[:, 2]
    nestCenter_x_mea = df_mea_all.iloc[:, 3]
    nestCenter_y_mea = df_mea_all.iloc[:, 4]
    relvelocity_x_mea = df_mea_all.iloc[:, 5]
    relvelocity_y_mea = df_mea_all.iloc[:, 6]


    def read_mea():
        data_mea = []
        for i in range(0, n_3):
            # 遍历观测数据的索引，当帧数时id的帧数（条件1） and 传感器ID=id（条件2），将对应索引的物理量添加进list
            if (frame_num_mea[i] in frame_num) and (track_id_mea[i] == id):
                data_mea.append([stamp_sec_mea[i], frame_num_mea[i], track_id_mea[i], nestCenter_x_mea[i],
                                 nestCenter_y_mea[i], relvelocity_x_mea[i], relvelocity_y_mea[i]])
            else:
                continue
        # 将list转为数组
        data_mea = np.array(data_mea)

        # 得到相应的传感器ID的相应时间戳、帧数、物理量，保存为CSV文件，以便观察和访问
        file_mea = pd.DataFrame({'stamp_sec_mea': data_mea[:, 0], 'frame_num_mea': data_mea[:, 1],
                                 'track_id_mea': data_mea[:, 2], 'nestCenter_x_mea': data_mea[:, 3],
                                 'nestCenter_y_mea': data_mea[:, 4], 'velocity_x_mea': data_mea[:, 5],
                                 'velocity_y_mea': data_mea[:, 6]})

        file_mea.to_csv('../Objects_Data/ID' + str(id) + '/ID' + str(id) + '_mea.csv', index=False, sep=',')

        return data_mea, file_mea


    # 读取目标id的观测值
    data_mea, file_mea = read_mea()

    # 第三模块：提取目标id对应的真值数据
    # 真值数据的总长度
    n_4 = len(data_tru_all)
    # 真值对应的物理量
    stamp_sec_tru = df_tru_all.iloc[:, 0]
    frame_num_tru = df_tru_all.iloc[:, 1]
    track_id_tru = df_tru_all.iloc[:, 2]
    nestCenter_x_tru = df_tru_all.iloc[:, 3]
    nestCenter_y_tru = df_tru_all.iloc[:, 4]
    relvelocity_x_tru = df_tru_all.iloc[:, 5]
    relvelocity_y_tru = df_tru_all.iloc[:, 6]

    # 定义读取目标id对应的真值数据
    def read_tru():
        data_tru = []
        for i in range(0, n_4):
            '''
                遍历真值数据的索引，当不同真值ID的帧数在一一对应的真值ID的索引中（条件1） 
                 and  不同的真值ID一一对应（条件2），将对应索引的物理量添加到list
            '''
            if (frame_num_tru[i] in f0) and (track_id_tru[i] == ele[0]):
                data_tru.append([stamp_sec_tru[i], frame_num_tru[i], track_id_tru[i], nestCenter_x_tru[i],
                                 nestCenter_y_tru[i], relvelocity_x_tru[i], relvelocity_y_tru[i]])
            elif (frame_num_tru[i] in f1) and (track_id_tru[i] == ele[1]):
                data_tru.append([stamp_sec_tru[i], frame_num_tru[i], track_id_tru[i], nestCenter_x_tru[i],
                                 nestCenter_y_tru[i], relvelocity_x_tru[i], relvelocity_y_tru[i]])
            elif (frame_num_tru[i] in f2) and (track_id_tru[i] == ele[2]):
                data_tru.append([stamp_sec_tru[i], frame_num_tru[i], track_id_tru[i], nestCenter_x_tru[i],
                                 nestCenter_y_tru[i], relvelocity_x_tru[i], relvelocity_y_tru[i]])
            elif (frame_num_tru[i] in f3) and (track_id_tru[i] == ele[3]):
                data_tru.append([stamp_sec_tru[i], frame_num_tru[i], track_id_tru[i], nestCenter_x_tru[i],
                                 nestCenter_y_tru[i], relvelocity_x_tru[i], relvelocity_y_tru[i]])
            elif (frame_num_tru[i] in f4) and (track_id_tru[i] == ele[4]):
                data_tru.append([stamp_sec_tru[i], frame_num_tru[i], track_id_tru[i], nestCenter_x_tru[i],
                                 nestCenter_y_tru[i], relvelocity_x_tru[i], relvelocity_y_tru[i]])
            elif (frame_num_tru[i] in f5) and (track_id_tru[i] == ele[5]):
                data_tru.append([stamp_sec_tru[i], frame_num_tru[i], track_id_tru[i], nestCenter_x_tru[i],
                                 nestCenter_y_tru[i], relvelocity_x_tru[i], relvelocity_y_tru[i]])
            elif (frame_num_tru[i] in f6) and (track_id_tru[i] == ele[6]):
                data_tru.append([stamp_sec_tru[i], frame_num_tru[i], track_id_tru[i], nestCenter_x_tru[i],
                                 nestCenter_y_tru[i], relvelocity_x_tru[i], relvelocity_y_tru[i]])
            elif (frame_num_tru[i] in f7) and (track_id_tru[i] == ele[7]):
                data_tru.append([stamp_sec_tru[i], frame_num_tru[i], track_id_tru[i], nestCenter_x_tru[i],
                                 nestCenter_y_tru[i], relvelocity_x_tru[i], relvelocity_y_tru[i]])
            elif (frame_num_tru[i] in f8) and (track_id_tru[i] == ele[8]):
                data_tru.append([stamp_sec_tru[i], frame_num_tru[i], track_id_tru[i], nestCenter_x_tru[i],
                                 nestCenter_y_tru[i], relvelocity_x_tru[i], relvelocity_y_tru[i]])
            elif (frame_num_tru[i] in f9) and (track_id_tru[i] == ele[9]):
                data_tru.append([stamp_sec_tru[i], frame_num_tru[i], track_id_tru[i], nestCenter_x_tru[i],
                                 nestCenter_y_tru[i], relvelocity_x_tru[i], relvelocity_y_tru[i]])
            elif (frame_num_tru[i] in f10) and (track_id_tru[i] == ele[10]):
                data_tru.append([stamp_sec_tru[i], frame_num_tru[i], track_id_tru[i], nestCenter_x_tru[i],
                                 nestCenter_y_tru[i], relvelocity_x_tru[i], relvelocity_y_tru[i]])
            elif (frame_num_tru[i] in f11) and (track_id_tru[i] == ele[11]):
                data_tru.append([stamp_sec_tru[i], frame_num_tru[i], track_id_tru[i], nestCenter_x_tru[i],
                                 nestCenter_y_tru[i], relvelocity_x_tru[i], relvelocity_y_tru[i]])
        # 将list转为数组
        data_tru = np.array(data_tru)
        # 将目标id对应的真值数据保存
        file_tru = pd.DataFrame({'stamp_sec_tru': data_tru[:, 0], 'frame_num_tru': data_tru[:, 1],
                                 'track_id_tru': data_tru[:, 2], 'nestCenter_x_tru': data_tru[:, 3],
                                 'nestCenter_y_tru': data_tru[:, 4], 'velocity_x_tru': data_tru[:, 5],
                                 'velocity_y_tru': data_tru[:, 6]})

        file_tru.to_csv('../Objects_Data/ID' + str(id) + '/ID' + str(id) + '_tru.csv', index=False, sep=',')
        return data_tru, file_tru


    data_tru, file_tru = read_tru()

    # 第四模块：将目标id对应的真值数据、观测数据一一对应匹配、保存
    n = len(data_id)

    # 定义观测误差函数
    def mear_error():
        me_error_1 = []
        me_error_2 = []
        me_error_3 = []
        me_error_4 = []
        for i in range(0, n):
            me_error_1.append(data_mea[i][3] - data_tru[i][3])
            me_error_2.append(data_mea[i][4] - data_tru[i][4])
            me_error_3.append(data_mea[i][5] - data_tru[i][5])
            me_error_4.append(data_mea[i][6] - data_tru[i][6])
        return me_error_1, me_error_2, me_error_3, me_error_4


    me_error_1, me_error_2, me_error_3, me_error_4 = mear_error()

    # 定义观测误差方差函数
    def me_variance():
        me_var_1 = []
        me_var_2 = []
        me_var_3 = []
        me_var_4 = []
        for i in range(0, n):
            me_var_1.append(np.var(me_error_1[:i + 1]))
            me_var_2.append(np.var(me_error_2[:i + 1]))
            me_var_3.append(np.var(me_error_3[:i + 1]))
            me_var_4.append(np.var(me_error_4[:i + 1]))
        return me_var_1, me_var_2, me_var_3, me_var_4


    me_var_1, me_var_2, me_var_3, me_var_4 = me_variance()

    me_error_1 = np.array(me_error_1)
    me_error_2 = np.array(me_error_2)
    me_error_3 = np.array(me_error_3)
    me_error_4 = np.array(me_error_4)
    me_var_1 = np.array(me_var_1)
    me_var_2 = np.array(me_var_2)
    me_var_3 = np.array(me_var_3)
    me_var_4 = np.array(me_var_4)

    # 目标id的观测值和真值以及对应物理量的观测误差和观测误差方差保存为CSV文件
    file_id_data = pd.DataFrame({'stamp_sec': data_mea[:n, 0], 'frame_num': data_mea[:n, 1],
                                 'track_id_tru': data_tru[:n, 2], 'track_id_mea': data_mea[:n, 2],
                                 'nestCenter_x_tru': data_tru[:n, 3], 'nestCenter_x_mea': data_mea[:n, 3],
                                 'mea_error_x': me_error_1, 'mea_var_x': me_var_1, 'nestCenter_y_tru': data_tru[:n, 4],
                                 'nestCenter_y_mea': data_mea[:n, 4], 'mea_error_y': me_error_2,
                                 'mea_var_y': me_var_2, 'velocity_x_tru': data_tru[:n, 5],
                                 'velocity_x_mea': data_mea[:n, 5], 'mea_error_vx': me_error_3,
                                 'mea_var_vx': me_var_3, 'velocity_y_tru': data_tru[:n, 6],
                                 'velocity_y_mea': data_mea[:n, 6], 'mea_error_vy': me_error_4, 'mea_var_vy': me_var_4})

    file_id_data.to_csv('../Objects_Data/ID' + str(id) + '/ID' + str(id) + '_data.csv',
                        index=False, sep=',')

    print('目标ID为' + str(id) + '的物理数据保存完成')
    print('-------------------')

end = timeit.default_timer()
print('Running time: %s Seconds' % (end - start))
print('数据处理完成')
