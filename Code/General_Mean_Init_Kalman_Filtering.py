'''使用常规的Kalman滤波，对车辆前方多目标进行状态估计。通过计算激光雷达测量值（真值）和毫米波雷达测量值
的误差，从而计算误差方差，近似观测噪声协方差矩阵。使用0.5、0.1、0.05、0.01经验初始化观测噪声协方差矩阵，
并和均值初始化做对比。'''
import shutil
import timeit

start = timeit.default_timer()
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
# 中文字体
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
# 读取关联数据CSV文件
data_match = pd.read_csv('../Data/match(数据关联).csv', encoding='gbk')
df_match = pd.DataFrame(data_match, columns=['frame_num', 'lidar_track_id', 'dt_track_id'], dtype=int)
# 关联数据的帧数、真值ID、传感器ID
frame_match = list(df_match.iloc[:, 0])
tru_id_match = list(df_match.iloc[:, 1])
lidar_id_match = list(df_match.iloc[:, 2])
# 设定20组样本数据进行测试
test_groups = int(1000 / 50)

# 创建test1-test20的20个文件夹
path1 = '../General_Kalman_Filtering_State_Test/'

name = 'test'
str1 = 'exp1', 'exp2', 'exp3', 'init_R'
str2 = 'hat', 'mea_err', 'rel_err'
# for i in range(0, test_groups):
#     sExists = os.path.exists(path1 + name + str(i + 1))
#     if sExists:
#         shutil.rmtree(path1 + name + str(i + 1))
#         os.makedirs(path1 + name + str(i + 1))
#     else:
#         os.makedirs(path1 + name + str(i + 1))
#     print("%s 目录创建成功" % (name + str(i + 1)))
#     path2 = path1 + name + str(i + 1) + '/'
#     for i_1 in str1:
#         sExists1 = os.path.exists(path2 + i_1)
#         if sExists1:
#             shutil.rmtree(path2 + i_1)
#             os.makedirs(path2 + i_1)
#         else:
#             os.makedirs(path2 + i_1)
#         print("%s 目录创建成功" % (i_1))
#         path3 = path2 + i_1 + '/'
#         for i_2 in str2:
#             sExists2 = os.path.exists(path3 + i_2)
#             if sExists2:
#                 shutil.rmtree(path3 + i_2)
#                 os.makedirs(path3 + i_2)
#             else:
#                 os.makedirs(path3 + i_2)
#             print("%s 目录创建成功" % (i_2))
#     print('\n')
#
# print('文件夹创建完成')
# print('-------------------')
# 提取50帧数据中的传感器ID（0-49，50-99, ...950-999）
group_id = []
for i2 in range(0, test_groups):
    ele_id = []
    for ele in lidar_id_match[frame_match.index(i2 * 50):frame_match.index(i2 * 50 + 50)]:
        if ele not in ele_id:
            ele_id.append(ele)
    group_id.append(ele_id)

# 提取50帧中满足达到50帧的传感器ID，其余不满50帧的（若帧数缺失较少，手动补全；缺失较多，不计该目标）
test_group_id = []
for i3 in range(0, test_groups):
    ele_id1 = []
    for ele1 in group_id[i3]:
        if lidar_id_match[frame_match.index(i3 * 50):frame_match.index(i3 * 50 + 50)].count(ele1) == 50:
            ele_id1.append(ele1)
    test_group_id.append(ele_id1)


def general_mean_init(str_mode):
    for mode in str_mode:
        mse_mea_tru_avg = []
        mse_hat_tru_avg = []
        for i4 in range(0, len(test_group_id)):  # 20组测试
            # 每组的目标ID个数
            obj_num = len(test_group_id[i4])
            #  分别存放多个目标的nestCenter.x、nestCenter.y、velocity.x、velocity.y的真值的list
            tru_nCx_all, tru_nCy_all, tru_vx_all, tru_vy_all = [], [], [], []
            # 分别存放多个目标的nestCenter.x、nestCenter.y、velocity.x、velocity.y的观测值的list
            mea_nCx_all, mea_nCy_all, mea_vx_all, mea_vy_all = [], [], [], []
            mea_nCx_err, mea_nCy_err, mea_vx_err, mea_vy_err = [], [], [], []
            # 分别存放多个目标的nestCenter.x、nestCenter.y、velocity.x、velocity.y的观测误差方差的list
            mea_nCx_var, mea_nCy_var, mea_vx_var, mea_vy_var = [], [], [], []
            # 遍历该数据样本的所有目标ID
            dt = []
            for id in test_group_id[i4]:
                # 读取目标数据
                obj_data = pd.read_csv('../Objects_Data/ID' + str(id) + '/ID' + str(id) + '_data.csv',
                    encoding='gbk')
                df_obj = obj_data.iloc[:, 1]  # 目标数据的所有的帧数
                n_obj = len(df_obj)

                # 提取对应帧数段的帧数
                def frame_num():
                    f_n = []
                    for i5 in range(i4 * 50, i4 * 50 + 50):
                        for ele2 in np.array(df_obj):
                            if ele2 == i5:
                                f_n.append(ele2)
                    return f_n

                f_n = frame_num()

                # 提取对应帧数帧数段的帧数的索引
                def frame_index():
                    index_obj = []
                    for i6 in range(0, 50):
                        for i7 in range(0, n_obj):
                            if df_obj[i7] == f_n[i6]:
                                index_obj.append(i7)
                    return index_obj


                index_obj = frame_index()
                # 提取对应帧数段的目标数据
                dt_obj = obj_data.iloc[index_obj[0]:index_obj[-1] + 1]
                dt.append(dt_obj)
                # 对应帧数段的目标物理量
                tru_nCx_obj, mea_nCx_obj, mea_nCx_err_obj, mea_nCx_var_obj, \
                tru_nCy_obj, mea_nCy_obj, mea_nCy_err_obj, mea_nCy_var_obj, \
                tru_vx_obj, mea_vx_obj, mea_vx_err_obj, mea_vx_var_obj, \
                tru_vy_obj, mea_vy_obj, mea_vy_err_obj, mea_vy_var_obj, \
                    = np.array(dt_obj.iloc[:, 4]), np.array(dt_obj.iloc[:, 5]), np.array(dt_obj.iloc[:, 6]), np.array(
                    dt_obj.iloc[:, 7]), \
                      np.array(dt_obj.iloc[:, 8]), np.array(dt_obj.iloc[:, 9]), np.array(dt_obj.iloc[:, 10]), np.array(
                    dt_obj.iloc[:, 11]), \
                      np.array(dt_obj.iloc[:, 12]), np.array(dt_obj.iloc[:, 13]), np.array(dt_obj.iloc[:, 14]), np.array(
                    dt_obj.iloc[:, 15]), \
                      np.array(dt_obj.iloc[:, 16]), np.array(dt_obj.iloc[:, 17]), np.array(dt_obj.iloc[:, 18]), np.array(
                    dt_obj.iloc[:, 19])
                # 分别添加多个目标的tru_nCx、tru_nCy、tru_vx、tru_vy
                tru_nCx_all.append(tru_nCx_obj)
                tru_nCy_all.append(tru_nCy_obj)
                tru_vx_all.append(tru_vx_obj)
                tru_vy_all.append(tru_vy_obj)
                # 分别添加多个目标的mea_nCx、mea_nCy、mea_vx、mea_vy
                mea_nCx_all.append(mea_nCx_obj)
                mea_nCy_all.append(mea_nCy_obj)
                mea_vx_all.append(mea_vx_obj)
                mea_vy_all.append(mea_vy_obj)
                # 分别添加多个目标的mea_nCx_err、mea_nCy_err、mea_vx_err、mea_vy_err
                mea_nCx_err.append(mea_nCx_err_obj)
                mea_nCy_err.append(mea_nCy_err_obj)
                mea_vx_err.append(mea_vx_err_obj)
                mea_vy_err.append(mea_vy_err_obj)
                # 分别添加多个目标的nCx、nCy、vx、vy的观测误差方差
                mea_nCx_var.append(mea_nCx_var_obj)
                mea_nCy_var.append(mea_nCy_var_obj)
                mea_vx_var.append(mea_vx_var_obj)
                mea_vy_var.append(mea_vy_var_obj)

            # 求观测误差方差均值
            '''
                均值初始化
            '''
            # 误差均值
            mean_nCx_err = np.mean(mea_nCx_err)
            mean_nCy_err = np.mean(mea_nCy_err)
            mean_vx_err = np.mean(mea_vx_err)
            mean_vy_err = np.mean(mea_vy_err)

            # 方差均值
            mean_nCx_var = np.mean(mea_nCx_var)
            mean_nCy_var = np.mean(mea_nCy_var)
            mean_vx_var = np.mean(mea_vx_var)
            mean_vy_var = np.mean(mea_vy_var)

            # 多个目标的nCx、nCx、vx、vy的真值
            tru_nCx, tru_nCy, tru_vx, tru_vy = [], [], [], []
            for i8 in range(0, 50):
                nCx_t = []
                for tru_nCxobj in tru_nCx_all:
                    nCx_t.append(tru_nCxobj[i8])
                tru_nCx.append(nCx_t)
                nCy_t = []
                for tru_nCyobj in tru_nCy_all:
                    nCy_t.append(tru_nCyobj[i8])
                tru_nCy.append(nCy_t)
                vx_t = []
                for tru_vxobj in tru_vx_all:
                    vx_t.append(tru_vxobj[i8])
                tru_vx.append(vx_t)
                vy_t = []
                for tru_vyobj in tru_vy_all:
                    vy_t.append(tru_vyobj[i8])
                tru_vy.append(vy_t)
            # 多个目标的nCx、nCx、vx、vy的观测值
            mea_nCx, mea_nCy, mea_vx, mea_vy = [], [], [], []
            for i9 in range(0, 50):
                nCx_m = []
                for mea_nCxobj in mea_nCx_all:
                    nCx_m.append(mea_nCxobj[i9])
                mea_nCx.append(nCx_m)
                nCy_m = []
                for mea_nCyobj in mea_nCy_all:
                    nCy_m.append(mea_nCyobj[i9])
                mea_nCy.append(nCy_m)
                vx_m = []
                for mea_vxobj in mea_vx_all:
                    vx_m.append(mea_vxobj[i9])
                mea_vx.append(vx_m)
                vy_m = []
                for mea_vyobj in mea_vy_all:
                    vy_m.append(mea_vyobj[i9])
                mea_vy.append(vy_m)

            # Kalman filtering估计
            '''
                Kalman filtering
                状态观测方程如下：
                                            Xk = Ak * Xk-1 + wk
                                            Yk = Hk * Xk +vk
                卡尔曼滤波有预测和校正两个步骤：
                预测：
                                        Xk|k-1 = Ak * Xk-1|k-1
                                        Pk|k-1 = Hk * Pk-1 * Hk.T +Qk
                校正：
                                        Kk = (Pk|k-1 * Hk.T) / (Hk * Pk|k-1 * Hk.T + Rk)
                                        Xk|k = Xk|k-1 + Kk * (Y - Hk * Xk|k-1)
                                        Pk|k = (I - Kk * Hk) * Pk|k-1
                k|k-1为先验估计值（预测值）、k|k为后验估计值
            '''
            Ak = np.array([[1, 0, 0.1, 0],
                           [0, 1, 0, 0.1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])  # 状态转移矩阵

            Hk = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])  # 观测矩阵
            if mode == 'exp1':
                Qk = np.array([[0.5, 0, 0, 0],
                               [0, 0.5, 0, 0],
                               [0, 0, 0.5, 0],
                               [0, 0, 0, 0.5]])  # 过程噪声协方差矩阵
                Rk = np.array([[0.5, 0, 0, 0],
                               [0, 0.5, 0, 0],
                               [0, 0, 0.5, 0],
                               [0, 0, 0, 0.5]])   # 观测噪声协方差矩阵
            if mode == 'exp2':
                Qk = np.array([[0.1, 0, 0, 0],
                               [0, 0.1, 0, 0],
                               [0, 0, 0.1, 0],
                               [0, 0, 0, 0.1]])  # 过程噪声协方差矩阵
                Rk = np.array([[0.1, 0, 0, 0],
                               [0, 0.1, 0, 0],
                               [0, 0, 0.1, 0],
                               [0, 0, 0, 0.1]])   # 观测噪声协方差矩阵
            if mode == 'exp3':
                Qk = np.array([[0.05, 0, 0, 0],
                               [0, 0.05, 0, 0],
                               [0, 0, 0.05, 0],
                               [0, 0, 0, 0.05]])  # 过程噪声协方差矩阵
                Rk = np.array([[0.05, 0, 0, 0],
                               [0, 0.05, 0, 0],
                               [0, 0, 0.05, 0],
                               [0, 0, 0, 0.5]])   # 观测噪声协方差矩阵
            if mode == 'init_R':
                Qk = np.array([[mean_nCx_var, 0, 0, 0],
                               [0, mean_nCy_var, 0, 0],
                               [0, 0, mean_vx_var, 0],
                               [0, 0, 0, mean_vy_var]])  # 过程噪声协方差矩阵
                Rk = np.array([[mean_nCx_var, 0, 0, 0],
                               [0, mean_nCy_var, 0, 0],
                               [0, 0, mean_vx_var, 0],
                               [0, 0, 0, mean_vy_var]])   # 观测噪声协方差矩阵
            #
            # Rk = np.array([[0.05, 0, 0, 0],
            #                [0, 0.05, 0, 0],
            #                [0, 0, 0.05, 0],
            #                [0, 0, 0, 0.05]])  # 观测噪声协方差矩阵
            # Rk = np.array([[mean_nCx_var, 0, 0, 0],
            #                [0, mean_nCx_var, 0, 0],
            #                [0, 0, mean_nCx_var, 0],
            #                [0, 0, 0, mean_nCx_var]])  # 观测噪声协

            I = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])  # 单位矩阵


            def kalmanfilter():

                mea_X = []
                tru_X = []
                pre_X = []
                hat_X = []

                Xplus1 = np.array([tru_nCx[0], tru_nCy[0], tru_vx[0], tru_vy[0]])  # 上一时刻的多个目标的估计值

                Pplus1 = np.array([[0.01, 0, 0, 0],
                                   [0, 0.01, 0, 0],
                                   [0, 0, 0.01, 0],
                                   [0, 0, 0, 0.01]])  # 上一时刻的误差协方差矩阵
                for i10 in range(0, 50):
                    #Y = np.array([[mea_nCx[i10]], [mea_nCy[i10]], [mea_vx[i10]], [mea_vy[i10]]])  # 当前时刻的观测值
                    Y = np.array([mea_nCx[i10], mea_nCy[i10], mea_vx[i10], mea_vy[i10]])  # 当前时刻的观测值
                    tru = np.array([tru_nCx[i10], tru_nCy[i10], tru_vx[i10], tru_vy[i10]])  # 当前时刻的真值
                    # Xminus1 =  np.array([tru_nCx[i10], tru_nCy[i10], tru_vx[i10], tru_vy[i10]])   # 预测当前时刻的多目标的状态
                    Xminus1 = np.dot(Ak, Xplus1)  # 预测当前时刻的多目标的状态
                    Pminus1 = np.dot(np.dot(Ak, Pplus1), Ak.T) + Qk  # 预测误差协方差矩阵
                    Kk = np.dot(np.dot(Pminus1, Hk.T), np.linalg.pinv(np.dot(np.dot(Hk, Pminus1), Hk.T) + Rk))  # 卡尔曼增益
                    Xplus1 = Xminus1 + np.dot(Kk, (Y - np.dot(Hk, Xminus1)))
                    Pplus1 = np.dot((I - np.dot(Kk, Hk)), Pminus1)

                    mea_X.append(Y)  # 获取多个目标的状态观测值
                    tru_X.append(tru)  # 获取多个目标的状态真实值
                    pre_X.append(Xminus1)  # 获取多个目标的状态先验估计值（预测值）
                    hat_X.append(Xplus1)  # 获取多个目标的状态后验估计值

                return mea_X, tru_X, pre_X, hat_X


            mea_X, tru_X, pre_X, hat_X = kalmanfilter()
            # print(mea_X)
        #     # 将list转数组
            mea_X = np.array(mea_X)  # (50, 4, 5)
            tru_X = np.array(tru_X)  # (50, 4, 5)
            pre_X = np.array(pre_X)  # (50, 4, 5)
            hat_X = np.array(hat_X)  # (50, 4, 5)

            # 保存观测值、真值、预测值、真值
            work_tru_mea_pre_hat = pd.ExcelWriter('../General_Kalman_Filtering_State_Test/test'
                                                  + str(i4 + 1) + '/'+mode+'/hat/tru_mea_pre_hat' + str(i4 + 1) + '.xlsx')
            # 保存均方误差（观测值和真值、估计值和真值）
            work_mea_hat_mse = pd.ExcelWriter('../General_Kalman_Filtering_State_Test/test'
                                              + str(i4 + 1) + '/'+mode+'/hat/mea_hat_mse' + str(i4 + 1) + '.xlsx')
            # 保存观测误差、估计误差
            work_mea_hat_err = pd.ExcelWriter('../General_Kalman_Filtering_State_Test/test'
                                              + str(i4 + 1) + '/'+mode+'/mea_err/mea_hat_err' + str(i4 + 1) + '.xlsx')
            # 保存观测误差均值、估计误差均值
            work_mean_mea_hat_err = pd.ExcelWriter('../General_Kalman_Filtering_State_Test/test'
                                                   + str(i4 + 1) + '/'+mode+'/mea_err/mean_mea_hat_err' + str(i4 + 1) + '.xlsx')
            # 保存观测相对误差、估计相对误差
            work_mea_hat_rel_err = pd.ExcelWriter('../General_Kalman_Filtering_State_Test/test'
                                                  + str(i4 + 1) + '/'+mode+'/rel_err/mea_hat_rel_err' + str(i4 + 1) + '.xlsx')
            work_mean_mea_hat_rel_err = pd.ExcelWriter('../General_Kalman_Filtering_State_Test/test'
                                               + str(i4 + 1) + '/'+mode+'/rel_err/mean_mea_hat_rel_err' + str(i4 + 1) + '.xlsx')

            mse_mea_tru_group = []
            mse_hat_tru_group = []
            for i11 in range(0, obj_num):
                nCx_tru, nCy_tru, vx_tru, vy_tru = tru_X[:, :, i11][:, 0], tru_X[:, :, i11][:, 1], tru_X[:, :, i11][:,2], tru_X[:, :, i11][:, 3]
                nCx_mea, nCy_mea, vx_mea, vy_mea = mea_X[:, :, i11][:, 0], mea_X[:, :, i11][:, 1], mea_X[:, :, i11][:,2], mea_X[:, :, i11][:, 3]
                nCx_pre, nCy_pre, vx_pre, vy_pre = pre_X[:, :, i11][:, 0], pre_X[:, :, i11][:, 1], pre_X[:, :, i11][:,2], pre_X[:, :, i11][:, 3]
                nCx_hat, nCy_hat, vx_hat, vy_hat = hat_X[:, :, i11][:, 0], hat_X[:, :, i11][:, 1], hat_X[:, :, i11][:,2], hat_X[:, :, i11][:, 3]

                '''计算误差'''
                # 观测误差
                nCx_mea_err,  nCy_mea_err = nCx_mea - nCx_tru, nCy_mea - nCy_tru
                vx_mea_err, vy_mea_err = vx_mea - vx_tru, vy_mea - vy_tru
                # 估计误差
                nCx_hat_err, nCy_hat_err = nCx_hat - nCx_tru, nCy_hat - nCy_tru
                vx_hat_err, vy_hat_err = vx_hat - vx_tru, vy_hat - vy_tru

                '''计算误差均值'''
                # 观测误差均值
                mean_nCx_mea_err, mean_nCy_mea_err = np.mean(nCx_mea_err), np.mean(nCy_mea_err)
                mean_vx_mea_err, mean_vy_mea_err = np.mean(vx_mea_err), np.mean(vy_mea_err)
                # 估计误差均值
                mean_nCx_hat_err, mean_nCy_hat_err = np.mean(nCx_hat_err), np.mean(nCy_hat_err)
                mean_vx_hat_err, mean_vy_hat_err = np.mean(vx_hat_err), np.mean(vy_hat_err)

                '''计算相对误差'''
                # 观测相对误差
                nCx_mea_rel_err, nCy_mea_rel_err = nCx_mea_err / nCx_tru, nCy_mea_err / nCy_tru
                vx_mea_rel_err, vy_mea_rel_err = vx_mea_err / vx_tru, vy_mea_err / vy_tru
                # 估计相对误差
                nCx_hat_rel_err, nCy_hat_rel_err = nCx_hat_err / nCx_tru, nCy_hat_err / nCy_tru
                vx_hat_rel_err, vy_hat_rel_err = vx_hat_err / vx_tru, vy_hat_err / vy_tru

                '''计算相对误差均值'''
                # 观测相对误差均值
                mean_nCx_mea_rel_err, mean_nCy_mea_rel_err = np.mean(nCx_mea_rel_err), np.mean(nCy_mea_rel_err)
                mean_vx_mea_rel_err, mean_vy_mea_rel_err = np.mean(vx_mea_rel_err), np.mean(vy_mea_rel_err)
                # 估计相对误差均值
                mean_nCx_hat_rel_err, mean_nCy_hat_rel_err = np.mean(nCx_hat_rel_err), np.mean(nCy_hat_rel_err)
                mean_vx_hat_rel_err, mean_vy_hat_rel_err = np.mean(vx_hat_rel_err), np.mean(vy_hat_rel_err)

                '''计算均方误差'''
                # 均方误差（观测值和真值）
                nCx_mea_tru_mse, nCy_mea_tru_mse = mean_squared_error(nCx_mea, nCx_tru), mean_squared_error(nCy_mea, nCy_tru)
                vx_mea_tru_mse, vy_mea_tru_mse = mean_squared_error(vx_mea, vx_tru), mean_squared_error(vy_mea, vy_tru)
                # 均方误差（估计值和真值）
                nCx_hat_tru_mse, nCy_hat_tru_mse = mean_squared_error(nCx_hat, nCx_tru), mean_squared_error(nCy_hat, nCy_tru)
                vx_hat_tru_mse, vy_hat_tru_mse = mean_squared_error(vx_hat, vx_tru), mean_squared_error(vy_hat, vy_tru)

                mse_mea_tru_id = np.mean([nCx_mea_tru_mse, nCy_mea_tru_mse, vx_mea_tru_mse, vy_mea_tru_mse])
                mse_hat_tru_id = np.mean([nCx_hat_tru_mse, nCy_hat_tru_mse, vx_hat_tru_mse, vy_hat_tru_mse])

                mse_mea_tru_group.append(mse_mea_tru_id)
                mse_hat_tru_group.append(mse_hat_tru_id)
            # print(np.mean(mse_mea_tru_group))

            # print(mse_mea_tru_avg)
        #         '''kalman滤波状态估计'''
        #         # kalman滤波状态估计
                plt.figure(figsize=(16, 10))
                plt.subplot(2, 2, 1)
                plt.title('ID' + str(test_group_id[i4][i11]) + 'nestCenter.x state estimate(normal)', fontsize=16)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.plot(nCx_tru, 'royalblue', label='True value', marker='o')
                plt.plot(nCx_mea, 'orange', label='Measured value', marker='s')
                plt.plot(nCx_pre, 'darkgrey', label='Predicted value', marker='*')
                plt.plot(nCx_hat, 'red', label='Estimated value', marker='^')
                plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
                plt.ylabel('Distance(m)', fontsize=12, loc='top')
                plt.legend(fontsize=10)

                plt.subplot(2, 2, 2)
                plt.title('ID' + str(test_group_id[i4][i11]) + 'nestCenter.y state estimate(normal)', fontsize=16)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.plot(nCy_tru, 'royalblue', label='True value', marker='o')
                plt.plot(nCy_mea, 'orange', label='Measured value', marker='s')
                plt.plot(nCy_pre, 'darkgrey', label='Predicted value', marker='*')
                plt.plot(nCy_hat, 'red', label='Estimated value', marker='^')
                plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
                plt.ylabel('Distance(m)', fontsize=12, loc='top')
                plt.legend(fontsize=10)

                plt.subplot(2, 2, 3)
                plt.title('ID' + str(test_group_id[i4][i11]) + 'velocity.x state estimate(normal)', fontsize=16)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.plot(vx_tru, 'royalblue', label='True value', marker='o')
                plt.plot(vx_mea, 'orange', label='Measured value', marker='s')
                plt.plot(vx_pre, 'darkgrey', label='Predicted value', marker='*')
                plt.plot(vx_hat, 'red', label='Estimated value', marker='^')
                plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
                plt.ylabel('Velocity(m/s)', fontsize=12, loc='top')
                plt.legend(fontsize=10)

                plt.subplot(2, 2, 4)
                plt.title('ID' + str(test_group_id[i4][i11]) + 'velocity.y state estimate(normal)', fontsize=16)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.plot(vy_tru, 'royalblue', label='True value', marker='o')
                plt.plot(vy_mea, 'orange', label='Measured value', marker='s')
                plt.plot(vy_pre, 'darkgrey', label='Predicted value', marker='*')
                plt.plot(vy_hat, 'red', label='Estimated value', marker='^')
                plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
                plt.ylabel('Velocity(m/s)', fontsize=12, loc='top')
                plt.legend(fontsize=10)
                plt.savefig(r'../General_Kalman_Filtering_State_Test/test' + str(
                    i4 + 1) + '/'+mode+'/hat/ID' + str(test_group_id[i4][i11]) + '_hat.svg', dpi=500, bbox_inches='tight')
                plt.close()
                plt.show()
                # 观测值、真值、预测值、估计值xlsx
                data_tru_mea_pre_hat = pd.DataFrame({'stamp_sec': np.array(dt[i11].iloc[:, 0]), 'frame_num': np.array(dt[i11].iloc[:, 1]),
                                       'track_id_tru': np.array(dt[i11].iloc[:, 2]),
                                       'track_id_mea': np.array(dt[i11].iloc[:, 3]),
                                       'nestCenter_x_tru': nCx_tru, 'nestCenter_x_mea': nCx_mea,
                                       'nestCenter_x_pre': nCx_pre, 'nestCenter_x_hat': nCx_hat,
                                       'nestCenter_y_tru': nCy_tru, 'nestCenter_y_mea': nCy_mea,
                                       'nestCenter_y_pre': nCy_pre, 'nestCenter_y_hat': nCy_hat,
                                       'velocity_x_tru': vx_tru, 'velocity_x_mea': vx_mea,
                                       'velocity_x_pre': vx_pre, 'velocity_x_hat': vx_hat,
                                       'velocity_y_tru': vy_tru, 'velocity_y_mea': vy_mea,
                                       'velocity_y_pre': vy_pre, 'velocity_y_hat': vy_hat})
                data_tru_mea_pre_hat.to_excel(work_tru_mea_pre_hat, sheet_name='ID' + str(test_group_id[i4][i11]), index=False)
                # 观测均方误差、预测均方误差、估计均方误差
                data_mea_hat_mse = pd.DataFrame({'nCx_mea_tru_mse':nCx_mea_tru_mse, 'nCx_hat_tru_mse':nCx_hat_tru_mse,
                                                     'nCy_mea_tru_mse':nCy_mea_tru_mse, 'nCy_hat_tru_mse':nCy_hat_tru_mse,
                                                     'vx_mea_tru_mse':vx_mea_tru_mse, 'vx_hat_tru_mse':vx_hat_tru_mse,
                                                     'vy_mea_tru_mse': vy_mea_tru_mse, 'vy_hat_tru_mse': vy_hat_tru_mse}, index=[0])
                data_mea_hat_mse.to_excel(work_mea_hat_mse, sheet_name='ID' + str(test_group_id[i4][i11]), index=False)

                '''误差对比'''
                # 观测误差、预测误差、估计误差对比
                plt.figure(figsize=(16, 10))
                plt.subplot(2, 2, 1)
                plt.title('ID' + str(test_group_id[i4][i11]) + 'nestCenter.x error comparison', fontsize=16)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.plot(nCx_mea_err, 'r', label='Measured error', marker='v')
                plt.plot(nCx_hat_err, 'b', label='Estimated error', marker='v')
                plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
                plt.ylabel('Distance(m)', fontsize=12, loc='top')
                plt.legend(fontsize=10)

                plt.subplot(2, 2, 2)
                plt.title('ID' + str(test_group_id[i4][i11]) + 'nestCenter.y error comparison', fontsize=16)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.plot(nCy_mea_err, 'r', label='Measured error', marker='v')
                plt.plot(nCy_hat_err, 'b', label='Estimated error', marker='v')
                plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
                plt.ylabel('Distance(m)', fontsize=12, loc='top')
                plt.legend(fontsize=10)

                plt.subplot(2, 2, 3)
                plt.title('ID' + str(test_group_id[i4][i11]) + 'velocity.x error comparison', fontsize=16)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.plot(vx_mea_err, 'r', label='Measured error', marker='v')
                plt.plot(vx_hat_err, 'b', label='Estimated error', marker='v')
                plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
                plt.ylabel('Velocity(m/s)', fontsize=12, loc='top')
                plt.legend(fontsize=10)

                plt.subplot(2, 2, 4)
                plt.title('ID' + str(test_group_id[i4][i11]) + 'velocity.y error comparison', fontsize=16)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.plot(vy_mea_err, 'r', label='Measured error', marker='v')
                plt.plot(vy_hat_err, 'b', label='Estimated error', marker='v')
                plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
                plt.ylabel('Velocity(m/s)', fontsize=12, loc='top')
                plt.legend(fontsize=10)
                plt.savefig(r'../General_Kalman_Filtering_State_Test/test' + str(i4 + 1) + '/'+mode+'/mea_err/ID' + str(test_group_id[i4][i11]) + '_mea_err.svg', dpi=500, bbox_inches='tight')
                plt.close()
                plt.show()

                #观测误差、估计误差xlsx
                data_mea_pre_hat_err = pd.DataFrame(
                    {'nCx_mea_err': nCx_mea_err, 'nCx_hat_err': nCx_hat_err,
                     'nCy_mea_err': nCy_mea_err, 'nCy_hat_err': nCy_hat_err,
                     'vx_mea_err': vx_mea_err, 'vx_hat_err': vx_hat_err,
                     'vy_mea_err': vy_mea_err, 'vy_hat_err': vy_hat_err, })
                data_mea_pre_hat_err.to_excel(work_mea_hat_err, sheet_name='ID' + str(test_group_id[i4][i11]), index=False)
                # 观测误差均值、预测误差均值、估计误差均值xlsx
                data_mean_mea_pre_hat_err = pd.DataFrame(
                    {'mean_nCx_mea_err': mean_nCx_mea_err, 'mean_nCx_hat_err': mean_nCx_hat_err,
                     'mean_nCy_mea_err': mean_nCy_mea_err, 'mean_nCy_hat_err': mean_nCy_hat_err,
                     'mean_vx_mea_err': mean_vx_mea_err, 'mean_vx_hat_err': mean_vx_hat_err,
                     'mean_vy_mea_err': mean_vy_mea_err, 'mean_vy_hat_err': mean_vy_hat_err, }, index=[0])
                data_mean_mea_pre_hat_err.to_excel(work_mean_mea_hat_err, sheet_name='ID' + str(test_group_id[i4][i11]),index=False)

                plt.figure(figsize=(16, 10))
                plt.subplot(2, 2, 1)
                plt.title('ID' + str(test_group_id[i4][i11]) + 'nestCenter.x rel-err comparison', fontsize=16)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.plot(nCx_mea_rel_err * 100, 'r', label='rel-err(measured and true)', marker='v')
                plt.plot(nCx_hat_rel_err * 100, 'y', label='rel-err(estimated and true)', marker='v')
                plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
                plt.ylabel('Rel-err(%)', fontsize=12, loc='top')
                plt.legend(fontsize=10)

                plt.subplot(2, 2, 2)
                plt.title('ID' + str(test_group_id[i4][i11]) + 'nestCenter.y rel-err comparison', fontsize=16)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.plot(nCy_mea_rel_err * 100, 'r', label='rel-err(measured and true)', marker='v')
                plt.plot(nCy_hat_rel_err * 100, 'y', label='rel-err(estimated and true)', marker='v')
                plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
                plt.ylabel('Rel-err(%)', fontsize=12, loc='top')
                plt.legend(fontsize=10)

                plt.subplot(2, 2, 3)
                plt.title('ID' + str(test_group_id[i4][i11]) + 'velocity.x rel-err comparison', fontsize=16)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.plot(vx_mea_rel_err * 100, 'r', label='rel-err(measured and true)', marker='v')
                plt.plot(vx_hat_rel_err * 100, 'y', label='rel-err(estimated and true)', marker='v')
                plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
                plt.ylabel('Rel-err(%)', fontsize=12, loc='top')
                plt.legend(fontsize=10)

                plt.subplot(2, 2, 4)
                plt.title('ID' + str(test_group_id[i4][i11]) + 'velocity.y rel-err comparison', fontsize=16)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.plot(vy_mea_rel_err * 100, 'r', label='rel-err(measured and true)', marker='v')
                plt.plot(vy_hat_rel_err * 100, 'y', label='rel-err(estimated and true)', marker='v')
                plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
                plt.ylabel('Rel-err(%)', fontsize=12, loc='top')
                plt.legend(fontsize=10)
                plt.savefig(r'../General_Kalman_Filtering_State_Test/test' + str(i4 + 1) + '/'+mode+'/rel_err/ID' + str(test_group_id[i4][i11]) + '_rel_err.svg', dpi=500, bbox_inches='tight')
                plt.close()
                plt.show()
                # 观测相对误差、预测相对误差、估计相对误差
                data_mea_pre_hat_rel_err = pd.DataFrame(
                    {'nCx_mea_rel_err': nCx_mea_rel_err, 'nCx_hat_rel_err': nCx_hat_rel_err,
                     'nCy_mea_rel_err': nCy_mea_rel_err, 'nCy_hat_rel_err': nCy_hat_rel_err,
                     'vx_mea_rel_err': vx_mea_rel_err, 'vx_hat_rel_err': vx_hat_rel_err,
                     'vy_mea_rel_err': vy_mea_rel_err, 'vy_hat_rel_err': vy_hat_rel_err})
                data_mea_pre_hat_rel_err.to_excel(work_mea_hat_rel_err, sheet_name='ID' + str(test_group_id[i4][i11]), index=False)
                # 观测相对误差均值，预测相对误差均值对比xlsx
                data_mean_mea_pre_hat_rel_err = pd.DataFrame(
                    {'mean_nCx_mea_rel_err': mean_nCx_mea_rel_err, 'mean_nCx_hat_rel_err': mean_nCx_hat_rel_err,
                     'mean_nCy_mea_rel_err': mean_nCy_mea_rel_err, 'mean_nCy_hat_rel_err': mean_nCy_hat_rel_err,
                     'mean_vx_mea_rel_err': mean_vx_mea_rel_err, 'mean_vx_hat_rel_err': mean_vx_hat_rel_err,
                     'mean_vy_mea_rel_err': mean_vy_mea_rel_err, 'mean_vy_hat_rel_err': mean_vy_hat_rel_err}, index=[0])
                data_mean_mea_pre_hat_rel_err.to_excel(work_mean_mea_hat_rel_err, sheet_name='ID' + str(test_group_id[i4][i11]), index=False)

            work_tru_mea_pre_hat.save()
            work_mea_hat_mse.save()
            work_mea_hat_err.save()
            work_mean_mea_hat_err.save()
            work_mea_hat_rel_err.save()
            work_mean_mea_hat_rel_err.save()

            print('{}情况下, 第' + str(i4 + 1) + '组样本数据下的Kalman Filtering状态估计测试完成'.format(mode))
            print('-------------------')

            mse_mea_tru_avg.append(np.mean(mse_mea_tru_group))
            mse_hat_tru_avg.append(np.mean(mse_hat_tru_group))

        print('{}情况下，20组样本数据观测值-真值的均方误差均值为：{}'.format(mode, np.mean(mse_mea_tru_avg)))
        print('{}情况下，20组样本数据估计值-真值的均方误差均值为：{}'.format(mode, np.mean(mse_hat_tru_avg)))

        mse_avg_whole = pd.ExcelWriter('../General_Kalman_Filtering_State_Test/mse_avg_whole_'+mode+'.xlsx')
        mse_avg = pd.DataFrame({'mse_mea_tru_avg': np.mean(mse_mea_tru_avg),
                                'mse_hat_tru_avg': np.mean(mse_hat_tru_avg)}, index=[0])
        mse_avg.to_excel(mse_avg_whole, index=False)
        mse_avg_whole.save()

        end = timeit.default_timer()
        print('Running time: %s Seconds' % (end - start))
        print('测试结束')

general_mean_init(str1)
