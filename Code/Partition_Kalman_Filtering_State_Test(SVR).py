import timeit

start = timeit.default_timer()
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kalmanfilter import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 读取关联数据CSV文件
data_match = pd.read_csv('../Data/match(数据关联).csv', encoding='gbk')
print(data_match)
df_match = pd.DataFrame(data_match, columns=['frame_num', 'lidar_track_id', 'dt_track_id'], dtype=int)
# 关联数据的帧数、真值ID、传感器ID
frame_match = list(df_match.iloc[:, 0])
tru_id_match = list(df_match.iloc[:, 1])
lidar_id_match = list(df_match.iloc[:, 2])
# 设定20组样本数据进行测试
test_groups = int(1000 / 50)

# 创建test1-test20的20个文件夹
path1 = '../Partition_Kalman_Filtering_State_Test(SVR)/'

name = 'test'
str1 = 'hat', 'mea_err', 'rel_err'
for i in range(0, test_groups):
    sExists = os.path.exists(path1 + name + str(i + 1))
    os.makedirs(path1 + name + str(i + 1))
    print("%s 目录创建成功" % (name + str(i + 1)))
    path2 = path1 + name + str(i + 1) + '/'
    for i_1 in str1:
        sExists1 = os.path.exists(path2 + i_1)
        os.makedirs(path2 + i_1)
        print("%s 目录创建成功" % (i_1))

print('文件夹创建完成')
print('-------------------')
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

# SVR拟合真值数据模块，RBF高斯核函数，网格搜索C惩罚系数和高斯核函数系数gamma
c_can = np.logspace(-2, 2, num=10, base=10)
gamma_can = np.logspace(-2, 2, num=10, base=10)
SVR1 = GridSearchCV(SVR(kernel='rbf'), param_grid={'C': c_can, 'gamma': gamma_can}, cv=5)
SVR2 = GridSearchCV(SVR(kernel='rbf'), param_grid={'C': c_can, 'gamma': gamma_can}, cv=5)
SVR3 = GridSearchCV(SVR(kernel='rbf'), param_grid={'C': c_can, 'gamma': gamma_can}, cv=5)
SVR4 = GridSearchCV(SVR(kernel='rbf'), param_grid={'C': c_can, 'gamma': gamma_can}, cv=5)

for i4 in range(0, len(test_group_id)):
    obj_num = len(test_group_id[i4])
    dt = []
    id1, id2, id3, id4 = [], [], [], []
    tru_nCx_all1, tru_nCy_all1, tru_vx_all1, tru_vy_all1 = [], [], [], []
    tru_nCx_all2, tru_nCy_all2, tru_vx_all2, tru_vy_all2 = [], [], [], []
    tru_nCx_all3, tru_nCy_all3, tru_vx_all3, tru_vy_all3 = [], [], [], []
    tru_nCx_all4, tru_nCy_all4, tru_vx_all4, tru_vy_all4 = [], [], [], []

    mea_nCx_all1, mea_nCy_all1, mea_vx_all1, mea_vy_all1 = [], [], [], []
    mea_nCx_all2, mea_nCy_all2, mea_vx_all2, mea_vy_all2 = [], [], [], []
    mea_nCx_all3, mea_nCy_all3, mea_vx_all3, mea_vy_all3 = [], [], [], []
    mea_nCx_all4, mea_nCy_all4, mea_vx_all4, mea_vy_all4 = [], [], [], []

    mea_nCx_var1, mea_nCy_var1, mea_vx_var1, mea_vy_var1 = [], [], [], []
    mea_nCx_var2, mea_nCy_var2, mea_vx_var2, mea_vy_var2 = [], [], [], []
    mea_nCx_var3, mea_nCy_var3, mea_vx_var3, mea_vy_var3 = [], [], [], []
    mea_nCx_var4, mea_nCy_var4, mea_vx_var4, mea_vy_var4 = [], [], [], []

    for id in test_group_id[i4]:
        obj_data = pd.read_csv(
            '../Objects_Data/ID' + str(id) + '/ID' + str(id) + '_data.csv',
            encoding='gbk')
        df_obj = obj_data.iloc[:, 1]  # 目标数据的所有的帧数
        n_obj = len(df_obj)


        def frame_num():
            f_n = []
            for i5 in range(i4 * 50, i4 * 50 + 50):
                for ele2 in np.array(df_obj):
                    if ele2 == i5:
                        f_n.append(ele2)
            return f_n


        f_n = frame_num()


        def frame_index():
            index_obj = []
            for i6 in range(0, 50):
                for i7 in range(0, n_obj):
                    if df_obj[i7] == f_n[i6]:
                        index_obj.append(i7)
            return index_obj


        index_obj = frame_index()

        dt_obj = obj_data.iloc[index_obj[0]:index_obj[-1] + 1]
        dt.append(dt_obj)

    # SVR进行每组的多个目标观测值进行拟合真值
    def true_fit():
        Ymea = []
        Yfit = []
        for data in dt:
            df1 = pd.DataFrame(data, columns=['stamp_sec', 'frame_num', 'track_id_tru', 'track_id_mea',
                                              'nestCenter_x_tru', 'nestCenter_x_mea', 'mea_error_x', 'mea_var_x',
                                              'nestCenter_y_tru', 'nestCenter_y_mea', 'mea_error_y', 'mea_var_y',
                                              'velocity_x_tru', 'velocity_x_mea', 'mea_error_vx', 'mea_var_vx',
                                              'velocity_y_tru', 'velocity_y_mea', 'mea_error_vy', 'mea_var_vy'],
                               dtype=float)

            X = df1.iloc[:, 1]
            Y1 = df1.iloc[:, 5]  # nestCenter.x观测值
            Y2 = df1.iloc[:, 9]  # nestCenter.y观测值
            Y3 = df1.iloc[:, 13]  # velocity.x观测值
            Y4 = df1.iloc[:, 17]  # velocity.y观测值
            # 归一化
            sc_x = StandardScaler()
            sc_y1 = StandardScaler()
            sc_y2 = StandardScaler()
            sc_y3 = StandardScaler()
            sc_y4 = StandardScaler()
            X = sc_x.fit_transform(np.array(X).reshape(-1, 1))
            Y1 = sc_y1.fit_transform(np.array(Y1).reshape(-1, 1))
            Y2 = sc_y2.fit_transform(np.array(Y2).reshape(-1, 1))
            Y3 = sc_y3.fit_transform(np.array(Y3).reshape(-1, 1))
            Y4 = sc_y4.fit_transform(np.array(Y4).reshape(-1, 1))
            # 进行回归训练
            SVR1.fit(X, Y1)
            SVR2.fit(X, Y2)
            SVR3.fit(X, Y3)
            SVR4.fit(X, Y4)
            # 进行拟合预测
            y_fit1 = SVR1.predict(X)
            y_fit2 = SVR2.predict(X)
            y_fit3 = SVR3.predict(X)
            y_fit4 = SVR4.predict(X)
            # 反归一化
            Y1 = sc_y1.inverse_transform(Y1.reshape(-1, 1))
            y_fit1 = sc_y1.inverse_transform(y_fit1.reshape(-1, 1))

            Y2 = sc_y2.inverse_transform(Y2.reshape(-1, 1))
            y_fit2 = sc_y2.inverse_transform(y_fit2.reshape(-1, 1))

            Y3 = sc_y3.inverse_transform(Y3.reshape(-1, 1))
            y_fit3 = sc_y3.inverse_transform(y_fit3.reshape(-1, 1))

            Y4 = sc_y4.inverse_transform(Y4.reshape(-1, 1))
            y_fit4 = sc_y4.inverse_transform(y_fit4.reshape(-1, 1))

            Ymea.append([Y1, Y2, Y3, Y4])
            # y_fit1为nestCenter.x拟合真值， y_fit2为nestCenter.y拟合真值，y_fit3为velocity.x拟合真值， y_fit4为velocity.y拟合真值
            Yfit.append([y_fit1, y_fit2, y_fit3, y_fit4])

        return Ymea, Yfit


    Ymea, Yfit = true_fit()

    mea_nCx, mea_nCy, mea_vx, mea_vy = [], [], [], []
    tru_nCx, tru_nCy, tru_vx, tru_vy = [], [], [], []
    mea_nCx_err, mea_nCy_err, mea_vx_err, mea_vy_err = [], [], [], []

    for i8 in range(0, 50):
        nCx_m, nCy_m, vx_m, vy_m = [], [], [], []
        nCx_t, nCy_t, vx_t, vy_t = [], [], [], []
        nCx_m_err, nCy_m_err, vx_m_err, vy_m_err = [], [], [], []
        for i9 in range(0, obj_num):
            # 获取观测值
            nCx_m.append(np.concatenate(Ymea[i9][0])[i8])
            nCy_m.append(np.concatenate(Ymea[i9][1])[i8])
            vx_m.append(np.concatenate(Ymea[i9][2])[i8])
            vy_m.append(np.concatenate(Ymea[i9][3])[i8])
            # 获取拟合真值
            nCx_t.append(np.concatenate(Yfit[i9][0])[i8])
            nCy_t.append(np.concatenate(Yfit[i9][1])[i8])
            vx_t.append(np.concatenate(Yfit[i9][2])[i8])
            vy_t.append(np.concatenate(Yfit[i9][3])[i8])
            # 获取观测误差
            nCx_m_err.append(np.concatenate(Ymea[i9][0])[i8] - np.concatenate(Yfit[i9][0])[i8])
            nCy_m_err.append(np.concatenate(Ymea[i9][1])[i8] - np.concatenate(Yfit[i9][1])[i8])
            vx_m_err.append(np.concatenate(Ymea[i9][2])[i8] - np.concatenate(Yfit[i9][2])[i8])
            vy_m_err.append(np.concatenate(Ymea[i9][3])[i8] - np.concatenate(Yfit[i9][3])[i8])
        # 将观测值添加到list
        mea_nCx.append(nCx_m)
        mea_nCy.append(nCy_m)
        mea_vx.append(vx_m)
        mea_vy.append(vy_m)
        # 将拟合真值添加到list
        tru_nCx.append(nCx_t)
        tru_nCy.append(nCy_t)
        tru_vx.append(vx_t)
        tru_vy.append(vy_t)
        # 将观测误差添加到list
        mea_nCx_err.append(nCx_m_err)
        mea_nCy_err.append(nCy_m_err)
        mea_vx_err.append(vx_m_err)
        mea_vy_err.append(vy_m_err)
    # 将观测误差list转为数组，便于读取
    mea_nCx_err = np.array(mea_nCx_err)
    mea_nCy_err = np.array(mea_nCy_err)
    mea_vx_err = np.array(mea_vx_err)
    mea_vy_err = np.array(mea_vy_err)

    # 定义观测误差方差函数
    def mea_err_variance():
        mea_nCx_var, mea_nCy_var, mea_vx_var, mea_vy_var = [], [], [], []
        for i10 in range(0, obj_num):
            nCx_err_var, nCy_err_var, vx_err_var, vy_err_var = [], [], [], []
            for i11 in range(0, 50):
                nCx_err_var.append(np.var(mea_nCx_err[:i11 + 1, i10]))
                nCy_err_var.append(np.var(mea_nCy_err[:i11 + 1, i10]))
                vx_err_var.append(np.var(mea_vx_err[:i11 + 1, i10]))
                vy_err_var.append(np.var(mea_vy_err[:i11 + 1, i10]))
            mea_nCx_var.append(nCx_err_var)
            mea_nCy_var.append(nCy_err_var)
            mea_vx_var.append(vx_err_var)
            mea_vy_var.append(vy_err_var)
        return mea_nCx_var, mea_nCy_var, mea_vx_var, mea_vy_var


    mea_nCx_var, mea_nCy_var, mea_vx_var, mea_vy_var = mea_err_variance()


    for i10 in range(0, obj_num):
        mea_ncx = list(np.array(mea_nCx)[:, i10])
        mea_ncy = list(np.array(mea_nCy)[:, i10])
        mea_vxx = list(np.array(mea_vx)[:, i10])
        mea_vyy = list(np.array(mea_vy)[:, i10])

        tru_ncx = list(np.array(tru_nCx)[:, i10])
        tru_ncy = list(np.array(tru_nCy)[:, i10])
        tru_vxx = list(np.array(tru_vx)[:, i10])
        tru_vyy = list(np.array(tru_vy)[:, i10])

        mea_ncx_var = mea_nCx_var[i10]
        mea_ncy_var = mea_nCy_var[i10]
        mea_vxx_var = mea_vx_var[i10]
        mea_vyy_var = mea_vy_var[i10]

        a = list(filter(lambda x: x >= 80, mea_ncx))
        b = list(filter(lambda x: x >= 0, mea_ncy))
        len_a = len(a)
        len_b = len(b)

        if len_a >= 25 and len_b >= 25:
            id1.append(test_group_id[i4][i10])
            tru_nCx_all1.append(tru_ncx)
            tru_nCy_all1.append(tru_ncy)
            tru_vx_all1.append(tru_vxx)
            tru_vy_all1.append(tru_vyy)

            mea_nCx_all1.append(mea_ncx)
            mea_nCy_all1.append(mea_ncy)
            mea_vx_all1.append(mea_vxx)
            mea_vy_all1.append(mea_vyy)

            mea_nCx_var1.append(mea_ncx_var)
            mea_nCy_var1.append(mea_ncy_var)
            mea_vx_var1.append(mea_vxx_var)
            mea_vy_var1.append(mea_vyy_var)
        elif len_a >= 25 and len_b < 25:
            id2.append(test_group_id[i4][i10])
            tru_nCx_all2.append(tru_ncx)
            tru_nCy_all2.append(tru_ncy)
            tru_vx_all2.append(tru_vxx)
            tru_vy_all2.append(tru_vyy)

            mea_nCx_all2.append(mea_ncx)
            mea_nCy_all2.append(mea_ncy)
            mea_vx_all2.append(mea_vxx)
            mea_vy_all2.append(mea_vyy)

            mea_nCx_var2.append(mea_ncx_var)
            mea_nCy_var2.append(mea_ncy_var)
            mea_vx_var2.append(mea_vxx_var)
            mea_vy_var2.append(mea_vyy_var)
        elif len_a < 25 and len_b >= 25:
            id3.append(test_group_id[i4][i10])
            tru_nCx_all3.append(tru_ncx)
            tru_nCy_all3.append(tru_ncy)
            tru_vx_all3.append(tru_vxx)
            tru_vy_all3.append(tru_vyy)

            mea_nCx_all3.append(mea_ncx)
            mea_nCy_all3.append(mea_ncy)
            mea_vx_all3.append(mea_vxx)
            mea_vy_all3.append(mea_vyy)

            mea_nCx_var3.append(mea_ncx_var)
            mea_nCy_var3.append(mea_ncy_var)
            mea_vx_var3.append(mea_vxx_var)
            mea_vy_var3.append(mea_vyy_var)
        else:
            id4.append(test_group_id[i4][i10])
            tru_nCx_all4.append(tru_ncx)
            tru_nCy_all4.append(tru_ncy)
            tru_vx_all4.append(tru_vxx)
            tru_vy_all4.append(tru_vyy)

            mea_nCx_all4.append(mea_ncx)
            mea_nCy_all4.append(mea_ncy)
            mea_vx_all4.append(mea_vxx)
            mea_vy_all4.append(mea_vyy)

            mea_nCx_var4.append(mea_ncx_var)
            mea_nCy_var4.append(mea_ncy_var)
            mea_vx_var4.append(mea_vxx_var)
            mea_vy_var4.append(mea_vyy_var)

    mean_nCx_var1 = np.mean(mea_nCx_var1)
    mean_nCy_var1 = np.mean(mea_nCy_var1)
    mean_vx_var1 = np.mean(mea_vx_var1)
    mean_vy_var1 = np.mean(mea_vy_var1)

    mean_nCx_var2 = np.mean(mea_nCx_var2)
    mean_nCy_var2 = np.mean(mea_nCy_var2)
    mean_vx_var2 = np.mean(mea_vx_var2)
    mean_vy_var2 = np.mean(mea_vy_var2)

    mean_nCx_var3 = np.mean(mea_nCx_var3)
    mean_nCy_var3 = np.mean(mea_nCy_var3)
    mean_vx_var3 = np.mean(mea_vx_var3)
    mean_vy_var3 = np.mean(mea_vy_var3)

    mean_nCx_var4 = np.mean(mea_nCx_var4)
    mean_nCy_var4 = np.mean(mea_nCy_var4)
    mean_vx_var4 = np.mean(mea_vx_var4)
    mean_vy_var4 = np.mean(mea_vy_var4)

    tru_nCx1, tru_nCy1, tru_vx1, tru_vy1 = [], [], [], []
    tru_nCx2, tru_nCy2, tru_vx2, tru_vy2 = [], [], [], []
    tru_nCx3, tru_nCy3, tru_vx3, tru_vy3 = [], [], [], []
    tru_nCx4, tru_nCy4, tru_vx4, tru_vy4 = [], [], [], []

    tru_nCx1, tru_nCy1, tru_vx1, tru_vy1 = tru_value_process(tru_nCx_all1, tru_nCy_all1, tru_vx_all1, tru_vy_all1,
                                                             tru_nCx1, tru_nCy1, tru_vx1, tru_vy1)
    tru_nCx2, tru_nCy2, tru_vx2, tru_vy2 = tru_value_process(tru_nCx_all2, tru_nCy_all2, tru_vx_all2, tru_vy_all2,
                                                             tru_nCx2, tru_nCy2, tru_vx2, tru_vy2)
    tru_nCx3, tru_nCy3, tru_vx3, tru_vy3 = tru_value_process(tru_nCx_all3, tru_nCy_all3, tru_vx_all3, tru_vy_all3,
                                                             tru_nCx3, tru_nCy3, tru_vx3, tru_vy3)
    tru_nCx4, tru_nCy4, tru_vx4, tru_vy4 = tru_value_process(tru_nCx_all4, tru_nCy_all4, tru_vx_all4, tru_vy_all4,
                                                             tru_nCx4, tru_nCy4, tru_vx4, tru_vy4)

    mea_nCx1, mea_nCy1, mea_vx1, mea_vy1 = [], [], [], []
    mea_nCx2, mea_nCy2, mea_vx2, mea_vy2 = [], [], [], []
    mea_nCx3, mea_nCy3, mea_vx3, mea_vy3 = [], [], [], []
    mea_nCx4, mea_nCy4, mea_vx4, mea_vy4 = [], [], [], []

    mea_nCx1, mea_nCy1, mea_vx1, mea_vy1 = mea_value_process(mea_nCx_all1, mea_nCy_all1, mea_vx_all1, mea_vy_all1,
                                                             mea_nCx1, mea_nCy1, mea_vx1, mea_vy1)
    mea_nCx2, mea_nCy2, mea_vx2, mea_vy2 = mea_value_process(mea_nCx_all2, mea_nCy_all2, mea_vx_all2, mea_vy_all2,
                                                             mea_nCx2, mea_nCy2, mea_vx2, mea_vy2)
    mea_nCx3, mea_nCy3, mea_vx3, mea_vy3 = mea_value_process(mea_nCx_all3, mea_nCy_all3, mea_vx_all3, mea_vy_all3,
                                                             mea_nCx3, mea_nCy3, mea_vx3, mea_vy3)
    mea_nCx4, mea_nCy4, mea_vx4, mea_vy4 = mea_value_process(mea_nCx_all4, mea_nCy_all4, mea_vx_all4, mea_vy_all4,
                                                             mea_nCx4, mea_nCy4, mea_vx4, mea_vy4)

    work_tru_mea_pre_hat = pd.ExcelWriter('../Partition_Kalman_Filtering_State_Test(SVR)/'+ 'test'
                                          + str(i4 + 1) + '/hat/tru_mea_pre_hat' + str(i4 + 1) + '.xlsx')
    work_mea_hat_mse = pd.ExcelWriter('../Partition_Kalman_Filtering_State_Test(SVR)/' + 'test'
                                      + str(i4 + 1) + '/hat/mea_hat_mse' + str(i4 + 1) + '.xlsx')

    work_mea_hat_err = pd.ExcelWriter('../Partition_Kalman_Filtering_State_Test(SVR)/' + 'test'
                                          + str(i4 + 1) + '/mea_err/mea_hat_err' + str(i4 + 1) + '.xlsx')
    work_mean_mea_hat_err = pd.ExcelWriter('../Partition_Kalman_Filtering_State_Test(SVR)/' + 'test'
                                      + str(i4 + 1) + '/mea_err/mean_mea_hat_err' + str(i4 + 1) + '.xlsx')

    work_mea_hat_rel_err = pd.ExcelWriter('../Partition_Kalman_Filtering_State_Test(SVR)/' + 'test'
                                          + str(i4 + 1) + '/rel_err/mea_hat_rel_err' + str(i4 + 1) + '.xlsx')
    work_mean_mea_hat_rel_err = pd.ExcelWriter('../Partition_Kalman_Filtering_State_Test(SVR)/' + 'test'
                                          + str(i4 + 1) + '/rel_err/mean_mea_hat_rel_err' + str(i4 + 1) + '.xlsx')


    Q1 = np.array([[mean_nCx_var1, 0, 0, 0],
                   [0, mean_nCy_var1, 0, 0],
                   [0, 0, mean_vx_var1, 0],
                   [0, 0, 0, mean_vy_var1]])
    R1 = np.array([[mean_nCx_var1, 0, 0, 0],
                   [0, mean_nCy_var1, 0, 0],
                   [0, 0, mean_vx_var1, 0],
                   [0, 0, 0, mean_vy_var1]])
    mea_X1, tru_X1, pre_X1, hat_X1 = [], [], [], []

    if np.isnan(Q1).any() == False:
        mea_X1, tru_X1, pre_X1, hat_X1 = kalmanfilter(Q1, R1, tru_nCx1, tru_nCy1, tru_vx1, tru_vy1, mea_nCx1, mea_nCy1,
                                                      mea_vx1, mea_vy1)
        mea_X1, tru_X1, pre_X1, hat_X1 = np.array(mea_X1), np.array(tru_X1), np.array(pre_X1), np.array(hat_X1)
        for i8 in range(0, len(id1)):
            nCx_tru, nCy_tru, vx_tru, vy_tru = tru_X1[:, :, i8][:, 0], tru_X1[:, :, i8][:, 1], tru_X1[:, :, i8][:, 2], tru_X1[:, :, i8][:, 3]
            nCx_mea, nCy_mea, vx_mea, vy_mea = mea_X1[:, :, i8][:, 0], mea_X1[:, :, i8][:, 1], mea_X1[:, :, i8][:, 2], mea_X1[:, :, i8][:, 3]
            nCx_pre, nCy_pre, vx_pre, vy_pre = pre_X1[:, :, i8][:, 0], pre_X1[:, :, i8][:, 1], pre_X1[:, :, i8][:, 2], pre_X1[:, :, i8][:, 3]
            nCx_hat, nCy_hat, vx_hat, vy_hat = hat_X1[:, :, i8][:, 0], hat_X1[:, :, i8][:, 1], hat_X1[:, :, i8][:, 2], hat_X1[:, :, i8][:, 3]

            '''kalman滤波状态估计'''
            # kalman滤波状态估计
            plt.figure(figsize=(16, 10))
            plt.subplot(2, 2, 1)
            plt.title('ID' + str(id1[i8]) + 'nestCenter.x state estimate(partition)', fontsize=16)
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
            plt.title('ID' + str(id1[i8]) + 'nestCenter.y state estimate(partition)', fontsize=16)
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
            plt.title('ID' + str(id1[i8]) + 'velocity.x state estimate(partition)', fontsize=16)
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
            plt.title('ID' + str(id1[i8]) + 'velocity.y state estimate(partition)', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.plot(vy_tru, 'royalblue', label='True value', marker='o')
            plt.plot(vy_mea, 'orange', label='Measured value', marker='s')
            plt.plot(vy_pre, 'darkgrey', label='Predicted value', marker='*')
            plt.plot(vy_hat, 'red', label='Estimated value', marker='^')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Velocity(m/s)', fontsize=12, loc='top')
            plt.legend(fontsize=10)
            plt.savefig(r'../Partition_Kalman_Filtering_State_Test(SVR)/' + 'test' + str(
                i4 + 1) + '/hat/ID' + str(id1[i8]) + '_hat.svg', dpi=500, bbox_inches='tight')
            plt.close()
            plt.show()

            # 观测值、真值、预测值、估计值xlsx
            data_tru_mea_pre_hat = pd.DataFrame({'nestCenter_x_tru': nCx_tru, 'nestCenter_x_mea': nCx_mea,
                                   'nestCenter_x_pre': nCx_pre, 'nestCenter_x_hat': nCx_hat,
                                   'nestCenter_y_tru': nCy_tru, 'nestCenter_y_mea': nCy_mea,
                                   'nestCenter_y_pre': nCy_pre, 'nestCenter_y_hat': nCy_hat,
                                   'velocity_x_tru': vx_tru, 'velocity_x_mea': vx_mea,
                                   'velocity_x_pre': vx_pre, 'velocity_x_hat': vx_hat,
                                   'velocity_y_tru': vy_tru, 'velocity_y_mea': vy_mea,
                                   'velocity_y_pre': vy_pre, 'velocity_y_hat': vy_hat})
            data_tru_mea_pre_hat.to_excel(work_tru_mea_pre_hat, sheet_name='ID' + str(id1[i8]), index=False)

            data_mea_hat_mse = pd.DataFrame({'nCx_mea_tru_mse':mean_squared_error(nCx_mea, nCx_tru), 'nCx_hat_tru_mse':mean_squared_error(nCx_hat, nCx_tru),
                                                 'nCy_mea_tru_mse':mean_squared_error(nCy_mea, nCy_tru), 'nCy_hat_tru_mse':mean_squared_error(nCy_hat, nCy_tru),
                                                 'vx_mea_tru_mse':mean_squared_error(vx_mea, vx_tru), 'vx_hat_tru_mse':mean_squared_error(vx_hat, vx_tru),
                                                 'vy_mea_tru_mse': mean_squared_error(vy_mea, vy_tru), 'vy_hat_tru_mse': mean_squared_error(vy_hat, vy_tru)}, index=[0])
            data_mea_hat_mse.to_excel(work_mea_hat_mse, sheet_name='ID' + str(id1[i8]), index=False)

            # 观测误差、预测误差、估计误差对比
            plt.figure(figsize=(16, 10))
            plt.subplot(2, 2, 1)
            plt.title('ID' + str(id1[i8]) + 'nestCenter.x error comparison', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.plot(nCx_mea-nCx_tru, 'r', label='Measured error', marker='v')
            plt.plot(nCx_hat-nCx_tru, 'b', label='Estimated error', marker='v')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Distance(m)', fontsize=12, loc='top')
            plt.legend(fontsize=10)

            plt.subplot(2, 2, 2)
            plt.title('ID' + str(id1[i8]) + 'nestCenter.y error comparison', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.plot(nCy_mea-nCy_tru, 'r', label='Measured error', marker='v')
            plt.plot(nCy_hat-nCy_tru, 'b', label='Estimated error', marker='v')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Distance(m)', fontsize=12, loc='top')
            plt.legend(fontsize=10)

            plt.subplot(2, 2, 3)
            plt.title('ID' + str(id1[i8]) + 'velocity.x error comparison', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.plot(vx_mea-vx_tru, 'r', label='Measured error', marker='v')
            plt.plot(vx_hat-vx_tru, 'b', label='Estimated error', marker='v')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Velocity(m/s)', fontsize=12, loc='top')
            plt.legend(fontsize=10)

            plt.subplot(2, 2, 4)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.title('ID' + str(id1[i8]) + 'velocity.y error comparison', fontsize=16)
            plt.plot(vy_mea-vy_tru, 'r', label='Measured error', marker='v')
            plt.plot(vy_hat-vy_tru, 'b', label='Estimated error', marker='v')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Velocity(m/s)', fontsize=12, loc='top')
            plt.legend(fontsize=10)
            plt.savefig(r'../Partition_Kalman_Filtering_State_Test(SVR)/' + 'test' + str(i4 + 1) + '/mea_err/ID' + str(id1[i8]) + '_mea_err.svg', dpi=500, bbox_inches='tight')
            plt.close()
            plt.show()

            #观测误差、估计误差xlsx
            data_mea_pre_hat_err = pd.DataFrame(
                {'nCx_mea_err': nCx_mea-nCx_tru, 'nCx_hat_err': nCx_hat-nCx_tru,
                 'nCy_mea_err': nCy_mea-nCy_tru, 'nCy_hat_err': nCy_hat-nCy_tru,
                 'vx_mea_err': vx_mea-vx_tru, 'vx_hat_err': vx_hat-vx_tru,
                 'vy_mea_err': vy_mea-vy_tru, 'vy_hat_err': vy_hat-vy_tru})
            data_mea_pre_hat_err.to_excel(work_mea_hat_err, sheet_name='ID' + str(id1[i8]), index=False)
            # 观测误差均值、预测误差均值、估计误差均值xlsx
            data_mean_mea_pre_hat_err = pd.DataFrame(
                {'mean_nCx_mea_err': np.mean(nCx_mea-nCx_tru), 'mean_nCx_hat_err':np.mean(nCx_hat-nCx_tru),
                 'mean_nCy_mea_err': np.mean(nCy_mea-nCy_tru), 'mean_nCy_hat_err': np.mean(nCy_hat-nCy_tru),
                 'mean_vx_mea_err': np.mean(vx_mea-vx_tru), 'mean_vx_hat_err': np.mean(vx_hat-vx_tru),
                 'mean_vy_mea_err': np.mean(vy_mea-vy_tru), 'mean_vy_hat_err': np.mean(vy_hat-vy_tru)}, index=[0])
            data_mean_mea_pre_hat_err.to_excel(work_mean_mea_hat_err, sheet_name='ID' + str(id1[i8]),index=False)

            plt.figure(figsize=(16, 10))
            plt.subplot(2, 2, 1)
            plt.title('ID' + str(id1[i8]) + 'nestCenter.x rel-err comparison', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.plot((nCx_mea-nCx_tru)/nCx_tru * 100, 'r', label='rel-err(measured and true)', marker='v')
            plt.plot((nCx_hat-nCx_tru)/nCx_tru * 100, 'y', label='rel-err(estimated and true)', marker='v')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Rel-err(%)', fontsize=12, loc='top')
            plt.legend(fontsize=10)

            plt.subplot(2, 2, 2)
            plt.title('ID' + str(id1[i8]) + 'nestCenter.y rel-err comparison', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.plot((nCy_mea-nCy_tru)/nCy_tru * 100, 'r', label='rel-err(measured and true)', marker='v')
            plt.plot((nCy_hat-nCy_tru)/nCy_tru * 100, 'y', label='rel-err(estimated and true)', marker='v')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Rel-err(%)', fontsize=12, loc='top')
            plt.legend(fontsize=10)

            plt.subplot(2, 2, 3)
            plt.title('ID' +str(id1[i8]) + 'velocity.x rel-err comparison', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.plot((vx_mea-vx_tru)/vx_tru * 100, 'r', label='rel-err(measured and true)', marker='v')
            plt.plot((vx_hat-vx_tru)/vx_tru * 100, 'y', label='rel-err(estimated and true)', marker='v')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Rel-err(%)', fontsize=12, loc='top')
            plt.legend(fontsize=10)

            plt.subplot(2, 2, 4)
            plt.title('ID' +str(id1[i8]) + 'velocity.y rel-err comparison', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.plot((vy_mea-vy_tru)/vy_tru * 100, 'r', label='rel-err(measured and true)', marker='v')
            plt.plot((vy_hat-vy_tru)/vy_tru * 100, 'y', label='rel-err(estimated and true)', marker='v')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Rel-err(%)', fontsize=12, loc='top')
            plt.legend(fontsize=10)
            plt.savefig(r'../Partition_Kalman_Filtering_State_Test(SVR)/' + 'test' + str(i4 + 1) + '/rel_err/ID' + str(id1[i8]) + '_rel_err.svg', dpi=500, bbox_inches='tight')
            plt.close()
            plt.show()
            # 观测相对误差、预测相对误差、估计相对误差
            data_mea_pre_hat_rel_err = pd.DataFrame(
                {'nCx_mea_rel_err': (nCx_mea-nCx_tru)/nCx_tru, 'nCx_hat_rel_err': (nCx_hat-nCx_tru)/nCx_tru,
                 'nCy_mea_rel_err': (nCy_mea-nCy_tru)/nCy_tru, 'nCy_hat_rel_err': (nCy_hat-nCy_tru)/nCy_tru,
                 'vx_mea_rel_err': (vx_mea-vx_tru)/vx_tru, 'vx_hat_rel_err': (vx_hat-vx_tru)/vx_tru,
                 'vy_mea_rel_err': (vy_mea-vy_tru)/vy_tru, 'vy_hat_rel_err':(vy_hat-vy_tru)/vy_tru})
            data_mea_pre_hat_rel_err.to_excel(work_mea_hat_rel_err, sheet_name='ID' + str(id1[i8]), index=False)
            # 观测相对误差均值，预测相对误差均值对比xlsx
            data_mean_mea_pre_hat_rel_err = pd.DataFrame(
                {'mean_nCx_mea_rel_err': np.mean((nCx_mea-nCx_tru)/nCx_tru), 'mean_nCx_hat_rel_err': np.mean((nCx_hat-nCx_tru)/nCx_tru),
                 'mean_nCy_mea_rel_err': np.mean((nCy_mea-nCy_tru)/nCy_tru), 'mean_nCy_hat_rel_err': np.mean((nCy_hat-nCy_tru)/nCy_tru),
                 'mean_vx_mea_rel_err': np.mean((vx_mea-vx_tru)/vx_tru), 'mean_vx_hat_rel_err': np.mean((vx_hat-vx_tru)/vx_tru),
                 'mean_vy_mea_rel_err': np.mean((vy_mea-vy_tru)/vy_tru), 'mean_vy_hat_rel_err': np.mean((vy_hat-vy_tru)/vy_tru)}, index=[0])
            data_mean_mea_pre_hat_rel_err.to_excel(work_mean_mea_hat_rel_err, sheet_name='ID' + str(id1[i8]), index=False)

    else:
        mea_X1 = np.nan_to_num(mea_X1)
        tru_X1 = np.nan_to_num(tru_X1)
        pre_X1 = np.nan_to_num(pre_X1)
        hat_X1 = np.nan_to_num(hat_X1)

    Q2 = np.array([[mean_nCx_var2, 0, 0, 0],
                   [0, mean_nCy_var2, 0, 0],
                   [0, 0, mean_vx_var2, 0],
                   [0, 0, 0, mean_vy_var2]])
    R2 = np.array([[mean_nCx_var2, 0, 0, 0],
                   [0, mean_nCy_var2, 0, 0],
                   [0, 0, mean_vx_var2, 0],
                   [0, 0, 0, mean_vy_var2]])
    mea_X2, tru_X2, pre_X2, hat_X2 = [], [], [], []
    if np.isnan(Q2).any() == False:
        mea_X2, tru_X2, pre_X2, hat_X2 = kalmanfilter(Q2, R2, tru_nCx2, tru_nCy2, tru_vx2, tru_vy2, mea_nCx2, mea_nCy2,
                                                      mea_vx2, mea_vy2)
        mea_X2, tru_X2, pre_X2, hat_X2 = np.array(mea_X2), np.array(tru_X2), np.array(pre_X2), np.array(hat_X2)
        for i8 in range(0, len(id2)):
            nCx_tru, nCy_tru, vx_tru, vy_tru = tru_X2[:, :, i8][:, 0], tru_X2[:, :, i8][:, 1], tru_X2[:, :, i8][:, 2], tru_X2[:, :, i8][:, 3]
            nCx_mea, nCy_mea, vx_mea, vy_mea = mea_X2[:, :, i8][:, 0], mea_X2[:, :, i8][:, 1], mea_X2[:, :, i8][:, 2], mea_X2[:, :, i8][:, 3]
            nCx_pre, nCy_pre, vx_pre, vy_pre = pre_X2[:, :, i8][:, 0], pre_X2[:, :, i8][:, 1], pre_X2[:, :, i8][:, 2], pre_X2[:, :, i8][:, 3]
            nCx_hat, nCy_hat, vx_hat, vy_hat = hat_X2[:, :, i8][:, 0], hat_X2[:, :, i8][:, 1], hat_X2[:, :, i8][:, 2], hat_X2[:, :, i8][:, 3]

            '''kalman滤波状态估计'''
            # kalman滤波状态估计
            plt.figure(figsize=(16, 10))
            plt.subplot(2, 2, 1)
            plt.title('ID' + str(id2[i8]) + 'nestCenter.x state estimate(partition)', fontsize=16)
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
            plt.title('ID' + str(id2[i8]) + 'nestCenter.y state estimate(partition)', fontsize=16)
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
            plt.title('ID' + str(id2[i8]) + 'velocity.x state estimate(partition)', fontsize=16)
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
            plt.title('ID' + str(id2[i8]) + 'velocity.y state estimate(partition)', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.plot(vy_tru, 'royalblue', label='True value', marker='o')
            plt.plot(vy_mea, 'orange', label='Measured value', marker='s')
            plt.plot(vy_pre, 'darkgrey', label='Predicted value', marker='*')
            plt.plot(vy_hat, 'red', label='Estimated value', marker='^')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Velocity(m/s)', fontsize=12, loc='top')
            plt.legend(fontsize=10)
            plt.savefig(r'../Partition_Kalman_Filtering_State_Test(SVR)/' + 'test' + str(
                i4 + 1) + '/hat/ID' + str(id2[i8]) + '_hat.svg', dpi=500, bbox_inches='tight')
            plt.close()
            plt.show()
            # 观测值、真值、预测值、估计值xlsx
            data_tru_mea_pre_hat = pd.DataFrame({'nestCenter_x_tru': nCx_tru, 'nestCenter_x_mea': nCx_mea,
                                                 'nestCenter_x_pre': nCx_pre, 'nestCenter_x_hat': nCx_hat,
                                                 'nestCenter_y_tru': nCy_tru, 'nestCenter_y_mea': nCy_mea,
                                                 'nestCenter_y_pre': nCy_pre, 'nestCenter_y_hat': nCy_hat,
                                                 'velocity_x_tru': vx_tru, 'velocity_x_mea': vx_mea,
                                                 'velocity_x_pre': vx_pre, 'velocity_x_hat': vx_hat,
                                                 'velocity_y_tru': vy_tru, 'velocity_y_mea': vy_mea,
                                                 'velocity_y_pre': vy_pre, 'velocity_y_hat': vy_hat})
            data_tru_mea_pre_hat.to_excel(work_tru_mea_pre_hat, sheet_name='ID' + str(id2[i8]), index=False)

            data_mea_hat_mse = pd.DataFrame({'nCx_mea_tru_mse': mean_squared_error(nCx_mea, nCx_tru),
                                             'nCx_hat_tru_mse': mean_squared_error(nCx_hat, nCx_tru),
                                             'nCy_mea_tru_mse': mean_squared_error(nCy_mea, nCy_tru),
                                             'nCy_hat_tru_mse': mean_squared_error(nCy_hat, nCy_tru),
                                             'vx_mea_tru_mse': mean_squared_error(vx_mea, vx_tru),
                                             'vx_hat_tru_mse': mean_squared_error(vx_hat, vx_tru),
                                             'vy_mea_tru_mse': mean_squared_error(vy_mea, vy_tru),
                                             'vy_hat_tru_mse': mean_squared_error(vy_hat, vy_tru)}, index=[0])
            data_mea_hat_mse.to_excel(work_mea_hat_mse, sheet_name='ID' + str(id2[i8]), index=False)

            # 观测误差、预测误差、估计误差对比
            plt.figure(figsize=(16, 10))
            plt.subplot(2, 2, 1)
            plt.title('ID' + str(id2[i8]) + 'nestCenter.x error comparison', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.plot(nCx_mea - nCx_tru, 'r', label='Measured error', marker='v')
            plt.plot(nCx_hat - nCx_tru, 'b', label='Estimated error', marker='v')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Distance(m)', fontsize=12, loc='top')
            plt.legend(fontsize=10)

            plt.subplot(2, 2, 2)
            plt.title('ID' + str(id2[i8]) + 'nestCenter.y error comparison', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.plot(nCy_mea - nCy_tru, 'r', label='Measured error', marker='v')
            plt.plot(nCy_hat - nCy_tru, 'b', label='Estimated error', marker='v')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Distance(m)', fontsize=12, loc='top')
            plt.legend(fontsize=10)

            plt.subplot(2, 2, 3)
            plt.title('ID' + str(id2[i8]) + 'velocity.x error comparison', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.plot(vx_mea - vx_tru, 'r', label='Measured error', marker='v')
            plt.plot(vx_hat - vx_tru, 'b', label='Estimated error', marker='v')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Velocity(m/s)', fontsize=12, loc='top')
            plt.legend(fontsize=10)

            plt.subplot(2, 2, 4)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.title('ID' + str(id2[i8]) + 'velocity.y error comparison', fontsize=16)
            plt.plot(vy_mea - vy_tru, 'r', label='Measured error', marker='v')
            plt.plot(vy_hat - vy_tru, 'b', label='Estimated error', marker='v')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Velocity(m/s)', fontsize=12, loc='top')
            plt.legend(fontsize=10)
            plt.savefig(path1 + 'test' + str(
                i4 + 1) + '/mea_err/ID' + str(id2[i8]) + '_mea_err.svg', dpi=500, bbox_inches='tight')
            plt.close()
            plt.show()

            # 观测误差、估计误差xlsx
            data_mea_pre_hat_err = pd.DataFrame(
                {'nCx_mea_err': nCx_mea - nCx_tru, 'nCx_hat_err': nCx_hat - nCx_tru,
                 'nCy_mea_err': nCy_mea - nCy_tru, 'nCy_hat_err': nCy_hat - nCy_tru,
                 'vx_mea_err': vx_mea - vx_tru, 'vx_hat_err': vx_hat - vx_tru,
                 'vy_mea_err': vy_mea - vy_tru, 'vy_hat_err': vy_hat - vy_tru})
            data_mea_pre_hat_err.to_excel(work_mea_hat_err, sheet_name='ID' + str(id2[i8]), index=False)
            # 观测误差均值、预测误差均值、估计误差均值xlsx
            data_mean_mea_pre_hat_err = pd.DataFrame(
                {'mean_nCx_mea_err': np.mean(nCx_mea - nCx_tru), 'mean_nCx_hat_err': np.mean(nCx_hat - nCx_tru),
                 'mean_nCy_mea_err': np.mean(nCy_mea - nCy_tru), 'mean_nCy_hat_err': np.mean(nCy_hat - nCy_tru),
                 'mean_vx_mea_err': np.mean(vx_mea - vx_tru), 'mean_vx_hat_err': np.mean(vx_hat - vx_tru),
                 'mean_vy_mea_err': np.mean(vy_mea - vy_tru), 'mean_vy_hat_err': np.mean(vy_hat - vy_tru)}, index=[0])
            data_mean_mea_pre_hat_err.to_excel(work_mean_mea_hat_err, sheet_name='ID' + str(id2[i8]), index=False)

            plt.figure(figsize=(16, 10))
            plt.subplot(2, 2, 1)
            plt.title('ID' + str(id2[i8]) + 'nestCenter.x rel-err comparison', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.plot((nCx_mea - nCx_tru) / nCx_tru * 100, 'r', label='rel-err(measured and true)', marker='v')
            plt.plot((nCx_hat - nCx_tru) / nCx_tru * 100, 'y', label='rel-err(estimated and true)', marker='v')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Rel-err(%)', fontsize=12, loc='top')
            plt.legend(fontsize=10)

            plt.subplot(2, 2, 2)
            plt.title('ID' + str(id2[i8]) + 'nestCenter.y rel-err comparison', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.plot((nCy_mea - nCy_tru) / nCy_tru * 100, 'r', label='rel-err(measured and true)', marker='v')
            plt.plot((nCy_hat - nCy_tru) / nCy_tru * 100, 'y', label='rel-err(estimated and true)', marker='v')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Rel-err(%)', fontsize=12, loc='top')
            plt.legend(fontsize=10)

            plt.subplot(2, 2, 3)
            plt.title('ID' + str(id2[i8]) + 'velocity.x rel-err comparison', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.plot((vx_mea - vx_tru) / vx_tru * 100, 'r', label='rel-err(measured and true)', marker='v')
            plt.plot((vx_hat - vx_tru) / vx_tru * 100, 'y', label='rel-err(estimated and true)', marker='v')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Rel-err(%)', fontsize=12, loc='top')
            plt.legend(fontsize=10)

            plt.subplot(2, 2, 4)
            plt.title('ID' + str(id2[i8]) + 'velocity.y rel-err comparison', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.plot((vy_mea - vy_tru) / vy_tru * 100, 'r', label='rel-err(measured and true)', marker='v')
            plt.plot((vy_hat - vy_tru) / vy_tru * 100, 'y', label='rel-err(estimated and true)', marker='v')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Rel-err(%)', fontsize=12, loc='top')
            plt.legend(fontsize=10)
            plt.savefig(path1 + 'test' + str(
                i4 + 1) + '/rel_err/ID' + str(id2[i8]) + '_rel_err.svg', dpi=500, bbox_inches='tight')
            plt.close()
            plt.show()
            # 观测相对误差、预测相对误差、估计相对误差
            data_mea_pre_hat_rel_err = pd.DataFrame(
                {'nCx_mea_rel_err': (nCx_mea - nCx_tru) / nCx_tru, 'nCx_hat_rel_err': (nCx_hat - nCx_tru) / nCx_tru,
                 'nCy_mea_rel_err': (nCy_mea - nCy_tru) / nCy_tru, 'nCy_hat_rel_err': (nCy_hat - nCy_tru) / nCy_tru,
                 'vx_mea_rel_err': (vx_mea - vx_tru) / vx_tru, 'vx_hat_rel_err': (vx_hat - vx_tru) / vx_tru,
                 'vy_mea_rel_err': (vy_mea - vy_tru) / vy_tru, 'vy_hat_rel_err': (vy_hat - vy_tru) / vy_tru})
            data_mea_pre_hat_rel_err.to_excel(work_mea_hat_rel_err, sheet_name='ID' + str(id2[i8]), index=False)
            # 观测相对误差均值，预测相对误差均值对比xlsx
            data_mean_mea_pre_hat_rel_err = pd.DataFrame(
                {'mean_nCx_mea_rel_err': np.mean((nCx_mea - nCx_tru) / nCx_tru),
                 'mean_nCx_hat_rel_err': np.mean((nCx_hat - nCx_tru) / nCx_tru),
                 'mean_nCy_mea_rel_err': np.mean((nCy_mea - nCy_tru) / nCy_tru),
                 'mean_nCy_hat_rel_err': np.mean((nCy_hat - nCy_tru) / nCy_tru),
                 'mean_vx_mea_rel_err': np.mean((vx_mea - vx_tru) / vx_tru),
                 'mean_vx_hat_rel_err': np.mean((vx_hat - vx_tru) / vx_tru),
                 'mean_vy_mea_rel_err': np.mean((vy_mea - vy_tru) / vy_tru),
                 'mean_vy_hat_rel_err': np.mean((vy_hat - vy_tru) / vy_tru)}, index=[0])
            data_mean_mea_pre_hat_rel_err.to_excel(work_mean_mea_hat_rel_err, sheet_name='ID' + str(id2[i8]),
                                                   index=False)

    else:
        mea_X2 = np.nan_to_num(mea_X2)
        tru_X2 = np.nan_to_num(tru_X2)
        pre_X2 = np.nan_to_num(pre_X2)
        hat_X2 = np.nan_to_num(hat_X2)

    Q3 = np.array([[mean_nCx_var3, 0, 0, 0],
                   [0, mean_nCy_var3, 0, 0],
                   [0, 0, mean_vx_var3, 0],
                   [0, 0, 0, mean_vy_var3]])
    R3 = np.array([[mean_nCx_var3, 0, 0, 0],
                   [0, mean_nCy_var3, 0, 0],
                   [0, 0, mean_vx_var3, 0],
                   [0, 0, 0, mean_vy_var3]])
    mea_X3, tru_X3, pre_X3, hat_X3 = [], [], [], []
    if np.isnan(Q3).any() == False:
        mea_X3, tru_X3, pre_X3, hat_X3 = kalmanfilter(Q3, R3, tru_nCx3, tru_nCy3, tru_vx3, tru_vy3, mea_nCx3, mea_nCy3,
                                                      mea_vx3, mea_vy3)
        mea_X3, tru_X3, pre_X3, hat_X3 = np.array(mea_X3), np.array(tru_X3), np.array(pre_X3), np.array(hat_X3)
        for i8 in range(0, len(id3)):
            nCx_tru, nCy_tru, vx_tru, vy_tru = tru_X3[:, :, i8][:, 0], tru_X3[:, :, i8][:, 1], tru_X3[:, :, i8][:, 2], tru_X3[:, :, i8][:, 3]
            nCx_mea, nCy_mea, vx_mea, vy_mea = mea_X3[:, :, i8][:, 0], mea_X3[:, :, i8][:, 1], mea_X3[:, :, i8][:, 2], mea_X3[:, :, i8][:, 3]
            nCx_pre, nCy_pre, vx_pre, vy_pre = pre_X3[:, :, i8][:, 0], pre_X3[:, :, i8][:, 1], pre_X3[:, :, i8][:, 2], pre_X3[:, :, i8][:, 3]
            nCx_hat, nCy_hat, vx_hat, vy_hat = hat_X3[:, :, i8][:, 0], hat_X3[:, :, i8][:, 1], hat_X3[:, :, i8][:, 2], hat_X3[:, :, i8][:, 3]

            '''kalman滤波状态估计'''
            # kalman滤波状态估计
            plt.figure(figsize=(16, 10))
            plt.subplot(2, 2, 1)
            plt.title('ID' + str(id3[i8]) + 'nestCenter.x state estimate(partition)', fontsize=16)
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
            plt.title('ID' + str(id3[i8]) + 'nestCenter.y state estimate(partition)', fontsize=16)
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
            plt.title('ID' + str(id3[i8]) + 'velocity.x state estimate(partition)', fontsize=16)
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
            plt.title('ID' + str(id3[i8]) + 'velocity.y state estimate(partition)', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.plot(vy_tru, 'royalblue', label='True value', marker='o')
            plt.plot(vy_mea, 'orange', label='Measured value', marker='s')
            plt.plot(vy_pre, 'darkgrey', label='Predicted value', marker='*')
            plt.plot(vy_hat, 'red', label='Estimated value', marker='^')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Velocity(m/s)', fontsize=12, loc='top')
            plt.legend(fontsize=10)
            plt.savefig(path1 + 'test' + str(
                i4 + 1) + '/hat/ID' + str(id3[i8]) + '_hat.svg', dpi=500, bbox_inches='tight')
            plt.close()
            plt.show()
            # 观测值、真值、预测值、估计值xlsx
            data_tru_mea_pre_hat = pd.DataFrame({'nestCenter_x_tru': nCx_tru, 'nestCenter_x_mea': nCx_mea,
                                                 'nestCenter_x_pre': nCx_pre, 'nestCenter_x_hat': nCx_hat,
                                                 'nestCenter_y_tru': nCy_tru, 'nestCenter_y_mea': nCy_mea,
                                                 'nestCenter_y_pre': nCy_pre, 'nestCenter_y_hat': nCy_hat,
                                                 'velocity_x_tru': vx_tru, 'velocity_x_mea': vx_mea,
                                                 'velocity_x_pre': vx_pre, 'velocity_x_hat': vx_hat,
                                                 'velocity_y_tru': vy_tru, 'velocity_y_mea': vy_mea,
                                                 'velocity_y_pre': vy_pre, 'velocity_y_hat': vy_hat})
            data_tru_mea_pre_hat.to_excel(work_tru_mea_pre_hat, sheet_name='ID' + str(id3[i8]), index=False)

            data_mea_hat_mse = pd.DataFrame({'nCx_mea_tru_mse': mean_squared_error(nCx_mea, nCx_tru),
                                             'nCx_hat_tru_mse': mean_squared_error(nCx_hat, nCx_tru),
                                             'nCy_mea_tru_mse': mean_squared_error(nCy_mea, nCy_tru),
                                             'nCy_hat_tru_mse': mean_squared_error(nCy_hat, nCy_tru),
                                             'vx_mea_tru_mse': mean_squared_error(vx_mea, vx_tru),
                                             'vx_hat_tru_mse': mean_squared_error(vx_hat, vx_tru),
                                             'vy_mea_tru_mse': mean_squared_error(vy_mea, vy_tru),
                                             'vy_hat_tru_mse': mean_squared_error(vy_hat, vy_tru)}, index=[0])
            data_mea_hat_mse.to_excel(work_mea_hat_mse, sheet_name='ID' + str(id3[i8]), index=False)

            # 观测误差、预测误差、估计误差对比
            plt.figure(figsize=(16, 10))
            plt.subplot(2, 2, 1)
            plt.title('ID' + str(id3[i8]) + 'nestCenter.x error comparison', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.plot(nCx_mea - nCx_tru, 'r', label='Measured error', marker='v')
            plt.plot(nCx_hat - nCx_tru, 'b', label='Estimated error', marker='v')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Distance(m)', fontsize=12, loc='top')
            plt.legend(fontsize=10)

            plt.subplot(2, 2, 2)
            plt.title('ID' + str(id3[i8]) + 'nestCenter.y error comparison', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.plot(nCy_mea - nCy_tru, 'r', label='Measured error', marker='v')
            plt.plot(nCy_hat - nCy_tru, 'b', label='Estimated error', marker='v')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Distance(m)', fontsize=12, loc='top')
            plt.legend(fontsize=10)

            plt.subplot(2, 2, 3)
            plt.title('ID' + str(id3[i8]) + 'velocity.x error comparison', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.plot(vx_mea - vx_tru, 'r', label='Measured error', marker='v')
            plt.plot(vx_hat - vx_tru, 'b', label='Estimated error', marker='v')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Velocity(m/s)', fontsize=12, loc='top')
            plt.legend(fontsize=10)

            plt.subplot(2, 2, 4)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.title('ID' + str(id3[i8]) + 'velocity.y error comparison', fontsize=16)
            plt.plot(vy_mea - vy_tru, 'r', label='Measured error', marker='v')
            plt.plot(vy_hat - vy_tru, 'b', label='Estimated error', marker='v')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Velocity(m/s)', fontsize=12, loc='top')
            plt.legend(fontsize=10)
            plt.savefig(path1 + 'test' + str(
                i4 + 1) + '/mea_err/ID' + str(id3[i8]) + '_mea_err.svg', dpi=500, bbox_inches='tight')
            plt.close()
            plt.show()

            # 观测误差、估计误差xlsx
            data_mea_pre_hat_err = pd.DataFrame(
                {'nCx_mea_err': nCx_mea - nCx_tru, 'nCx_hat_err': nCx_hat - nCx_tru,
                 'nCy_mea_err': nCy_mea - nCy_tru, 'nCy_hat_err': nCy_hat - nCy_tru,
                 'vx_mea_err': vx_mea - vx_tru, 'vx_hat_err': vx_hat - vx_tru,
                 'vy_mea_err': vy_mea - vy_tru, 'vy_hat_err': vy_hat - vy_tru})
            data_mea_pre_hat_err.to_excel(work_mea_hat_err, sheet_name='ID' + str(id3[i8]), index=False)
            # 观测误差均值、预测误差均值、估计误差均值xlsx
            data_mean_mea_pre_hat_err = pd.DataFrame(
                {'mean_nCx_mea_err': np.mean(nCx_mea - nCx_tru), 'mean_nCx_hat_err': np.mean(nCx_hat - nCx_tru),
                 'mean_nCy_mea_err': np.mean(nCy_mea - nCy_tru), 'mean_nCy_hat_err': np.mean(nCy_hat - nCy_tru),
                 'mean_vx_mea_err': np.mean(vx_mea - vx_tru), 'mean_vx_hat_err': np.mean(vx_hat - vx_tru),
                 'mean_vy_mea_err': np.mean(vy_mea - vy_tru), 'mean_vy_hat_err': np.mean(vy_hat - vy_tru)}, index=[0])
            data_mean_mea_pre_hat_err.to_excel(work_mean_mea_hat_err, sheet_name='ID' + str(id3[i8]), index=False)

            plt.figure(figsize=(16, 10))
            plt.subplot(2, 2, 1)
            plt.title('ID' + str(id3[i8]) + 'nestCenter.x rel-err comparison', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.plot((nCx_mea - nCx_tru) / nCx_tru * 100, 'r', label='rel-err(measured and true)', marker='v')
            plt.plot((nCx_hat - nCx_tru) / nCx_tru * 100, 'y', label='rel-err(estimated and true)', marker='v')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Rel-err(%)', fontsize=12, loc='top')
            plt.legend(fontsize=10)

            plt.subplot(2, 2, 2)
            plt.title('ID' + str(id3[i8]) + 'nestCenter.y rel-err comparison', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.plot((nCy_mea - nCy_tru) / nCy_tru * 100, 'r', label='rel-err(measured and true)', marker='v')
            plt.plot((nCy_hat - nCy_tru) / nCy_tru * 100, 'y', label='rel-err(estimated and true)', marker='v')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Rel-err(%)', fontsize=12, loc='top')
            plt.legend(fontsize=10)

            plt.subplot(2, 2, 3)
            plt.title('ID' + str(id3[i8]) + 'velocity.x rel-err comparison', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.plot((vx_mea - vx_tru) / vx_tru * 100, 'r', label='rel-err(measured and true)', marker='v')
            plt.plot((vx_hat - vx_tru) / vx_tru * 100, 'y', label='rel-err(estimated and true)', marker='v')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Rel-err(%)', fontsize=12, loc='top')
            plt.legend(fontsize=10)

            plt.subplot(2, 2, 4)
            plt.title('ID' + str(id3[i8]) + 'velocity.y rel-err comparison', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.plot((vy_mea - vy_tru) / vy_tru * 100, 'r', label='rel-err(measured and true)', marker='v')
            plt.plot((vy_hat - vy_tru) / vy_tru * 100, 'y', label='rel-err(estimated and true)', marker='v')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Rel-err(%)', fontsize=12, loc='top')
            plt.legend(fontsize=10)
            plt.savefig(path1 + 'test' + str(
                i4 + 1) + '/rel_err/ID' + str(id3[i8]) + '_rel_err.svg', dpi=500, bbox_inches='tight')
            plt.close()
            plt.show()
            # 观测相对误差、预测相对误差、估计相对误差
            data_mea_pre_hat_rel_err = pd.DataFrame(
                {'nCx_mea_rel_err': (nCx_mea - nCx_tru) / nCx_tru, 'nCx_hat_rel_err': (nCx_hat - nCx_tru) / nCx_tru,
                 'nCy_mea_rel_err': (nCy_mea - nCy_tru) / nCy_tru, 'nCy_hat_rel_err': (nCy_hat - nCy_tru) / nCy_tru,
                 'vx_mea_rel_err': (vx_mea - vx_tru) / vx_tru, 'vx_hat_rel_err': (vx_hat - vx_tru) / vx_tru,
                 'vy_mea_rel_err': (vy_mea - vy_tru) / vy_tru, 'vy_hat_rel_err': (vy_hat - vy_tru) / vy_tru})
            data_mea_pre_hat_rel_err.to_excel(work_mea_hat_rel_err, sheet_name='ID' + str(id3[i8]), index=False)
            # 观测相对误差均值，预测相对误差均值对比xlsx
            data_mean_mea_pre_hat_rel_err = pd.DataFrame(
                {'mean_nCx_mea_rel_err': np.mean((nCx_mea - nCx_tru) / nCx_tru),
                 'mean_nCx_hat_rel_err': np.mean((nCx_hat - nCx_tru) / nCx_tru),
                 'mean_nCy_mea_rel_err': np.mean((nCy_mea - nCy_tru) / nCy_tru),
                 'mean_nCy_hat_rel_err': np.mean((nCy_hat - nCy_tru) / nCy_tru),
                 'mean_vx_mea_rel_err': np.mean((vx_mea - vx_tru) / vx_tru),
                 'mean_vx_hat_rel_err': np.mean((vx_hat - vx_tru) / vx_tru),
                 'mean_vy_mea_rel_err': np.mean((vy_mea - vy_tru) / vy_tru),
                 'mean_vy_hat_rel_err': np.mean((vy_hat - vy_tru) / vy_tru)}, index=[0])
            data_mean_mea_pre_hat_rel_err.to_excel(work_mean_mea_hat_rel_err, sheet_name='ID' + str(id3[i8]),
                                                   index=False)
    else:
        mea_X3 = np.nan_to_num(mea_X3)
        tru_X3 = np.nan_to_num(tru_X3)
        pre_X3 = np.nan_to_num(pre_X3)
        hat_X3 = np.nan_to_num(hat_X3)

    Q4 = np.array([[mean_nCx_var4, 0, 0, 0],
                   [0, mean_nCy_var4, 0, 0],
                   [0, 0, mean_vx_var4, 0],
                   [0, 0, 0, mean_vy_var4]])
    R4 = np.array([[mean_nCx_var4, 0, 0, 0],
                   [0, mean_nCy_var4, 0, 0],
                   [0, 0, mean_vx_var4, 0],
                   [0, 0, 0, mean_vy_var4]])
    mea_X4, tru_X4, pre_X4, hat_X4 = [], [], [], []
    if np.isnan(Q4).any() == False:
        mea_X4, tru_X4, pre_X4, hat_X4 = kalmanfilter(Q4, R4, tru_nCx4, tru_nCy4, tru_vx4, tru_vy4, mea_nCx4, mea_nCy4,
                                                      mea_vx4, mea_vy4)
        mea_X4, tru_X4, pre_X4, hat_X4 = np.array(mea_X4), np.array(tru_X4), np.array(pre_X4), np.array(hat_X4)
        for i8 in range(0, len(id4)):
            nCx_tru, nCy_tru, vx_tru, vy_tru = tru_X4[:, :, i8][:, 0], tru_X4[:, :, i8][:, 1], tru_X4[:, :, i8][:, 2], tru_X4[:, :, i8][:, 3]
            nCx_mea, nCy_mea, vx_mea, vy_mea = mea_X4[:, :, i8][:, 0], mea_X4[:, :, i8][:, 1], mea_X4[:, :, i8][:, 2], mea_X4[:, :, i8][:, 3]
            nCx_pre, nCy_pre, vx_pre, vy_pre = pre_X4[:, :, i8][:, 0], pre_X4[:, :, i8][:, 1], pre_X4[:, :, i8][:, 2], pre_X4[:, :, i8][:,3]
            nCx_hat, nCy_hat, vx_hat, vy_hat = hat_X4[:, :, i8][:, 0], hat_X4[:, :, i8][:, 1], hat_X4[:, :, i8][:, 2], hat_X4[:, :, i8][:,3]

            '''kalman滤波状态估计'''
            # kalman滤波状态估计
            plt.figure(figsize=(16, 10))
            plt.subplot(2, 2, 1)
            plt.title('ID' + str(id4[i8]) + 'nestCenter.x state estimate(partition)', fontsize=16)
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
            plt.title('ID' + str(id4[i8]) + 'nestCenter.y state estimate(partition)', fontsize=16)
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
            plt.title('ID' + str(id4[i8]) + 'velocity.x state estimate(partition)', fontsize=16)
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
            plt.title('ID' + str(id4[i8]) + 'velocity.y state estimate(partition)', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.plot(vy_tru, 'royalblue', label='True value', marker='o')
            plt.plot(vy_mea, 'orange', label='Measured value', marker='s')
            plt.plot(vy_pre, 'darkgrey', label='Predicted value', marker='*')
            plt.plot(vy_hat, 'red', label='Estimated value', marker='^')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Velocity(m/s)', fontsize=12, loc='top')
            plt.legend(fontsize=10)
            plt.savefig(path1 + 'test' + str(
                i4 + 1) + '/hat/ID' + str(id4[i8]) + '_hat.svg', dpi=500, bbox_inches='tight')
            plt.close()
            plt.show()
            # 观测值、真值、预测值、估计值xlsx
            data_tru_mea_pre_hat = pd.DataFrame({'nestCenter_x_tru': nCx_tru, 'nestCenter_x_mea': nCx_mea,
                                                 'nestCenter_x_pre': nCx_pre, 'nestCenter_x_hat': nCx_hat,
                                                 'nestCenter_y_tru': nCy_tru, 'nestCenter_y_mea': nCy_mea,
                                                 'nestCenter_y_pre': nCy_pre, 'nestCenter_y_hat': nCy_hat,
                                                 'velocity_x_tru': vx_tru, 'velocity_x_mea': vx_mea,
                                                 'velocity_x_pre': vx_pre, 'velocity_x_hat': vx_hat,
                                                 'velocity_y_tru': vy_tru, 'velocity_y_mea': vy_mea,
                                                 'velocity_y_pre': vy_pre, 'velocity_y_hat': vy_hat})
            data_tru_mea_pre_hat.to_excel(work_tru_mea_pre_hat, sheet_name='ID' + str(id4[i8]), index=False)

            data_mea_hat_mse = pd.DataFrame({'nCx_mea_tru_mse': mean_squared_error(nCx_mea, nCx_tru),
                                             'nCx_hat_tru_mse': mean_squared_error(nCx_hat, nCx_tru),
                                             'nCy_mea_tru_mse': mean_squared_error(nCy_mea, nCy_tru),
                                             'nCy_hat_tru_mse': mean_squared_error(nCy_hat, nCy_tru),
                                             'vx_mea_tru_mse': mean_squared_error(vx_mea, vx_tru),
                                             'vx_hat_tru_mse': mean_squared_error(vx_hat, vx_tru),
                                             'vy_mea_tru_mse': mean_squared_error(vy_mea, vy_tru),
                                             'vy_hat_tru_mse': mean_squared_error(vy_hat, vy_tru)}, index=[0])
            data_mea_hat_mse.to_excel(work_mea_hat_mse, sheet_name='ID' + str(id4[i8]), index=False)

            # 观测误差、预测误差、估计误差对比
            plt.figure(figsize=(16, 10))
            plt.subplot(2, 2, 1)
            plt.title('ID' + str(id4[i8]) + 'nestCenter.x error comparison', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.plot(nCx_mea - nCx_tru, 'r', label='Measured error', marker='v')
            plt.plot(nCx_hat - nCx_tru, 'b', label='Estimated error', marker='v')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Distance(m)', fontsize=12, loc='top')
            plt.legend(fontsize=10)

            plt.subplot(2, 2, 2)
            plt.title('ID' + str(id4[i8]) + 'nestCenter.y error comparison', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.plot(nCy_mea - nCy_tru, 'r', label='Measured error', marker='v')
            plt.plot(nCy_hat - nCy_tru, 'b', label='Estimated error', marker='v')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Distance(m)', fontsize=12, loc='top')
            plt.legend(fontsize=10)

            plt.subplot(2, 2, 3)
            plt.title('ID' + str(id4[i8]) + 'velocity.x error comparison', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.plot(vx_mea - vx_tru, 'r', label='Measured error', marker='v')
            plt.plot(vx_hat - vx_tru, 'b', label='Estimated error', marker='v')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Velocity(m/s)', fontsize=12, loc='top')
            plt.legend(fontsize=10)

            plt.subplot(2, 2, 4)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.title('ID' + str(id4[i8]) + 'velocity.y error comparison', fontsize=16)
            plt.plot(vy_mea - vy_tru, 'r', label='Measured error', marker='v')
            plt.plot(vy_hat - vy_tru, 'b', label='Estimated error', marker='v')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Velocity(m/s)', fontsize=12, loc='top')
            plt.legend(fontsize=10)
            plt.savefig(path1 + 'test' + str(
                i4 + 1) + '/mea_err/ID' + str(id4[i8]) + '_mea_err.svg', dpi=500, bbox_inches='tight')
            plt.close()
            plt.show()

            # 观测误差、估计误差xlsx
            data_mea_pre_hat_err = pd.DataFrame(
                {'nCx_mea_err': nCx_mea - nCx_tru, 'nCx_hat_err': nCx_hat - nCx_tru,
                 'nCy_mea_err': nCy_mea - nCy_tru, 'nCy_hat_err': nCy_hat - nCy_tru,
                 'vx_mea_err': vx_mea - vx_tru, 'vx_hat_err': vx_hat - vx_tru,
                 'vy_mea_err': vy_mea - vy_tru, 'vy_hat_err': vy_hat - vy_tru})
            data_mea_pre_hat_err.to_excel(work_mea_hat_err, sheet_name='ID' + str(id4[i8]), index=False)
            # 观测误差均值、预测误差均值、估计误差均值xlsx
            data_mean_mea_pre_hat_err = pd.DataFrame(
                {'mean_nCx_mea_err': np.mean(nCx_mea - nCx_tru), 'mean_nCx_hat_err': np.mean(nCx_hat - nCx_tru),
                 'mean_nCy_mea_err': np.mean(nCy_mea - nCy_tru), 'mean_nCy_hat_err': np.mean(nCy_hat - nCy_tru),
                 'mean_vx_mea_err': np.mean(vx_mea - vx_tru), 'mean_vx_hat_err': np.mean(vx_hat - vx_tru),
                 'mean_vy_mea_err': np.mean(vy_mea - vy_tru), 'mean_vy_hat_err': np.mean(vy_hat - vy_tru)}, index=[0])
            data_mean_mea_pre_hat_err.to_excel(work_mean_mea_hat_err, sheet_name='ID' + str(id4[i8]), index=False)

            plt.figure(figsize=(16, 10))
            plt.subplot(2, 2, 1)
            plt.title('ID' + str(id4[i8]) + 'nestCenter.x rel-err comparison', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.plot((nCx_mea - nCx_tru) / nCx_tru * 100, 'r', label='rel-err(measured and true)', marker='v')
            plt.plot((nCx_hat - nCx_tru) / nCx_tru * 100, 'y', label='rel-err(estimated and true)', marker='v')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Rel-err(%)', fontsize=12, loc='top')
            plt.legend(fontsize=10)

            plt.subplot(2, 2, 2)
            plt.title('ID' + str(id4[i8]) + 'nestCenter.y rel-err comparison', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.plot((nCy_mea - nCy_tru) / nCy_tru * 100, 'r', label='rel-err(measured and true)', marker='v')
            plt.plot((nCy_hat - nCy_tru) / nCy_tru * 100, 'y', label='rel-err(estimated and true)', marker='v')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Rel-err(%)', fontsize=12, loc='top')
            plt.legend(fontsize=10)

            plt.subplot(2, 2, 3)
            plt.title('ID' + str(id4[i8]) + 'velocity.x rel-err comparison', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.plot((vx_mea - vx_tru) / vx_tru * 100, 'r', label='rel-err(measured and true)', marker='v')
            plt.plot((vx_hat - vx_tru) / vx_tru * 100, 'y', label='rel-err(estimated and true)', marker='v')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Rel-err(%)', fontsize=12, loc='top')
            plt.legend(fontsize=10)

            plt.subplot(2, 2, 4)
            plt.title('ID' + str(id4[i8]) + 'velocity.y rel-err comparison', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.plot((vy_mea - vy_tru) / vy_tru * 100, 'r', label='rel-err(measured and true)', marker='v')
            plt.plot((vy_hat - vy_tru) / vy_tru * 100, 'y', label='rel-err(estimated and true)', marker='v')
            plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
            plt.ylabel('Rel-err(%)', fontsize=12, loc='top')
            plt.legend(fontsize=10)
            plt.savefig(path1 + 'test' + str(
                i4 + 1) + '/rel_err/ID' + str(id4[i8]) + '_rel_err.svg', dpi=500, bbox_inches='tight')
            plt.close()
            plt.show()
            # 观测相对误差、预测相对误差、估计相对误差
            data_mea_pre_hat_rel_err = pd.DataFrame(
                {'nCx_mea_rel_err': (nCx_mea - nCx_tru) / nCx_tru, 'nCx_hat_rel_err': (nCx_hat - nCx_tru) / nCx_tru,
                 'nCy_mea_rel_err': (nCy_mea - nCy_tru) / nCy_tru, 'nCy_hat_rel_err': (nCy_hat - nCy_tru) / nCy_tru,
                 'vx_mea_rel_err': (vx_mea - vx_tru) / vx_tru, 'vx_hat_rel_err': (vx_hat - vx_tru) / vx_tru,
                 'vy_mea_rel_err': (vy_mea - vy_tru) / vy_tru, 'vy_hat_rel_err': (vy_hat - vy_tru) / vy_tru})
            data_mea_pre_hat_rel_err.to_excel(work_mea_hat_rel_err, sheet_name='ID' + str(id4[i8]), index=False)
            # 观测相对误差均值，预测相对误差均值对比xlsx
            data_mean_mea_pre_hat_rel_err = pd.DataFrame(
                {'mean_nCx_mea_rel_err': np.mean((nCx_mea - nCx_tru) / nCx_tru),
                 'mean_nCx_hat_rel_err': np.mean((nCx_hat - nCx_tru) / nCx_tru),
                 'mean_nCy_mea_rel_err': np.mean((nCy_mea - nCy_tru) / nCy_tru),
                 'mean_nCy_hat_rel_err': np.mean((nCy_hat - nCy_tru) / nCy_tru),
                 'mean_vx_mea_rel_err': np.mean((vx_mea - vx_tru) / vx_tru),
                 'mean_vx_hat_rel_err': np.mean((vx_hat - vx_tru) / vx_tru),
                 'mean_vy_mea_rel_err': np.mean((vy_mea - vy_tru) / vy_tru),
                 'mean_vy_hat_rel_err': np.mean((vy_hat - vy_tru) / vy_tru)}, index=[0])
            data_mean_mea_pre_hat_rel_err.to_excel(work_mean_mea_hat_rel_err, sheet_name='ID' + str(id4[i8]),
                                                   index=False)
    else:
        mea_X4 = np.nan_to_num(mea_X4)
        tru_X4 = np.nan_to_num(tru_X4)
        pre_X4 = np.nan_to_num(pre_X4)
        hat_X4 = np.nan_to_num(hat_X4)

    work_tru_mea_pre_hat.save()
    work_mea_hat_mse.save()
    work_mea_hat_err.save()
    work_mean_mea_hat_err.save()
    work_mea_hat_rel_err.save()
    work_mean_mea_hat_rel_err.save()

    print('第' + str(i4 + 1) + '组样本数据匀加速真值情况下的Kalman Filtering状态估计测试完成')
    print('-------------------')

end = timeit.default_timer()
print('Running time: %s Seconds' % (end - start))
print('测试结束')