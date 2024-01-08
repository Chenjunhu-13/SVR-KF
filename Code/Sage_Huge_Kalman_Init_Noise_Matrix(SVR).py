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
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 读取关联数据CSV文件
data_match = pd.read_csv('../Data/match(数据关联).csv',
                         encoding='gbk')
df_match = pd.DataFrame(data_match, columns=['frame_num', 'lidar_track_id', 'dt_track_id'], dtype=int)
# 关联数据的帧数、真值ID、传感器ID
frame_match = list(df_match.iloc[:, 0])
tru_id_match = list(df_match.iloc[:, 1])
lidar_id_match = list(df_match.iloc[:, 2])
# 设定20组样本数据进行测试
test_groups = int(1000 / 50)

# 创建test1-test20的20个文件夹
path1 = '../Sage_Huge_Kalman_Init_Matrix_State_Test/'

name = 'test'
for i in range(0, test_groups):
    sExists = os.path.exists(path1 + name + str(i + 1))
    os.makedirs(path1 + name + str(i + 1))
    print("%s 目录创建成功" % (name + str(i + 1)))

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

# 每50帧中满足50帧的传感器ID
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

# 遍历20组
for i4 in range(0, len(test_group_id)):  # 20组测试
    obj_num = len(test_group_id[i4])  # 每组的目标ID个数
    # tru_nCx_all, tru_nCy_all, tru_vx_all, tru_vy_all = [], [], [], []
    # mea_nCx_all, mea_nCy_all, mea_vx_all, mea_vy_all = [], [], [], []
    dt = []
    # 遍历每组的目标ID
    for id in test_group_id[i4]:
        obj_data = pd.read_csv(
            '../Objects_Data/ID' + str(id) + '/ID' + str(
                id) + '_data.csv',
            encoding='gbk')
        df_obj = obj_data.iloc[:, 1]
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


        # 提取对应帧数段的目标数据
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
    '''
        计算观测误差方差均值
        均值初始化 
    '''
    mean_nCx_var = np.mean(mea_nCx_var)
    mean_nCy_var = np.mean(mea_nCy_var)
    mean_vx_var = np.mean(mea_vx_var)
    mean_vy_var = np.mean(mea_vy_var)
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

    def kalmanfilter():

        Ak = np.array([[1, 0, 0.1, 0],
                       [0, 1, 0, 0.1],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])  # 状态转移矩阵

        Hk = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])  # 观测矩阵

        # qk = np.array([[1], [1], [1], [1]])  # 过程噪声均值

        Qk = np.array([[mean_nCx_var, 0, 0, 0],
                       [0, mean_nCy_var, 0, 0],
                       [0, 0, mean_vx_var, 0],
                       [0, 0, 0, mean_vy_var]])  # 过程噪声协方差矩阵

        b = 0.99  # 遗忘因子

        rk = np.array([[1], [1], [1], [1]])

        Rk = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])  # 过程噪声协方差矩阵

        I = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])  # 单位矩阵

        r = [rk]
        R = [Rk]

        Xhat = np.array([tru_nCx[0], tru_nCy[0], tru_vx[0], tru_vy[0]])  # 上一时刻的多个目标的估计值

        Phat = np.array([[0.1, 0, 0, 0],
                         [0, 0.1, 0, 0],
                         [0, 0, 0.1, 0],
                         [0, 0, 0, 0.1]])  # 上一时刻的误差协方差矩阵

        mea_X, tru_X, pre_X, hat_X = [], [], [], []
        pre_P, hat_P = [], []

        for i10 in range(0, 50):
            Z = np.array([mea_nCx[i10], mea_nCy[i10], mea_vx[i10], mea_vy[i10]])  # 当前时刻的观测值
            tru = np.array([tru_nCx[i10], tru_nCy[i10], tru_vx[i10], tru_vy[i10]])  # 当前时刻的真值
            # 预测过程
            Xpre = np.dot(Ak, Xhat)  # 预测当前时刻的多目标的状态
            Ppre = np.dot(np.dot(Ak, Phat), Ak.T) + Qk  # 预测误差协方差矩阵

            dk = (1 - b) / (1 - b ** (i10 + 1))
            ek = Z - np.dot(Hk, Xpre) - rk
            rk = np.dot((1 - dk), rk) + np.dot(dk, (Z - np.dot(Hk, Xpre)))
            Rk = np.dot((1 - dk), Rk) + np.dot(dk, (np.dot(ek, ek.T) - np.dot(np.dot(Hk, Ppre), Hk.T)))

            Kk = np.dot(np.dot(Ppre, Hk.T), np.linalg.pinv(np.dot(np.dot(Hk, Ppre), Hk.T) + Rk))  # 卡尔曼增益
            Xhat = Xpre + np.dot(Kk, ek)
            Phat = np.dot((I - np.dot(Kk, Hk)), Ppre)

            mea_X.append(Z)  # 获取多个目标的状态观测值
            tru_X.append(tru)  # 获取多个目标的状态真实值
            pre_X.append(Xpre)  # 获取多个目标的状态先验估计值（预测值）
            hat_X.append(Xhat)  # 获取多个目标的状态后验估计值
            pre_P.append(Ppre)
            hat_P.append(Phat)

            # qk = np.dot((1 - dk), qk) + np.dot(dk, (hat_X[i10 + 1] - np.dot(Ak, hat_X[i10])))
            # Qk = np.dot((1 - dk), Qk) + np.dot(dk, (np.dot(np.dot(np.dot(Kk, ek), ek.T), Kk.T) + hat_P[i10 + 1] - np.dot(np.dot(Ak, hat_P[i10]), Ak.T)))

            # q.append(qk)
            # Q.append(Qk)
            r.append(rk)
            R.append(Rk)

        return mea_X, tru_X, pre_X, hat_X


    mea_X, tru_X, pre_X, hat_X = kalmanfilter()

    mea_X = np.array(mea_X)  # (50, 4, 5)
    tru_X = np.array(tru_X)  # (50, 4, 5)
    pre_X = np.array(pre_X)  # (50, 4, 5)
    hat_X = np.array(hat_X)  # (50, 4, 5)

    # 保存观测值、真值、预测值、真值
    work_tru_mea_pre_hat = pd.ExcelWriter(
        '../Sage_Huge_Kalman_Init_Matrix_State_Test/test'
        + str(i4 + 1) + '/tru_mea_pre_hat' + str(i4 + 1) + '.xlsx')
    # 保存均方误差（观测值和真值、估计值和真值）
    work_mea_hat_mse = pd.ExcelWriter(
        '../Sage_Huge_Kalman_Init_Matrix_State_Test/test'
        + str(i4 + 1) + '/mea_hat_mse' + str(i4 + 1) + '.xlsx')

    for i13 in range(0, obj_num):
        nCx_tru, nCy_tru, vx_tru, vy_tru = tru_X[:, :, i13][:, 0], tru_X[:, :, i13][:, 1], tru_X[:, :, i13][:,
                                                                                           2], tru_X[:, :, i13][:, 3]
        nCx_mea, nCy_mea, vx_mea, vy_mea = mea_X[:, :, i13][:, 0], mea_X[:, :, i13][:, 1], mea_X[:, :, i13][:,
                                                                                           2], mea_X[:, :, i13][:, 3]
        nCx_pre, nCy_pre, vx_pre, vy_pre = pre_X[:, :, i13][:, 0], pre_X[:, :, i13][:, 1], pre_X[:, :, i13][:,
                                                                                           2], pre_X[:, :, i13][:, 3]
        nCx_hat, nCy_hat, vx_hat, vy_hat = hat_X[:, :, i13][:, 0], hat_X[:, :, i13][:, 1], hat_X[:, :, i13][:,
                                                                                           2], hat_X[:, :, i13][:, 3]

        '''计算均方误差'''
        # 均方误差（观测值和真值）
        nCx_mea_tru_mse, nCy_mea_tru_mse = mean_squared_error(nCx_mea, nCx_tru), mean_squared_error(nCy_mea, nCy_tru)
        vx_mea_tru_mse, vy_mea_tru_mse = mean_squared_error(vx_mea, vx_tru), mean_squared_error(vy_mea, vy_tru)
        # 均方误差（估计值和真值）
        nCx_hat_tru_mse, nCy_hat_tru_mse = mean_squared_error(nCx_hat, nCx_tru), mean_squared_error(nCy_hat, nCy_tru)
        vx_hat_tru_mse, vy_hat_tru_mse = mean_squared_error(vx_hat, vx_tru), mean_squared_error(vy_hat, vy_tru)

        # Kalman滤波估计
        # 状态估计曲线图
        plt.figure(figsize=(16, 10))
        plt.subplot(2, 2, 1)
        plt.title('ID' + str(test_group_id[i4][i13]) + 'nestCenter.x state estimate(Sage-Huge SVR)', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.plot(nCx_tru, 'cornflowerblue', label='True value(fit)', marker='o')
        plt.plot(nCx_mea, 'orange', label='Measured value', marker='s')
        plt.plot(nCx_pre, 'darkgrey', label='Predicted value', marker='*')
        plt.plot(nCx_hat, 'red', label='Estimated value', marker='^')
        plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
        plt.ylabel('Distance(m)', fontsize=12, loc='top')
        plt.legend(fontsize=10)

        plt.subplot(2, 2, 2)
        plt.title('ID' + str(test_group_id[i4][i13]) + 'nestCenter.y state estimate(Sage-Huge SVR)', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.plot(nCy_tru, 'cornflowerblue', label='True value(fit)', marker='o')
        plt.plot(nCy_mea, 'orange', label='Measured value', marker='s')
        plt.plot(nCy_pre, 'darkgrey', label='Predicted value', marker='*')
        plt.plot(nCy_hat, 'red', label='Estimated value', marker='^')
        plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
        plt.ylabel('Distance(m)', fontsize=12, loc='top')
        plt.legend(fontsize=10)

        plt.subplot(2, 2, 3)
        plt.title('ID' + str(test_group_id[i4][i13]) + 'velocity.x state estimate(Sage-Huge SVR)', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.plot(vx_tru, 'cornflowerblue', label='True value(fit)', marker='o')
        plt.plot(vx_mea, 'orange', label='Measured value', marker='s')
        plt.plot(vx_pre, 'darkgrey', label='Predicted value', marker='*')
        plt.plot(vx_hat, 'red', label='Estimated value', marker='^')
        plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
        plt.ylabel('Velocity(m/s)', fontsize=12, loc='top')
        plt.legend(fontsize=10)

        plt.subplot(2, 2, 4)
        plt.title('ID' + str(test_group_id[i4][i13]) + 'velocity.y state estimate(Sage-Huge SVR)', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.plot(vy_tru, 'cornflowerblue', label='True value(fit)', marker='o')
        plt.plot(vy_mea, 'orange', label='Measured value', marker='s')
        plt.plot(vy_pre, 'darkgrey', label='Predicted value', marker='*')
        plt.plot(vy_hat, 'red', label='Estimated value', marker='^')
        plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
        plt.ylabel('Velocity(m/s)', fontsize=12, loc='top')
        plt.legend(fontsize=10)
        plt.savefig(
            r'../Sage_Huge_Kalman_Init_Matrix_State_Test/test' + str(
                i4 + 1) + '//ID' + str(test_group_id[i4][i13]) + '_hat_mean.svg',
            dpi=500, bbox_inches='tight')
        plt.close()
        plt.show()

        '保存观测值、真值、预测值、估计值'
        data_tru_mea_pre_hat = pd.DataFrame(
            {'stamp_sec': np.array(dt[i13].iloc[:, 0]), 'frame_num': np.array(dt[i13].iloc[:, 1]),
             'track_id_tru': np.array(dt[i13].iloc[:, 2]),
             'track_id_mea': np.array(dt[i13].iloc[:, 3]),
             'nestCenter_x_tru': nCx_tru, 'nestCenter_x_mea': nCx_mea,
             'nestCenter_x_pre': nCx_pre, 'nestCenter_x_hat': nCx_hat,
             'nestCenter_y_tru': nCy_tru, 'nestCenter_y_mea': nCy_mea,
             'nestCenter_y_pre': nCy_pre, 'nestCenter_y_hat': nCy_hat,
             'velocity_x_tru': vx_tru, 'velocity_x_mea': vx_mea,
             'velocity_x_pre': vx_pre, 'velocity_x_hat': vx_hat,
             'velocity_y_tru': vy_tru, 'velocity_y_mea': vy_mea,
             'velocity_y_pre': vy_pre, 'velocity_y_hat': vy_hat})
        data_tru_mea_pre_hat.to_excel(work_tru_mea_pre_hat, sheet_name='ID' + str(test_group_id[i4][i13]), index=False)

        # 均方误差xlsx
        data_mea_hat_mse = pd.DataFrame({'nCx_mea_tru_mse': nCx_mea_tru_mse, 'nCx_hat_tru_mse': nCx_hat_tru_mse,
                                         'nCy_mea_tru_mse': nCy_mea_tru_mse, 'nCy_hat_tru_mse': nCy_hat_tru_mse,
                                         'vx_mea_tru_mse': vx_mea_tru_mse, 'vx_hat_tru_mse': vx_hat_tru_mse,
                                         'vy_mea_tru_mse': vy_mea_tru_mse, 'vy_hat_tru_mse': vy_hat_tru_mse}, index=[0])
        data_mea_hat_mse.to_excel(work_mea_hat_mse, sheet_name='ID' + str(test_group_id[i4][i13]), index=False)

    work_tru_mea_pre_hat.save()
    work_mea_hat_mse.save()
        
    print('第' + str(i4 + 1) + '组样本数据经验初始化Rk的Kalman Filtering状态估计测试完成')
    print('-------------------')
    # print('第' + str(i4 + 1) + '组样本数据经验初始化Rk的Kalman Filtering状态估计测试完成')
    # print('-------------------')

end = timeit.default_timer()
print('Running time: %s Seconds' % (end - start))
print('测试结束')

