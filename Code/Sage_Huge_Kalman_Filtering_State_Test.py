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
data_match = pd.read_csv('../Data/match(数据关联).csv', encoding='gbk')
df_match = pd.DataFrame(data_match, columns=['frame_num', 'lidar_track_id', 'dt_track_id'], dtype=int)
# 关联数据的帧数、真值ID、传感器ID
frame_match = list(df_match.iloc[:, 0])
tru_id_match = list(df_match.iloc[:, 1])
lidar_id_match = list(df_match.iloc[:, 2])
# 设定20组样本数据进行测试
test_groups = int(1000 / 50)

# 创建test1-test20的20个文件夹
path1 = '../Sage_Huge_Kalman_Filtering_State_Test/'

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

# 提取50帧中满足达到50帧的传感器ID，其余不满50帧的（若帧数缺失较少，手动补全；缺失较多，不计该目标）
test_group_id = []
for i3 in range(0, test_groups):
    ele_id1 = []
    for ele1 in group_id[i3]:
        if lidar_id_match[frame_match.index(i3 * 50):frame_match.index(i3 * 50 + 50)].count(ele1) == 50:
            ele_id1.append(ele1)
    test_group_id.append(ele_id1)

for i4 in range(0, len(test_group_id)):  # 20组测试
    # 每组的目标ID个数
    obj_num = len(test_group_id[i4])
    #  分别存放多个目标的nestCenter.x、nestCenter.y、velocity.x、velocity.y的真值的list
    tru_nCx_all, tru_nCy_all, tru_vx_all, tru_vy_all = [], [], [], []
    # 分别存放多个目标的nestCenter.x、nestCenter.y、velocity.x、velocity.y的观测值的list
    mea_nCx_all, mea_nCy_all, mea_vx_all, mea_vy_all = [], [], [], []
    # 分别存放多个目标的nestCenter.x、nestCenter.y、velocity.x、velocity.y的观测误差的list
    mea_nCx_var, mea_nCy_var, mea_vx_var, mea_vy_var = [], [], [], []
    # 遍历该数据样本的所有目标ID
    for id in test_group_id[i4]:
        # 读取目标数据
        obj_data = pd.read_csv(
            '../Objects_Data/ID' + str(id) + '/ID' + str(
                id) + '_data.csv', encoding='gbk')
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
        # 分别添加多个目标的nCx、nCy、vx、vy的观测误差方差
        mea_nCx_var.append(mea_nCx_var_obj)
        mea_nCy_var.append(mea_nCy_var_obj)
        mea_vx_var.append(mea_vx_var_obj)
        mea_vy_var.append(mea_vy_var_obj)
    # 求观测误差方差均值
    '''
        均值初始化
    '''
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
        Sage-husa Kalman filtering
        Xk|k-1 = A * Xk-1 + qk-1
        Pk|k-1 = A * Pk-1 * A.T +Qk-1

        dk = (1 - b)/(1 - b**(k+1))
        ek = Zk - H * Xk|k-1 -rk-1
        rk = (1 -dk) * rk-1 + dk * (Zk - H * Xk|k-1)
        Rk = (1 - dk) Rk-1 + dk * (ek * ek.T - H * Pk|k-1 * H.T)

        Kk = Pk|k-1 * H.T/(H * Pk|k-1 * H.T + Rk)
        Xk = Xk|k-1 + Kk * ek
        Pk = (I - Kk * H) * Pk|k-1
        qk = (1 - dk) * qk-1) + dk * (Xk - A * Xk-1)
        Qk = (1 - dk) * Qk-1 + dk * (Kk * ek * ek.T * K.T + Pk - A * Pk-1 * A.T) 
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

        Qk = np.array([[0.1, 0, 0, 0],
                       [0, 0.1, 0, 0],
                       [0, 0, 0.1, 0],
                       [0, 0, 0, 0.1]])  # 过程噪声协方差矩阵

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

        # q = [qk]
        # Q = [Qk]
        r = [rk]
        R = [Rk]

        Xhat = np.array([tru_nCx[0], tru_nCy[0], tru_vx[0], tru_vy[0]])  # 上一时刻的多个目标的估计值

        Phat = np.array([[0.1, 0, 0, 0],
                         [0, 0.1, 0, 0],
                         [0, 0, 0.1, 0],
                         [0, 0, 0, 0.1]])  # 上一时刻的误差协方差矩阵

        mea_X, tru_X, pre_X, hat_X = [], [], [], [Xhat]
        pre_P, hat_P = [], [Phat]

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
    # 将list转数组
    mea_X = np.array(mea_X)  # (50, 4, 5)
    tru_X = np.array(tru_X)  # (50, 4, 5)
    pre_X = np.array(pre_X)  # (50, 4, 5)
    hat_X = np.array(hat_X)  # (50, 4, 5)

    # 将多个目标状态的真值、观测、预测、估计可视化
    for i11 in range(0, obj_num):
        plt.figure(figsize=(16, 10))
        plt.subplot(2, 2, 1)
        plt.title('ID' + str(test_group_id[i4][i11]) + 'nestCenter.x state estimate(Sage-Huge)', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.plot(tru_X[:, :, i11][:, 0], 'royalblue', label='True value', marker='o')
        plt.plot(mea_X[:, :, i11][:, 0], 'orange', label='Measured value', marker='s')
        plt.plot(pre_X[:, :, i11][:, 0], 'darkgrey', label='Predicted value', marker='*')
        plt.plot(hat_X[:, :, i11][:, 0], 'red', label='Estimated value', marker='^')
        plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
        plt.ylabel('Distance(m)', fontsize=12, loc='top')
        plt.legend(fontsize=10)

        plt.subplot(2, 2, 2)
        plt.title('ID' + str(test_group_id[i4][i11]) + 'nestCenter.y state estimate(Sage-Huge)', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.plot(tru_X[:, :, i11][:, 1], 'royalblue', label='True value', marker='o')
        plt.plot(mea_X[:, :, i11][:, 1], 'orange', label='Measured value', marker='s')
        plt.plot(pre_X[:, :, i11][:, 1], 'darkgrey', label='Predicted value', marker='*')
        plt.plot(hat_X[:, :, i11][:, 1], 'red', label='Estimated value', marker='^')
        plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
        plt.ylabel('Distance(m)', fontsize=12, loc='top')
        plt.legend(fontsize=10)

        plt.subplot(2, 2, 3)
        plt.title('ID' + str(test_group_id[i4][i11]) + 'velocity.x state estimate(Sage-Huge)', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.plot(tru_X[:, :, i11][:, 2], 'royalblue', label='True value', marker='o')
        plt.plot(mea_X[:, :, i11][:, 2], 'orange', label='Measured value', marker='s')
        plt.plot(pre_X[:, :, i11][:, 2], 'darkgrey', label='Predicted value', marker='*')
        plt.plot(hat_X[:, :, i11][:, 2], 'red', label='Estimated value', marker='^')
        plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
        plt.ylabel('Velocity(m/s)', fontsize=12, loc='top')
        plt.legend(fontsize=10)

        plt.subplot(2, 2, 4)
        plt.title('ID' + str(test_group_id[i4][i11]) + 'velocity.y state estimate(Sage-Huge)', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.plot(tru_X[:, :, i11][:, 3], 'royalblue', label='True value', marker='o')
        plt.plot(mea_X[:, :, i11][:, 3], 'orange', label='Measured value', marker='s')
        plt.plot(pre_X[:, :, i11][:, 3], 'darkgrey', label='Predicted value', marker='*')
        plt.plot(hat_X[:, :, i11][:, 3], 'red', label='Estimated value', marker='^')
        plt.xlabel('Frame(frame_num)', fontsize=12, loc='right')
        plt.ylabel('Velocity(m/s)', fontsize=12, loc='top')
        plt.legend(fontsize=10)
        plt.savefig(
            r'../Sage_Huge_Kalman_Filtering_State_Test/test' + str(
                i4 + 1) + '/ID' + str(test_group_id[i4][i11]) + '.svg', dpi=500, bbox_inches='tight')
        plt.close()
        plt.show()

    print('第' + str(i4 + 1) + '组样本数据真值情况下的Sage-Huge Kalman Filtering状态估计测试完成')
    print('-------------------')

end = timeit.default_timer()
print('Running time: %s Seconds' % (end - start))
print('测试结束')