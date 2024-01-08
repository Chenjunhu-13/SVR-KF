import numpy as np
import scipy

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


def kalmanfilter(Q, R, tru_nCx, tru_nCy, tru_vx, tru_vy, mea_nCx, mea_nCy, mea_vx, mea_vy):
    A = np.array([[1, 0, 0.1, 0],
                  [0, 1, 0, 0.1],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])  # 状态转移矩阵

    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])  # 观测矩阵

    I = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])  # 单位矩阵

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
        Y = np.array([mea_nCx[i10], mea_nCy[i10], mea_vx[i10], mea_vy[i10]])  # 当前时刻的观测值
        tru = np.array([tru_nCx[i10], tru_nCy[i10], tru_vx[i10], tru_vy[i10]])  # 当前时刻的真值
        # Xminus1 =  np.array([tru_nCx[i10], tru_nCy[i10], tru_vx[i10], tru_vy[i10]])   # 预测当前时刻的多目标的状态
        Xminus1 = np.dot(A, Xplus1)  # 预测当前时刻的多目标的状态
        Pminus1 = np.dot(np.dot(A, Pplus1), A.T) + Q  # 预测误差协方差矩阵
        Kk = np.dot(np.dot(Pminus1, H.T), scipy.linalg.pinv(np.dot(np.dot(H, Pminus1), H.T) + R))  # 卡尔曼增益
        Xplus1 = Xminus1 + np.dot(Kk, (Y - np.dot(H, Xminus1)))
        Pplus1 = np.dot((I - np.dot(Kk, H)), Pminus1)

        mea_X.append(Y)  # 获取多个目标的状态观测值
        tru_X.append(tru)  # 获取多个目标的状态真实值
        pre_X.append(Xminus1)  # 获取多个目标的状态先验估计值（预测值）
        hat_X.append(Xplus1)  # 获取多个目标的状态后验估计值

    return mea_X, tru_X, pre_X, hat_X


def tru_value_process(tru_nCx_all, tru_nCy_all, tru_vx_all, tru_vy_all, tru_nCx, tru_nCy, tru_vx, tru_vy):
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
    return tru_nCx, tru_nCy, tru_vx, tru_vy


def mea_value_process(mea_nCx_all, mea_nCy_all, mea_vx_all, mea_vy_all, mea_nCx, mea_nCy, mea_vx, mea_vy):
    for i8 in range(0, 50):
        nCx_m = []
        for mea_nCxobj in mea_nCx_all:
            nCx_m.append(mea_nCxobj[i8])
        mea_nCx.append(nCx_m)
        nCy_m = []
        for mea_nCyobj in mea_nCy_all:
            nCy_m.append(mea_nCyobj[i8])
        mea_nCy.append(nCy_m)
        vx_m = []
        for mea_vxobj in mea_vx_all:
            vx_m.append(mea_vxobj[i8])
        mea_vx.append(vx_m)
        vy_m = []
        for mea_vyobj in mea_vy_all:
            vy_m.append(mea_vyobj[i8])
        mea_vy.append(vy_m)
    return mea_nCx, mea_nCy, mea_vx, mea_vy