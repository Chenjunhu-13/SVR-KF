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
from make_dir import create_dir
import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
# 中文字体
# plt.rcParams['font.sans-serif'] = ['simsun']
# plt.rcParams['axes.unicode_minus'] = False

config = {
            "font.family": 'serif',
            "font.size": 16,# 相当于小四大小
            "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
            "font.serif": ['SimSun'],#宋体
            'axes.unicode_minus': False # 处理负号，即-号
         }
plt.rcParams.update(config)

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
path1 = '../SVR_Fit_Error_Analysis/'

name = 'test'

str1 = 'mea_err', 'rel_err', 'fit_result'
for i in range(0, test_groups):
    path_test = path1 + name + str(i + 1)
    sExists = os.path.exists(path_test)
    if sExists:
        shutil.rmtree(path_test)
        os.makedirs(path_test)
        print("%s 目录创建成功" % (name + str(i + 1)))
    else:
        os.makedirs(path_test)
        print("%s 目录创建成功" % (name + str(i + 1)))
    path2 = path1 + name + str(
        i + 1) + '/'
    for i1 in str1:
        sExists1 = os.path.exists(path2 + i1)
        if sExists1:
            shutil.rmtree(path2 + i1)
            os.makedirs(path2 + i1)
            print("%s 目录创建成功" % (i1))
        else:
            os.makedirs(path2 + i1)
            print("%s 目录创建成功" % (i1))


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
for i4 in range(0, len(test_group_id)):
    obj_num = len(test_group_id[i4])
    # tru_nCx_all, tru_nCy_all, tru_vx_all, tru_vy_all = [], [], [], []
    # mea_nCx_all, mea_nCy_all, mea_vx_all, mea_vy_all = [], [], [], []
    # mea_nCx_var, mea_nCy_var, mea_vx_var, mea_vy_var = [], [], [], []
    dt = []
    # 遍历每组的目标ID
    for id in test_group_id[i4]:
        obj_data = pd.read_csv(
            '../Objects_Data/ID' + str(id) + '/ID' + str(
                id) + '_data.csv', encoding='gbk')
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


        index_obj = frame_index()
        # 提取对应帧数段的目标数据
        dt_obj = obj_data.iloc[index_obj[0]:index_obj[-1] + 1]
        dt.append(dt_obj)


    # SVR进行每组的多个目标观测值进行拟合真值
    start1 = timeit.default_timer()
    def true_fit():
        Ymea = []
        Yfit = []
        Ytru = []
        for data in dt:
            df1 = pd.DataFrame(data, columns=['stamp_sec', 'frame_num', 'track_id_tru', 'track_id_mea',
                                              'nestCenter_x_tru', 'nestCenter_x_mea', 'mea_error_x', 'mea_var_x',
                                              'nestCenter_y_tru', 'nestCenter_y_mea', 'mea_error_y', 'mea_var_y',
                                              'velocity_x_tru', 'velocity_x_mea', 'mea_error_vx', 'mea_var_vx',
                                              'velocity_y_tru', 'velocity_y_mea', 'mea_error_vy', 'mea_var_vy'],
                               dtype=float)

            X = df1.iloc[:, 1]

            T1 = df1.iloc[:, 4]
            T2 = df1.iloc[:, 8]
            T3 = df1.iloc[:, 12]
            T4 = df1.iloc[:, 16]

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
            sc_t1  = StandardScaler()
            sc_t2 = StandardScaler()
            sc_t3 = StandardScaler()
            sc_t4 = StandardScaler()

            X = sc_x.fit_transform(np.array(X).reshape(-1, 1))

            T1 = sc_t1.fit_transform(np.array(T1).reshape(-1, 1))
            T2 = sc_t2.fit_transform(np.array(T2).reshape(-1, 1))
            T3 = sc_t3.fit_transform(np.array(T3).reshape(-1, 1))
            T4 = sc_t4.fit_transform(np.array(T4).reshape(-1, 1))

            Y1 = sc_y1.fit_transform(np.array(Y1).reshape(-1, 1))
            Y2 = sc_y2.fit_transform(np.array(Y2).reshape(-1, 1))
            Y3 = sc_y3.fit_transform(np.array(Y3).reshape(-1, 1))
            Y4 = sc_y4.fit_transform(np.array(Y4).reshape(-1, 1))

            # 进行回归训练
            SVR1.fit(X, Y1)
            print(SVR1.best_estimator_.support_vectors_)
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
            T1 = sc_t1.inverse_transform(T1.reshape(-1, 1))
            y_fit1 = sc_y1.inverse_transform(y_fit1.reshape(-1, 1))

            Y2 = sc_y2.inverse_transform(Y2.reshape(-1, 1))
            T2 = sc_t2.inverse_transform(T2.reshape(-1, 1))
            y_fit2 = sc_y2.inverse_transform(y_fit2.reshape(-1, 1))

            Y3 = sc_y3.inverse_transform(Y3.reshape(-1, 1))
            T3 = sc_t3.inverse_transform(T3.reshape(-1, 1))
            y_fit3 = sc_y3.inverse_transform(y_fit3.reshape(-1, 1))

            Y4 = sc_y4.inverse_transform(Y4.reshape(-1, 1))
            T4 = sc_t4.inverse_transform(T4.reshape(-1, 1))
            y_fit4 = sc_y4.inverse_transform(y_fit4.reshape(-1, 1))

            Ymea.append([Y1, Y2, Y3, Y4])

            Ytru.append([T1, T2, T3, T4])
            # y_fit1为nestCenter.x拟合真值， y_fit2为nestCenter.y拟合真值，y_fit3为velocity.x拟合真值， y_fit4为velocity.y拟合真值
            Yfit.append([y_fit1, y_fit2, y_fit3, y_fit4])

        return Ymea, Ytru, Yfit


    Ymea, Ytru, Yfit = true_fit()
    end1 = timeit.default_timer()
    jiange = end1-start1
    print('本次SVR回归拟合时间: ', jiange)

    
    # 观测误差均值写入
    work_mea_err_mean = pd.ExcelWriter('../SVR_Fit_Error_Analysis/test'
                                       + str(i4 + 1) + '/mea_err/mea_err' + str(i4 + 1) + '.xlsx')
    # 相对误差均值写入
    work_rel_err_mean = pd.ExcelWriter('../SVR_Fit_Error_Analysis/test'
                                       + str(i4 + 1) + '/rel_err/rel_err' + str(i4 + 1) + '.xlsx')

    for i8 in range(0, obj_num):
        df2 = pd.DataFrame(dt[i8], columns=['stamp_sec', 'frame_num', 'track_id_tru', 'track_id_mea',
                                            'nestCenter_x_tru', 'nestCenter_x_mea', 'mea_error_x', 'mea_var_x',
                                            'nestCenter_y_tru', 'nestCenter_y_mea', 'mea_error_y', 'mea_var_y',
                                            'velocity_x_tru', 'velocity_x_mea', 'mea_error_vx', 'mea_var_vx',
                                            'velocity_y_tru', 'velocity_y_mea', 'mea_error_vy', 'mea_var_vy'],
                           dtype=float)
        # nestCenter.x的真值，观测值，观测误差，相对误差
        tru_nCx = np.array(df2.iloc[:50, 4])
        mea_nCx = np.array((df2.iloc[:50, 5]))
        mea_nCx_err = np.array(df2.iloc[:50, 6])
        rel_nCx_err = (mea_nCx_err / tru_nCx) * 100
        # nestCenter.y的真值，观测值，观测误差，相对误差
        tru_nCy = np.array(df2.iloc[:50, 8])
        mea_nCy = np.array(df2.iloc[:50, 9])
        mea_nCy_err = np.array(df2.iloc[:50, 10])
        rel_nCy_err = (mea_nCy_err / tru_nCy) * 100
        # velocity.x的真值，观测值，观测误差，相对误差
        tru_vx = np.array(df2.iloc[:50, 12])
        mea_vx = np.array(df2.iloc[:50, 13])
        mea_vx_err = np.array(df2.iloc[:50, 14])
        rel_vx_err = (mea_vx_err / tru_vx) * 100
        # velocity.y的真值，观测值，观测误差，相对误差
        tru_vy = np.array(df2.iloc[:50, 16])
        mea_vy = np.array(df2.iloc[:, 17])
        mea_vy_err = np.array(df2.iloc[:50, 18])
        rel_vy_err = (mea_vy_err / tru_vy) * 100
        # nestCenter.x的拟合真值，观测误差，相对误差
        fit_nCx = np.concatenate(Yfit[i8][0])
        fit_nCx_err = np.concatenate(Ymea[i8][0] - Yfit[i8][0])
        rel_fit_nCx_err = np.concatenate((Ymea[i8][0] - Yfit[i8][0]) / Yfit[i8][0]) * 100
        # nestCenter.y的拟合真值，观测误差，相对误差
        fit_nCy = np.concatenate(Yfit[i8][1])
        fit_nCy_err = np.concatenate(Ymea[i8][1] - Yfit[i8][1])
        rel_fit_nCy_err = np.concatenate((Ymea[i8][1] - Yfit[i8][1]) / Yfit[i8][1]) * 100
        # velocity.x的拟合真值，观测误差，相对误差
        fit_vx = np.concatenate(Yfit[i8][2])
        fit_vx_err = np.concatenate(Ymea[i8][2] - Yfit[i8][2])
        rel_fit_vx_err = np.concatenate((Ymea[i8][2] - Yfit[i8][2]) / Yfit[i8][2]) * 100
        # velocity.y的拟合真值，观测误差，相对误差
        fit_vy = np.concatenate(Yfit[i8][3])
        fit_vy_err = np.concatenate(Ymea[i8][3] - Yfit[i8][3])
        rel_fit_vy_err = np.concatenate((Ymea[i8][3] - Yfit[i8][3]) / Yfit[i8][3]) * 100

        # 观测误差可视化
        plt.figure(figsize=(16, 10))
        plt.subplot(2, 2, 1)
        plt.title('ID' + str(test_group_id[i4][i8]) + ' nestCenter.x观测误差', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.plot(mea_nCx_err, 'b', label='观测误差(CA)', marker='v')
        plt.plot(fit_nCx_err, 'g', label='观测误差(fit)', marker='v')
        plt.xlabel('帧(frame_num)', fontsize=12, loc='right')
        plt.ylabel('距离误差(m)', fontsize=12, loc='top')
        plt.legend(fontsize=10)

        plt.subplot(2, 2, 2)
        plt.title('ID' + str(test_group_id[i4][i8]) + ' nestCenter.y观测误差', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.plot(mea_nCy_err, 'b', label='观测误差(CA)', marker='v')
        plt.plot(fit_nCy_err, 'g', label='观测误差(fit)', marker='v')
        plt.xlabel('帧(frame_num)', fontsize=12, loc='right')
        plt.ylabel('距离误差(m)', fontsize=12, loc='top')
        plt.legend(fontsize=10)

        plt.subplot(2, 2, 3)
        plt.title('ID' + str(test_group_id[i4][i8]) + ' velocity.x观测误差', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.plot(mea_vx_err, 'b', label='观测误差(CA)', marker='v')
        plt.plot(fit_vx_err, 'g', label='观测误差(fit)', marker='v')
        plt.xlabel('帧(frame_num)', fontsize=12, loc='right')
        plt.ylabel('速度误差(m/s)', fontsize=12, loc='top')
        plt.legend(fontsize=10)

        plt.subplot(2, 2, 4)
        plt.title('ID' + str(test_group_id[i4][i8]) + ' velocity.y观测误差', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.plot(mea_vy_err, 'b', label='观测误差(CA)', marker='v')
        plt.plot(fit_vy_err, 'g', label='观测误差(fit)', marker='v')
        plt.xlabel('帧(frame_num)', fontsize=12, loc='right')
        plt.ylabel('速度误差(m/s)', fontsize=12, loc='top')
        plt.legend(fontsize=10)
        plt.savefig(r'../SVR_Fit_Error_Analysis/test' + str(
            i4 + 1) + '/mea_err/ID' + str(test_group_id[i4][i8]) + '.svg', dpi=500, bbox_inches='tight')
        plt.close()
        plt.show()
        '''观测误差均值写入'''
        data_mea_err_mean = pd.DataFrame({'nCx_mea_tru_err_mean':np.mean(mea_nCx_err), 'nCx_mea_fit_err_mean':np.mean(fit_nCx_err),
                                          'nCy_mea_tru_err_mean':np.mean(mea_nCy_err), 'nCy_mea_fit_err_mean':np.mean(fit_nCy_err),
                                          'vx_mea_tru_err_mean':np.mean(mea_vx_err), 'vx_mea_fit_err_mean':np.mean(fit_vx_err),
                                          'vy_mea_tru_err_mean':np.mean(mea_vy_err), 'vy_mea_fit_err_mean':np.mean(fit_vy_err)}, index=[0])
        data_mea_err_mean.to_excel(work_mea_err_mean, sheet_name='ID' + str(test_group_id[i4][i8]), index=False)

        # 观测值、真值、SVR拟合真值对比
        plt.figure(figsize=(16, 10))
        plt.subplot(2, 2, 1)
        plt.title('ID' + str(test_group_id[i4][i8]) + ' nestCenter.x状态拟合', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.plot(mea_nCx, 'orange', label='观测值', marker='s')
        plt.plot(tru_nCx, 'royalblue', label='真实值(CA)', marker='o')
        plt.plot(fit_nCx, 'cornflowerblue', label='真实值(fit)', marker='o')
        plt.xlabel('帧(frame_num)', fontsize=12, loc='right')
        plt.ylabel('距离(m)', fontsize=12, loc='top')
        plt.legend(fontsize=10)

        plt.subplot(2, 2, 2)
        plt.title('ID' + str(test_group_id[i4][i8]) + ' nestCenter.y状态拟合', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.plot(mea_nCy, 'orange', label='观测值', marker='s')
        plt.plot(tru_nCy, 'royalblue', label='真实值(CA)', marker='o')
        plt.plot(fit_nCy, 'cornflowerblue', label='真实值(fit)', marker='o')
        plt.xlabel('帧(frame_num)', fontsize=12, loc='right')
        plt.ylabel('距离(m)', fontsize=12, loc='top')
        plt.legend(fontsize=10)

        plt.subplot(2, 2, 3)
        plt.title('ID' + str(test_group_id[i4][i8]) + ' velocity.x状态拟合', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.plot(mea_vx, 'orange', label='观测值', marker='s')
        plt.plot(tru_vx, 'royalblue', label='真实值(CA)', marker='o')
        plt.plot(fit_vx, 'cornflowerblue', label='真实值(fit)', marker='o')
        plt.xlabel('帧(frame_num)', fontsize=12, loc='right')
        plt.ylabel('速度(m/s)', fontsize=12, loc='top')
        plt.legend(fontsize=10)

        plt.subplot(2, 2, 4)
        plt.title('ID' + str(test_group_id[i4][i8]) + ' velocity.y状态拟合', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.plot(mea_vy, 'orange', label='观测值', marker='s')
        plt.plot(tru_vy, 'royalblue', label='真实值(CA)', marker='o')
        plt.plot(fit_vy, 'cornflowerblue', label='真实值(fit)', marker='o')
        plt.xlabel('帧(frame_num)', fontsize=12, loc='right')
        plt.ylabel('速度(m/s)', fontsize=12, loc='top')
        plt.legend(fontsize=10)
        plt.savefig(r'../SVR_Fit_Error_Analysis/test' + str(
            i4 + 1) + '/fit_result/ID' + str(test_group_id[i4][i8]) + '.svg', dpi=500, bbox_inches='tight')
        plt.close()
        plt.show()

        # 相对误差可视化
        plt.figure(figsize=(16, 10))
        plt.subplot(2, 2, 1)
        plt.title('ID' + str(test_group_id[i4][i8]) + ' nestCenter.x相对误差', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.plot(rel_nCx_err, 'r', label='相对误差(CA)', marker='v')
        plt.plot(rel_fit_nCx_err, 'y', label='相对误差(fit)', marker='v')
        plt.xlabel('帧(frame_num)', fontsize=12, loc='right')
        plt.ylabel('距离相对误差(%)', fontsize=12, loc='top')
        plt.legend(fontsize=10)

        plt.subplot(2, 2, 2)
        plt.title('ID' + str(test_group_id[i4][i8]) + ' nestCenter.y相对误差', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.plot(rel_nCy_err, 'r', label='相对误差(CA)', marker='v')
        plt.plot(rel_fit_nCy_err, 'y', label='相对误差(fit)', marker='v')
        plt.xlabel('帧(frame_num)', fontsize=12, loc='right')
        plt.ylabel('距离相对误差(%)', fontsize=12, loc='top')
        plt.legend(fontsize=10)

        plt.subplot(2, 2, 3)
        plt.title('ID' + str(test_group_id[i4][i8]) + ' velocity.x相对误差', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.plot(rel_vx_err, 'r', label='相对误差(CA)', marker='v')
        plt.plot(rel_fit_vx_err, 'y', label='相对误差(fit)', marker='v')
        plt.xlabel('帧(frame_num)', fontsize=12, loc='right')
        plt.ylabel('速度相对误差(%)', fontsize=12, loc='top')
        plt.legend(fontsize=10)

        plt.subplot(2, 2, 4)
        plt.title('ID' + str(test_group_id[i4][i8]) + ' velocity.y相对误差', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.plot(rel_vy_err, 'r', label='相对误差(CA)', marker='v')
        plt.plot(rel_fit_vy_err, 'y', label='相对误差(fit)', marker='v')
        plt.xlabel('帧(frame_num)', fontsize=12, loc='right')
        plt.ylabel('速度相对误差(%)', fontsize=12, loc='top')
        plt.legend(fontsize=10)
        plt.savefig(r'../SVR_Fit_Error_Analysis/test' + str(
            i4 + 1) + '/rel_err/ID' + str(test_group_id[i4][i8]) + '.svg', dpi=500, bbox_inches='tight')
        plt.close()
        plt.show()

        '''相对误差均值写入'''
        data_rel_err_mean = pd.DataFrame({'nCx_rel_err_mean': np.mean(rel_nCx_err), 'nCx_rel_fit_err_mean': np.mean(rel_fit_nCx_err),
                                          'nCy_rel_err_mean': np.mean(rel_nCy_err), 'nCy_rel_fit_err_mean': np.mean(rel_fit_nCy_err),
                                          'vx_rel_err_mean': np.mean(rel_vx_err),  'vx_rel_fit_err_mean': np.mean(rel_fit_vx_err),
                                          'vy_rel_err_mean': np.mean(rel_vy_err),  'vy_rel_fit_err_mean': np.mean(rel_fit_vy_err)}, index=[0])
        data_rel_err_mean.to_excel(work_rel_err_mean, sheet_name='ID' + str(test_group_id[i4][i8]), index=False)


    work_mea_err_mean.save()
    work_rel_err_mean.save()

    print('第' + str(i4 + 1) + '组样本数据误差分析完成')
    print('-------------------')

end = timeit.default_timer()
print('Running time: %s Seconds' % (end - start))
print('测试结束')





