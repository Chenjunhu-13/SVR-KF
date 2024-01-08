# Observation noise covariance matrix initialization-based objective state estimation for Kalman filter using SVR

### **[Observation noise covariance matrix initialization-based objective state estimation for Kalman filter using SVR](https://link.springer.com/chapter/10.1007/978-981-99-1252-0_10)**

## Abstract

Kalman filtering is a filtering algorithm for optimal estimation of the system state. The optimal estimate of the system state is obtained through iteratively updating the mean and variance of the system by establishing the equations of motion and observation equations for the system. In this paper, the Kalman filtering algorithm is used to estimate the state of multiple targets which is collected by vehicle radar. Since the true values are not accurate enough, Support Vector Regression (SVR) is applied to fit the multi-target data collected by the radar to obtain the approximate true values. Then, the observation error and the sample variance of the observation error are calculated. Finally, the mean of the variances of the different physical quantities is formed into a diagonal matrix as the initialized value of the observation noise covariance matrix Rk. The Kalman filtering of mean initialized Rk is then compared with the empirically initialized Rk for state estimation, it is found that our method can improve the accuracy of the Kalman filtering for state estimation of the target.

## Introduction

We have implemented a Kalman filtering algorithm with mean initialized noise covariance

## Usage

Preprocess sensor data, truth data, and correlation data to obtain the IDs, corresponding timestamps, and physical quantities of all sensor data

```
run Data_Processing.py
```

Use conventional Kalman filtering to estimate the state of multiple targets in front of the vehicle. By calculating the error between the laser radar measurement value (true value) and the millimeter wave radar measurement value, the error variance is calculated to approximate the observation noise covariance matrix. Use 0.5, 0.1, 0.05, 0.01 experience to initialize the observation noise covariance matrix and compare it with the mean initialization.

```
run General_Mean_Init_Kalman_Filtering.py
```

## Experimental Result

![image-20240108223259331](C:\Users\cjh\AppData\Roaming\Typora\typora-user-images\image-20240108223259331.png)![image-20240108223306748](C:\Users\cjh\AppData\Roaming\Typora\typora-user-images\image-20240108223306748.png)![image-20240108223312596](C:\Users\cjh\AppData\Roaming\Typora\typora-user-images\image-20240108223312596.png)![image-20240108223317213](C:\Users\cjh\AppData\Roaming\Typora\typora-user-images\image-20240108223317213.png)