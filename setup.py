from setuptools import setup

setup(
    name='Mean_Initialized_Noise_Covariance_Matrix',
    version='1.0',
    packages=['Code', 'Data'],
    include_package_data=True,
    # exclude_package_date={'':['.gitignore']},
    install_requires=[   # 依赖列表
       'numpy>=1.23.5',
       'matplotlib>=3.7.1',
       'pandas>=1.5.3',
       'scikit-learn>=1.2.2'
   ]
)