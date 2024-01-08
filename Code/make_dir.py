import os
import shutil


def create_dir(path, name, batch_name=None):
    if batch_name:
        for ele in batch_name:
            path_id = path + name + str(ele)
            sExists = os.path.exists((path_id))
            if sExists:
                shutil.rmtree(path_id)
                os.makedirs(path_id)
            else:
                os.makedirs(path_id)
            print("%s 目录创建成功" % (name + str(ele)))
    else:
        path_id = path + name
        sExists = os.path.exists(path_id)
        if sExists:
            shutil.rmtree(path_id)
            os.makedirs(path_id)
            print("%s 目录创建成功" % (name))
        else:
            os.makedirs(path_id)
            print("%s 目录创建成功" % (name))
    return print("文件夹已创建完成!")