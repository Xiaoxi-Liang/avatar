import numpy as np
from scipy.spatial.transform import Rotation as R


def average_cam2world(matrices):
    # 提取旋转和平移
    rotations = [matrix[:3, :3] for matrix in matrices]
    translations = [matrix[:3, 3] for matrix in matrices]
    
    # 平均化平移
    avg_translation = np.mean(translations, axis=0)
    
    # 使用四元数平均化旋转
    quaternions = [R.from_matrix(rot).as_quat() for rot in rotations]
    avg_quaternion = np.mean(quaternions, axis=0)
    avg_rotation = R.from_quat(avg_quaternion).as_matrix()
    
    # 构建平均化的cam2world矩阵
    avg_matrix = np.eye(4)
    avg_matrix[:3, :3] = avg_rotation
    avg_matrix[:3, 3] = avg_translation
    
    return avg_matrix

# make sure all use opencv cooridinate
gl2cv = np.array([
    [1,  0, 0, 0],
    [ 0, -1, 0, 0],
    [ 0,  0, -1, 0],
    [ 0,  0, 0, 1]
])
import json
filename = '/nas5/xiaoxi/dataset/3Dtalk/May/transforms_train.json'
with open(filename, 'r') as file:
    # 使用 json.load() 函数读取文件内容并转换为Python字典或列表
    data = json.load(file)
frames = data['frames']
c2w_matrices = [np.array(frame['transform_matrix'])@gl2cv for frame in frames]

ac2w = average_cam2world(c2w_matrices)
print(ac2w)