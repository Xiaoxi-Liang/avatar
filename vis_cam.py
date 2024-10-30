import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D






def call_x_rot(angle):
    theta = np.deg2rad(angle)  # 将角度转为弧度
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    return R_x

def call_y_rot(angle):
    theta = np.deg2rad(angle)  # 将角度转为弧度
    R_y = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    return R_y

def get_rotated_poses(pose):


    x_angle =[-60, -50, -40, -30,-20,-10,10,20,30,40,50,60]
    y_angle =[-60, -50, -40, -30,-20,-10,10,20,30,40,50,60]
    # x_angle =[-90,-60,-30,30,60,90]
    # y_angle =[-90,-60,-30,30,60,90]

    i = 0
    poses=[]
    for angle in x_angle:
        i += 1
        R_x = call_x_rot(angle)

        W2C = np.eye(4)
        W2C[:3, :3] = R_x @ np.linalg.inv(pose)[:3, :3]
        W2C[:3, 3] = np.linalg.inv(pose)[:3, 3]
        poses.append(np.linalg.inv(W2C))

    for angle in y_angle:
        i += 1
        R_y = call_y_rot(angle)
        W2C = np.eye(4)
        W2C[:3, :3] = R_y @ np.linalg.inv(pose)[:3, :3]
        W2C[:3, 3] = np.linalg.inv(pose)[:3, 3]
        poses.append(np.linalg.inv(W2C))

    return poses

def plot_camera(ax, c2w, idx, lookat, scale=0.7, label=0):
    origin = c2w[:3, 3]
    R = c2w[:3, :3]
    
    x_axis = R[:, 0] * scale
    y_axis = R[:, 1] * scale
    z_axis = R[:, 2] * scale

    if label == 0:
        ax.quiver(*origin, *x_axis, color='red',  label='Right Vector' if idx == 0 else "")
        ax.quiver(*origin, *y_axis, color='green', label='Up Vector' if idx == 0 else "")
        ax.quiver(*origin, *z_axis, color='blue', label='Forward Vector' if idx == 0 else "")
    
    elif label==1:
        ax.plot([origin[0], origin[0] + x_axis[0]],
            [origin[1], origin[1] + x_axis[1]],
            [origin[2], origin[2] + x_axis[2]],
            linestyle='--', color='red', alpha=0.3,
            label='Right Vector' if idx == 0 else "")
    
        ax.plot([origin[0], origin[0] + y_axis[0]],
                [origin[1], origin[1] + y_axis[1]],
                [origin[2], origin[2] + y_axis[2]],
                linestyle='--', color='green', alpha=0.3,
                label='Up Vector' if idx == 0 else "")

        ax.plot([origin[0], origin[0] + z_axis[0]],
                [origin[1], origin[1] + z_axis[1]],
                [origin[2], origin[2] + z_axis[2]],
                linestyle='--', color='blue', alpha=0.3,
                label='Forward Vector' if idx == 0 else "")

    elif label == 2:
        double_dash = (5, 2, 1, 2)  # (线长, 间隔, 线长, 间隔)

        ax.plot([origin[0], origin[0] + x_axis[0]],
                [origin[1], origin[1] + x_axis[1]],
                [origin[2], origin[2] + x_axis[2]],
                color='red', alpha=0.3, dashes=double_dash,
                label='Right Vector' if idx == 0 else "")
        
        ax.plot([origin[0], origin[0] + y_axis[0]],
                [origin[1], origin[1] + y_axis[1]],
                [origin[2], origin[2] + y_axis[2]],
                color='green', alpha=0.3, dashes=double_dash,
                label='Up Vector' if idx == 0 else "")

        ax.plot([origin[0], origin[0] + z_axis[0]],
                [origin[1], origin[1] + z_axis[1]],
                [origin[2], origin[2] + z_axis[2]],
                color='blue', alpha=0.3, dashes=double_dash,
                label='Forward Vector' if idx == 0 else "")
    # ax.quiver(*origin, *x_axis, color='red', alpha=0.3, label='Right Vector' if idx == 0 else "")
    # ax.quiver(*origin, *y_axis, color='green',alpha=0.3, label='Up Vector' if idx == 0 else "")
    # ax.quiver(*origin, *z_axis, color='blue',alpha=0.3, label='Forward Vector' if idx == 0 else "")
    # ax.text(*origin, str(idx + 1), color='black')

    # # # Draw dashed line to lookat point
    # ax.plot([origin[0], lookat[0]], [origin[1], lookat[1]], [origin[2], lookat[2]], 
    #         linestyle='--', color='gray')


def plot_world(ax, scale=1):
    c2w = np.eye(4)
    origin = c2w[:3, 3]
    R = c2w[:3, :3]
    
    x_axis = R[:, 0] * scale
    y_axis = R[:, 1] * scale
    z_axis = R[:, 2] * scale


    ax.quiver(*origin, *x_axis, color='red' )
    ax.quiver(*origin, *y_axis, color='green')
    ax.quiver(*origin, *z_axis, color='blue')

from tqdm import tqdm

def visualize_cameras(c2w_matrices1, c2w_matrices2=None, c2w_matrices3=None,lookat=np.array([0, 0, 0]), filename='camera_visualization.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot world origin
    ax.scatter(0, 0, 0, color='black', label='World Origin')
    plot_world(ax)

    for idx, c2w in enumerate(tqdm(c2w_matrices1)):
        plot_camera(ax, c2w, idx, lookat, label=0)

    if c2w_matrices2:
        for idx, c2w in enumerate(tqdm(c2w_matrices2)):
            plot_camera(ax, c2w, 1, lookat, label=1)

    if c2w_matrices3:
        for idx, c2w in enumerate(tqdm(c2w_matrices3)):
            plot_camera(ax, c2w, 1, lookat, label=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-4, 4) 
    ax.legend(loc='upper right')

    # 保存图像
    plt.savefig(filename)

# 示例用法，您可以根据需要调整矩阵


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

#############################################################################
from dreifus.camera import CameraCoordinateConvention, PoseType
from dreifus.image import Img
from dreifus.matrix import Pose
from dreifus.trajectory import circle_around_axis
from dreifus.vector import Vec3

move_z = 4.4

poses = circle_around_axis(96, up=Vec3(0, 1, 0), move=Vec3(0, 0, move_z), distance=0.3 * move_z / 2.7,
                                theta_to=2 * np.pi)
poses = [np.array(pose) for pose in poses]


###########################################################################
eg3d_poses = list(np.load('eg3d_cam.npy'))
#########################################################################
p4d_poses = list(np.load('portrait4d.npy'))
p4d_poses = [pose@gl2cv for pose in p4d_poses]





lookat_point = np.array([0, 0, 0])
round_poses = get_rotated_poses(c2w_matrices[0])

# visualize_cameras(c2w_matrices[:10], eg3d_poses, round_poses, lookat=lookat_point, filename='cameras.png')
visualize_cameras(round_poses, p4d_poses, lookat=lookat_point, filename='cameras_p4d.png')