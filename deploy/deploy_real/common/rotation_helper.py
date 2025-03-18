import numpy as np
from scipy.spatial.transform import Rotation as R


def get_gravity_orientation(quaternion):
    # get directional vector or projection of gravity vector in a local frame
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def transform_imu_data(waist_yaw, waist_yaw_omega, imu_quat, imu_omega):
    # R_torso_pelvis = R.from_euler("z", waist_yaw).as_matrix()
    # R_torso = R.from_quat([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]]).as_matrix()  
    # R_pelvis = np.dot(R_torso_pelvis.T, R_torso)  
    # w_pelvis = np.dot(R_torso_pelvis, imu_omega) - np.array([0, 0, waist_yaw_omega])
    # return R.from_matrix(R_pelvis).as_quat()[[3, 0, 1, 2]], w_pelvis

    R_torso_pelvis = R.from_euler("z", waist_yaw).as_matrix()
    R_pelvis = R.from_quat([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]]).as_matrix()  
    R_torso = np.dot(R_pelvis, R_torso_pelvis) 
    w_torso = np.dot(R_torso.T, imu_omega) - np.array([0, 0, waist_yaw_omega])
    return R.from_matrix(R_torso).as_quat()[[3, 0, 1, 2]], w_torso


def quatToEuler(quat):
    eulerVec = np.zeros(3)
    qw = quat[0] 
    qx = quat[1] 
    qy = quat[2]
    qz = quat[3]
    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    eulerVec[0] = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        eulerVec[1] = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        eulerVec[1] = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    eulerVec[2] = np.arctan2(siny_cosp, cosy_cosp)
    
    return eulerVec
