import numpy as np
import warnings
from scipy.spatial.transform import Rotation

def quat_to_euler(quat, degrees=False, normalize=True, ignore_warnings=True):
    with warnings.catch_warnings():
        if ignore_warnings:
            warnings.simplefilter("ignore", category=UserWarning)
        R = Rotation(quat.T, normalize=normalize)
        return R.as_euler('ZYX', degrees=degrees).T

def euler_to_quat(yaw, pitch, roll, degrees=False):
    angles = np.asarray([yaw, pitch, roll], dtype=float) / 2.
    if degrees:
        angles = np.deg2rad(angles)

    cos = np.cos(angles)
    sin = np.sin(angles)

    yaw_pitch = np.vstack((
        cos[0] * cos[1],
        cos[0] * sin[1],
        sin[0] * cos[1],
        sin[0] * sin[1]
    ))
    roll = np.vstack((cos[-1], sin[-1]))

    yaw_pitch_roll = np.einsum('ik,jk->ijk', yaw_pitch, roll)

    q = np.vstack((
        yaw_pitch_roll[0,1] - yaw_pitch_roll[3,0],
        yaw_pitch_roll[1,0] + yaw_pitch_roll[2,1],
        yaw_pitch_roll[2,0] - yaw_pitch_roll[1,1],
        yaw_pitch_roll[0,0] + yaw_pitch_roll[3,1]
    ))

    if angles.ndim == 1:
        return q[:,0]

    return q
