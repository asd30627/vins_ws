import math


def stamp_to_filename(sec: int, nsec: int) -> str:
    return f"{int(sec)}_{int(nsec):09d}"


def normalize_quaternion(qx: float, qy: float, qz: float, qw: float):
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-12:
        return 0.0, 0.0, 0.0, 1.0
    return qx / norm, qy / norm, qz / norm, qw / norm


def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float):
    qx, qy, qz, qw = normalize_quaternion(qx, qy, qz, qw)

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)

    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)

    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)

    return [
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22],
    ]


def pose_to_matrix_row(px: float, py: float, pz: float,
                       qx: float, qy: float, qz: float, qw: float):
    r = quaternion_to_rotation_matrix(qx, qy, qz, qw)

    return [
        r[0][0], r[0][1], r[0][2], px,
        r[1][0], r[1][1], r[1][2], py,
        r[2][0], r[2][1], r[2][2], pz,
    ]


def quaternion_to_yaw_deg(qx: float, qy: float, qz: float, qw: float) -> float:
    qx, qy, qz, qw = normalize_quaternion(qx, qy, qz, qw)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(yaw)