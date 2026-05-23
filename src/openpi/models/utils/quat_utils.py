import jax
import jax.numpy as jnp

def quat_multiply(q1, q2):
    """Multipliserer to kvaternions (xyzw)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return jnp.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def get_deltas(states, dt):
    """
    states: array av form (N, 7) -> [x, y, z, qx, qy, qz, qw]
    dt: tidssteg mellom målinger
    """
    pos = states[:, :3]
    quats = states[:, 3:]

    # 1. Lineær hastighet (v = dx/dt)
    # jnp.diff regner ut states[i+1] - states[i]
    v = jnp.diff(pos, axis=0, prepend=0.0) / dt

    # 2. Angulær hastighet
    q_t = quats[:-1]     # q_nå
    q_next = quats[1:]   # q_neste

    # Invers av q_t er konjugatet [ -qx, -qy, -qz, qw ]
    q_inv = q_t * jnp.array([-1, -1, -1, 1])

    # Finn rotasjonsendringen: dq = q_next * q_inv
    # Vi bruker vmap for å multiplisere alle parene samtidig
    dq = jax.vmap(quat_multiply)(q_next, q_inv)

    # Beregn angulær hastighet (omega)
    # Enkel tilnærming: w = 2 * imag(dq) / dt
    # (Dette antar små rotasjoner mellom stegene)
    # omega = (2.0 * dq[:, :3]) / dt

    return v, dq

def quatvel_to_angvel(quat, dquat):
    """
    Konverterer kvaternionhastighet (dquat) til angulær hastighet (omega).
    quat: array av form (N, 4) -> [qx, qy, qz, qw]
    dquat: array av form (N, 4) -> [dqx, dqy, dqz, dqw]
    """
    qx, qy, qz, qw = quat
    dqx, dqy, dqz, dqw = dquat

    # Beregn angulær hastighet
    omega_x = 2 * (dqw * qx + dqy * qz - dqz * qy)
    omega_y = 2 * (dqw * qy + dqz * qx - dqx * qz)
    omega_z = 2 * (dqw * qz + dqx * qy - dqy * qx)

    return jnp.array([omega_x, omega_y, omega_z])

def quaternion_to_rpy(x, y, z, w, degrees=False):
    """
    Convert quaternion [x, y, z, w] to roll, pitch, yaw.

    roll  = rotation around X
    pitch = rotation around Y
    yaw   = rotation around Z

    Returns:
        jnp.array([roll, pitch, yaw])
    """

    q = jnp.array([x, y, z, w])
    q = q / jnp.linalg.norm(q)

    x, y, z, w = q

    # Roll: X axis
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = jnp.arctan2(sinr_cosp, cosr_cosp)

    # Pitch: Y axis
    sinp = 2.0 * (w * y - z * x)

    # JAX-safe replacement for:
    # if abs(sinp) >= 1: pitch = ±pi/2
    sinp = jnp.clip(sinp, -1.0, 1.0)
    pitch = jnp.arcsin(sinp)

    # Yaw: Z axis
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = jnp.arctan2(siny_cosp, cosy_cosp)

    rpy = jnp.array([roll, pitch, yaw])

    if degrees:
        rpy = jnp.rad2deg(rpy)

    return rpy

import jax
import jax.numpy as jnp


# -----------------------------
# Basic quaternion operations
# -----------------------------

def quat_normalize(q, eps=1e-8):
    """
    q: [..., 4], format [x, y, z, w]
    """
    return q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + eps)
    


def quat_conj(q):
    """
    Quaternion conjugate.
    q: [..., 4], format [w, x, y, z]
    """
    return jnp.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def quat_inv(q):
    """
    Quaternion inverse.
    For unit quaternions this is just the conjugate.
    """
    q = quat_normalize(q)
    return quat_conj(q)


def quat_mul(q1, q2):
    """
    Hamilton product q = q1 ⊗ q2.
    q1, q2: [..., 4], format [w, x, y, z]
    """
    w1, x1, y1, z1 = jnp.split(q1, 4, axis=-1)
    w2, x2, y2, z2 = jnp.split(q2, 4, axis=-1)

    return jnp.concatenate([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def quat_shortest(q):
    """
    Enforce q and -q ambiguity by choosing q with positive scalar part.
    This avoids discontinuous orientation errors.
    """
    return jnp.where(q[..., :1] < 0.0, -q, q)


def euler_rpy_to_quat(euler):
    """
    Convert roll-pitch-yaw Euler angles to quaternion.

    euler: [..., 3], [roll, pitch, yaw]
    returns: [..., 4], quaternion [w, x, y, z]

    Convention:
        roll  about x
        pitch about y
        yaw   about z

    Equivalent to R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    roll = euler[..., 0:1]
    pitch = euler[..., 1:2]
    yaw = euler[..., 2:3]

    cr = jnp.cos(roll * 0.5)
    sr = jnp.sin(roll * 0.5)
    cp = jnp.cos(pitch * 0.5)
    sp = jnp.sin(pitch * 0.5)
    cy = jnp.cos(yaw * 0.5)
    sy = jnp.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    q = jnp.concatenate([x, y, z, w], axis=-1)
    return quat_normalize(q)

def quat_to_euler_rpy(q, eps=1e-8):
    """
    Convert quaternion to roll-pitch-yaw Euler angles.

    q: [..., 4], format [w, x, y, z]
    returns: [..., 3], [roll, pitch, yaw]

    Convention:
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    q = quat_normalize(q, eps=eps)
    w, x, y, z = jnp.split(q, 4, axis=-1)

    # roll: rotation around x
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = jnp.arctan2(sinr_cosp, cosr_cosp)

    # pitch: rotation around y
    sinp = 2.0 * (w * y - z * x)
    sinp = jnp.clip(sinp, -1.0, 1.0)
    pitch = jnp.arcsin(sinp)

    # yaw: rotation around z
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = jnp.arctan2(siny_cosp, cosy_cosp)

    return jnp.concatenate([roll, pitch, yaw], axis=-1)

def quat_error_current_minus_desired(q_hat, q_star):
    """
    Orientation error quaternion:
        q_err = q_star^{-1} ⊗ q_hat

    This represents the rotation from desired orientation to current orientation.
    Good when your residual is current - desired.
    """
    q_hat = quat_normalize(q_hat)
    q_star = quat_normalize(q_star)

    q_err = quat_mul(quat_inv(q_star), q_hat)
    q_err = quat_shortest(quat_normalize(q_err))
    return q_err

def quat_delta_to_desired(q_hat, q_star):
    """
    Corrective delta quaternion:
        q_delta = q_hat^{-1} ⊗ q_star

    This is the rotation that takes current orientation toward desired orientation.
    """
    q_hat = quat_normalize(q_hat)
    q_star = quat_normalize(q_star)

    q_delta = quat_mul(quat_inv(q_hat), q_star)
    q_delta = quat_shortest(quat_normalize(q_delta))
    return q_delta

def quat_log(q, eps=1e-8):
    """
    Quaternion logarithm for unit quaternion.

    q: [..., 4], format [w, x, y, z]
    returns: [..., 3], rotation vector theta * axis

    For small rotations:
        log(q) ≈ 2 * vector_part(q)
    """
    q = quat_shortest(quat_normalize(q, eps=eps))

    w = jnp.clip(q[..., :1], -1.0, 1.0)
    v = q[..., 1:]

    v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)

    theta = 2.0 * jnp.arctan2(v_norm, w)

    rotvec = jnp.where(
        v_norm > eps,
        theta * v / (v_norm + eps),
        2.0 * v,
    )

    return rotvec

def quat_exp(rotvec, eps=1e-8):
    """
    Exponential map from rotation vector to quaternion.

    rotvec: [..., 3]
    returns: [..., 4], quaternion [w, x, y, z]
    """
    theta = jnp.linalg.norm(rotvec, axis=-1, keepdims=True)
    half_theta = 0.5 * theta

    w = jnp.cos(half_theta)

    xyz = jnp.where(
        theta > eps,
        jnp.sin(half_theta) * rotvec / (theta + eps),
        0.5 * rotvec,
    )

    q = jnp.concatenate([w, xyz], axis=-1)
    return quat_normalize(q)

def quat_orientation_error(q_hat, q_star):
    """
    Returns 3D orientation error vector e_R.

    q_hat:  [..., 4], current / predicted quaternion
    q_star: [..., 4], desired quaternion

    Error convention:
        e_R = Log(q_star^{-1} ⊗ q_hat)

    This means:
        e_R = current orientation relative to desired orientation.
    """
    q_err = quat_error_current_minus_desired(q_hat, q_star)
    return quat_log(q_err)

def quat_to_rotmat(q, eps=1e-8):
    """
    q: [..., 4], [w, x, y, z]
    returns: [..., 3, 3]
    """
    q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + eps)

    w, x, y, z = jnp.split(q, 4, axis=-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z

    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    row0 = jnp.concatenate([
        ww + xx - yy - zz,
        2.0 * (xy - wz),
        2.0 * (xz + wy),
    ], axis=-1)

    row1 = jnp.concatenate([
        2.0 * (xy + wz),
        ww - xx + yy - zz,
        2.0 * (yz - wx),
    ], axis=-1)

    row2 = jnp.concatenate([
        2.0 * (xz - wy),
        2.0 * (yz + wx),
        ww - xx - yy + zz,
    ], axis=-1)

    return jnp.stack([row0, row1, row2], axis=-2)