import numpy as np

# Rotation matrices
def rotation_x(theta):
    """Rotation around the X-axis"""

    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])

def rotation_z(theta):
    """Rotation around the X-axis"""

    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])

def transformation(position, fixed_x_angle, joint_angle=0):

    # Create a 4x4 identity matrix
    T = np.eye(4)

    # Calculate the rotation matrix
    R = (
        rotation_x(fixed_x_angle)
        @ rotation_z(joint_angle)
    )

    # Insert rotation into top-left 3x3 block
    T[:3, :3] = R

    # Insert translation into the last column
    T[:3, 3] = position

    return T

# Franka Panda home joint angle

q = np.array([
    0,
    0,
    0,
    -1.57059,
    0,
    1.57079,
    -0.7853
])

# Define the seven joint transforms
pi = np.pi

joint_parameters = [
    ([0, 0, 0.333], 0),
    ([0, 0, 0], -pi / 2),
    ([0, -0.316, 0], pi / 2),
    ([0.0825, 0, 0], pi / 2),
    ([-0.0825, 0.384, 0], -pi / 2),
    ([0, 0, 0], pi / 2),
    ([0.088, 0, 0], pi / 2)
]

# Multiply transformation matrices
T_total = np.eye(4)

for i, (pos, fixed_rotation) in enumerate(joint_parameters):
    T_joint = transformation(
        pos,
        fixed_rotation,
        q[i]
    )

    T_total = T_total @ T_joint

    print(f"\nTransformation after Joint {i+1}:")
    print(np.round(T_total, 4))

# Add the fixed hand translation
T_hand = transformation(
    [0, 0, 0.107],
    0
)

T_total = T_total @ T_hand

# Extract hand position
hand_pos = T_total[:3, 3]

print("\nMANUALLY CALCULATED HAND POSITION")

print(f"X: {hand_pos[0]:.4f} m")
print(f"Y: {hand_pos[1]:.4f} m")
print(f"Z: {hand_pos[2]:.4f} m")

# Compare with previous MuJoCo output
mujoco_pos = np.array([
    0.5545,
    0.0000,
    0.6245
])

error = np.linalg.norm(
    hand_pos - mujoco_pos
)

print("\nVERIFICATION")
print(f"Position error: {error:.8f} m")