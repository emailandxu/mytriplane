import numpy as np

def projection(fov=90.0, aspect_ratio=1.0, near=0.1, far=1000.0):
    # Convert the fov from degrees to radians
    fov_rad = np.deg2rad(fov)
    
    # Calculate the scale using the fov
    scale = 1.0 / np.tan(fov_rad / 2.0)
    
    # Create the projection matrix
    return np.array([[scale / aspect_ratio, 0, 0, 0],
                     [0, scale, 0, 0],
                     [0, 0, -(far+near) / (far - near), -2 * far * near / (far - near)],
                     [0, 0, -1, 0]]).astype(np.float32)

                    
def translate(x, y, z):
    return np.array([[1, 0, 0, x], 
                     [0, 1, 0, y], 
                     [0, 0, 1, z], 
                     [0, 0, 0, 1]]).astype(np.float32)

def rotate_x(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[1,  0, 0, 0], 
                     [0,  c, s, 0], 
                     [0, -s, c, 0], 
                     [0,  0, 0, 1]]).astype(np.float32)

def rotate_y(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[ c, 0, s, 0], 
                     [ 0, 1, 0, 0], 
                     [-s, 0, c, 0], 
                     [ 0, 0, 0, 1]]).astype(np.float32)

def rotate_z(a): # chat
    s, c = np.sin(a), np.cos(a)
    return np.array([[c, -s, 0, 0], 
                     [s,  c, 0, 0], 
                     [0,  0, 1, 0], 
                     [0,  0, 0, 1]]).astype(np.float32)

def scale(s):
    return np.array([[ s, 0, 0, 0], 
                     [ 0, s, 0, 0], 
                     [ 0, 0, s, 0], 
                     [ 0, 0, 0, 1]]).astype(np.float32)

def scale_xyz(x, y, z):
    return np.array([[ x, 0, 0, 0], 
                     [ 0, y, 0, 0], 
                     [ 0, 0, z, 0], 
                     [ 0, 0, 0, 1]]).astype(np.float32)

def lookAt(eye, at, up):
    a = eye - at
    b = up
    w = a / np.linalg.norm(a)
    u = np.cross(b, w)
    u = u / np.linalg.norm(u)
    v = np.cross(w, u)
    translate = np.array([[1, 0, 0, -eye[0]], 
                          [0, 1, 0, -eye[1]], 
                          [0, 0, 1, -eye[2]], 
                          [0, 0, 0, 1]]).astype(np.float32)
    rotate =  np.array([[u[0], u[1], u[2], 0], 
                        [v[0], v[1], v[2], 0], 
                        [w[0], w[1], w[2], 0], 
                        [0, 0, 0, 1]]).astype(np.float32)
    # return np.matmul(rotate, translate)
    return rotate @ translate

def quat2mat(quat):
    """
    note that quat is assumed [x, y, z, w]
    """
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(quat).as_matrix()

def mat2quat(mat):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(mat[:3, :3]).as_quat()

def posemat(center=None, quat=None, scale=None):
    rot_mat = np.identity(4)
    center_mat = np.identity(4)
    scale_mat = np.identity(4)

    if quat is not None:
        rot_mat[:3, :3] = quat2mat(quat)

    if center is not None:
        center_mat[0, 3] = center[0]
        center_mat[1, 3] = center[1]
        center_mat[2, 3] = center[2]

    if scale is not None:
        scale_mat[0, 0] = scale[0]
        scale_mat[1, 1] = scale[1]
        scale_mat[2, 2] = scale[2]

    return (center_mat @ rot_mat @ scale_mat)

def spherical(theta, phi, radius):
    # https://community.khronos.org/t/moving-the-camera-using-spherical-coordinates/49549
    # theta = np.radians(theta)
    # phi = np.radians(phi)
    y = radius * np.sin(phi)
    x = radius * np.cos(phi) * np.cos(theta)
    z = radius * np.cos(phi) * np.sin(theta)
    return x, y, z

def cartesian(x,y,z, radius=1):
    phi = np.arccos(z/radius)
    theta = np.arccos(x/(np.arcsin(radius*np.sin(phi))))
    return theta, phi

def checkerboard(width, repetitions) -> np.ndarray:
    tilesize = int(width//repetitions//2)
    check = np.kron([[1, 0] * repetitions, [0, 1] * repetitions] * repetitions, np.ones((tilesize, tilesize)))*0.33 + 0.33
    return np.stack((check, check, check), axis=-1)

if __name__ == "__main__":
    pass