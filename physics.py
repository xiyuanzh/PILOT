import torch

def angular_velocities(q1, q2, dt):
    x1, y1, z1, w1 = torch.split(q1, 1, dim=-1)
    x2, y2, z2, w2 = torch.split(q2, 1, dim=-1)
    x = w1*x2 - x1*w2 - y1*z2 + z1*y2
    y = w1*y2 + x1*z2 - y1*w2 - z1*x2
    z = w1*z2 - x1*y2 + y1*x2 - z1*w2
    w = (2 / dt) * torch.cat((x,y,z), dim=-1)
    return w
    
def rotation_matrix_from_quat(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    
    q1, q2, q3, q0 = torch.split(Q, 1, dim=-1)
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    r0 = torch.cat((r00, r01, r02), dim=-1)
    r1 = torch.cat((r10, r11, r12), dim=-1)
    r2 = torch.cat((r20, r21, r22), dim=-1)
    rot_matrix = torch.stack((r0, r1, r2), dim=-2)
                            
    return rot_matrix

def compute_w(vi, mat_r=None):
    w_from_quat = angular_velocities(vi[:,:-1,-4:], vi[:,1:,-4:], 0.01) #b,t,3
    end = w_from_quat[:,-1,:].unsqueeze(1)
    w_from_quat = torch.cat((w_from_quat, end), axis=1)
    return w_from_quat

def compute_a(vi, mat_r=None):
    velocity_vi_global = (vi[:,1:,:3] - vi[:,:-1,:3]) / 0.01
    accel_vi_global = (velocity_vi_global[:,1:] - velocity_vi_global[:,:-1]) / 0.01 
    end = accel_vi_global[:,-1,:].unsqueeze(1)
    accel_vi_global = torch.cat((accel_vi_global, end, end), axis=1)
    accel_vi_global[:,:,-1] += 9.8

    accel_vi_local = torch.einsum('btji,bti->btj', \
        rotation_matrix_from_quat(vi[:,:,-4:]).permute(0,1,3,2), accel_vi_global) #btij -> btji,bti -> btj
    return accel_vi_local