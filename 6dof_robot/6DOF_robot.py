import numpy as np

class SixDoFRobot:
    def __init__(self, a, d):
        self.a = a
        self.d = d
        self.alpha = [0, np.pi*0.5, 0, np.pi*0.5, -np.pi*0.5, np.pi*0.5]
        return None
    
    @staticmethod
    def round_rad(rad):
        while np.pi < rad:
            rad -= 2 * np.pi
            
        while rad < -np.pi:
            rad += 2 * np.pi
    
        return rad
    
    @staticmethod
    def Calc_HT_matrix_from_6DOF(x, y, z, rx, ry, rz):
        rx = np.deg2rad(rx)
        ry = np.deg2rad(ry)
        rz = np.deg2rad(rz)

        sin_rx = np.sin(rx)
        cos_rx = np.cos(rx)
        sin_ry = np.sin(ry)
        cos_ry = np.cos(ry)
        sin_rz = np.sin(rz)
        cos_rz = np.cos(rz)
        HT_matrix = np.matrix([[cos_ry * cos_rz, sin_rx * sin_ry * cos_rz - cos_rx * sin_rz, cos_rx * sin_ry * cos_rz + sin_rx * sin_rz, x],
                               [cos_ry * sin_rz, sin_rx * sin_ry * sin_rz + cos_rx * cos_rz, cos_rx * sin_ry * sin_rz - sin_rx * cos_rz, y],
                               [        -sin_ry,                            sin_rx * cos_ry,                            cos_rx * cos_ry, z],
                               [              0,                                          0,                                          0, 1]])
        return HT_matrix

    def Eval_figure(self, joints_angle):
        figures = []
        j1 = joints_angle[0]
        j3 = joints_angle[2]
        j5 = joints_angle[4]
        
        if j1 >= 0:
            figures.append("lefty")
        else:
            figures.append("righty")

        if j3 < np.pi / 2 :
            figures.append("elbow up")
        else:
            figures.append("elbow down")

        if j5 <= 0:
            figures.append("wrist up")
        else:
            figures.append("wrist down")
        return figures
    
    def Calc_6DOF(self, HT_matrix):
        x = HT_matrix[0, 3]
        y = HT_matrix[1, 3]
        z = HT_matrix[2, 3]
        flag = np.finfo(float).eps > np.abs(1-np.abs(HT_matrix[2, 0]))
        ry = - np.arcsin(HT_matrix[2,0])

        if flag:
            rx = 0
            rz = np.arctan2(-HT_matrix[0, 1], HT_matrix[1, 1])
        else:
            rx = np.arctan2(HT_matrix[2, 1], HT_matrix[2, 2])
            rz = np.arctan2(HT_matrix[1, 0], HT_matrix[0, 0])
        return x, y, z, rx, ry, rz
    
    def Calc_rotation_matrix_of_joint(self, index, joint_rad):
        sin_x = np.sin(joint_rad)
        cos_x = np.cos(joint_rad)
        sinalpha = np.sin(self.alpha[index])
        cosalpha = np.cos(self.alpha[index])

        rotation_matrix = np.matrix([[           cos_x,       -1 * sin_x,            0],
                                     [sin_x * cosalpha, cos_x * cosalpha, -1* sinalpha],
                                     [sin_x * sinalpha, cos_x * sinalpha,     cosalpha]])
        return rotation_matrix
    
    def Calc_HT_matrix_of_joint(self, index, joint_rad):
        sin_x = np.sin(joint_rad)
        cos_x = np.cos(joint_rad)
        sinalpha = np.sin(self.alpha[index])
        cosalpha = np.cos(self.alpha[index])
        HT_matrix = np.matrix([[          cos_x,       -1 * sin_x,            0,                self.a[index]],
                              [sin_x * cosalpha, cos_x * cosalpha, -1* sinalpha, -1 *self.d[index] * sinalpha],
                              [sin_x * sinalpha, cos_x * sinalpha,     cosalpha,     self.d[index] * cosalpha],
                              [               0,                0,            0,                            1]])
        return HT_matrix
    
    def Calc_joints_HT_matrix(self, x, y, z, rx, ry, rz):
        joints_HT_matrix = self.Calc_HT_matrix_from_6DOF(x, y, z, rx, ry, rz)
        return joints_HT_matrix
    
    def Forward_kinematics(self, joint_angles):
        joint_rads = np.deg2rad(joint_angles)
        tcp_HT_matrix = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        for index, joint_rad in enumerate(joint_rads):   
            joint_HT_matrix = self.Calc_HT_matrix_of_joint(index, joint_rad)
            tcp_HT_matrix = tcp_HT_matrix * joint_HT_matrix
        x, y, z, rx, ry, rz = self.Calc_6DOF(tcp_HT_matrix)
        figures = self.Eval_figure(joint_rads)
        return [(x, y, z, np.rad2deg(rx), np.rad2deg(ry), np.rad2deg(rz)), figures]
    
    def Inverse_kinematics(self, targer_6DOF, figures):
        joints_HT_matrix = self.Calc_HT_matrix_from_6DOF(*targer_6DOF)

        px = joints_HT_matrix[0, 3]
        py = joints_HT_matrix[1, 3]
        pz = joints_HT_matrix[2, 3]
        
        # solve j1
        tmp_for_j1 = (self.d[1] + self.d[2])/(np.sqrt(px**2 + py**2))
        j1_1 = np.arctan2(py, px) + np.arcsin(tmp_for_j1)
        j1_2 = np.pi + np.arctan2(py, px) - np.arcsin(tmp_for_j1)
        
        # select j1 answer
        if figures[0] == "lefty":
            if j1_1 >= 0:
                j1 = j1_1
            else:
                j1 = j1_2
        else:
            if j1_1 >= 0:
                j1 = j1_2
            else:
                j1 = j1_1
    
        # solve j3
        tmp1_for_j3 = (px**2 + py**2 + pz**2 - 2 * self.a[1] * (px * np.cos(j1) + py * np.sin(j1))  \
           + self.a[1]**2 - self.a[2]**2 - self.a[3]**2 - self.d[1]**2 \
           - 2 * self.d[1] * self.d[2] - self.d[2]**2 - self.d[3]**2) / (2 * self.a[2]) 
        tmp2_for_j3 = tmp1_for_j3 / (np.sqrt(self.a[3]**2 + self.d[3]** 2))
        j3_1 = np.arcsin(tmp2_for_j3) - np.arctan2(self.a[3], self.d[3])
        j3_2 = np.pi - np.arcsin(tmp2_for_j3) - np.arctan2(self.a[3], self.d[3])
        
        if figures[1] == "elbow down":
            if j3_1 > 0.5 * np.pi:
                j3 = j3_1
            else:
                j3 = j3_2
        else: 
            if j3_1 <= 0.5* np.pi:
                j3 = j3_1
            else:
                j3 = j3_2
                
        j3 = self.round_rad(j3)

        # solve j2
        tmp1_for_j2 = px * self.a[3] * np.sin(j3) * np.cos(j1) - px * self.d[3] * np.cos(j1) * np.cos(j3) + py * self.a[3] * np.sin(j1) * np.sin(j3) \
          - py * self.d[3] * np.sin(j1) * np.cos(j3) - pz * self.a[2] - pz * self.a[3] * np.cos(j3) - pz * self.d[3] * np.sin(j3) \
          - self.a[1] * self.a[3] * np.sin(j3) + self.a[1] * self.d[3] * np.cos(j3)

        tmp2_for_j2 = -1 * px * self.a[2] * np.cos(j1) - px * self.a[3] * np.cos(j1) * np.cos(j3) - px * self.d[3] * np.sin(j3) * np.cos(j1) \
            - py * self.a[2] * np.sin(j1) - py * self.a[3] * np.sin(j1) * np.cos(j3) - py * self.d[3] * np.sin(j1) * np.sin(j3) \
            - pz * self.a[3] * np.sin(j3) + pz * self.d[3] * np.cos(j3) + self.a[1] * self.a[2] + self.a[1] * self.a[3] * np.cos(j3) \
            + self.a[1] * self.d[3] * np.sin(j3)
        
        det = - self.a[2]**2 - 2 * self.a[2] * self.a[3] * np.cos(j3) - 2 * self.a[2] * self.d[3] * np.sin(j3) - self.a[3]**2 -self.d[3]**2
        if det < 0:
            j2 = np.arctan2(-tmp1_for_j2, -tmp2_for_j2)
        else:
            j2 = np.arctan2(tmp1_for_j2, tmp2_for_j2)
            
        # solve j4, j5, j6
        joints_R_matrix = joints_HT_matrix[0:3, 0:3]
        j1_R_matrix = self.Calc_rotation_matrix_of_joint(0, j1)
        j2_R_matrix = self.Calc_rotation_matrix_of_joint(1, j2)
        j3_R_matrix = self.Calc_rotation_matrix_of_joint(2, j3)
        j1_inv_R_matrix = np.linalg.inv(j1_R_matrix)
        j2_inv_R_matrix = np.linalg.inv(j2_R_matrix)
        j3_inv_R_matrix = np.linalg.inv(j3_R_matrix)
        j_4_6_R_matrix = j3_inv_R_matrix * j2_inv_R_matrix * j1_inv_R_matrix * joints_R_matrix
        
        j4_1 = np.arctan2(j_4_6_R_matrix[2, 2], j_4_6_R_matrix[0, 2])
        j4_2 = self.round_rad(j4_1 + np.pi)
        
        j5_1 = np.arctan2(j_4_6_R_matrix[0, 2] * np.cos(j4_1) + j_4_6_R_matrix[2, 2] * np.sin(j4_1), -j_4_6_R_matrix[1,2])
        j5_2 = np.arctan2(j_4_6_R_matrix[0, 2] * np.cos(j4_2) + j_4_6_R_matrix[2, 2] * np.sin(j4_2), -j_4_6_R_matrix[1,2])
        
        j6_1 = np.arctan2(j_4_6_R_matrix[2, 0] * np.cos(j4_1) - j_4_6_R_matrix[0, 0] * np.sin(j4_1),
                          j_4_6_R_matrix[2, 1] * np.cos(j4_1) - j_4_6_R_matrix[0, 1] * np.sin(j4_1))
        j6_2 = np.arctan2(j_4_6_R_matrix[2, 0] * np.cos(j4_2) - j_4_6_R_matrix[0, 0] * np.sin(j4_2),
                          j_4_6_R_matrix[2, 1] * np.cos(j4_2) - j_4_6_R_matrix[0, 1] * np.sin(j4_2))
        
        if figures[2] == "wrist up":
            if j5_1 <= 0.0:
                j4 = j4_1
                j5 = j5_1
                j6 = j6_1
            else:
                j4 = j4_2
                j5 = j5_2
                j6 = j6_2
        else:
            if j5_1 > 0.0:
                j4 = j4_1
                j5 = j5_1
                j6 = j6_1
            else:
                j4 = j4_2
                j5 = j5_2
                j6 = j6_2
        joints_rad = [j1, j2, j3, j4, j5, j6]
        joints_angles = np.rad2deg(joints_rad)
        return joints_angles

if __name__ == "__main__":
    a = [0, 100, 50, 0, 0, 0]
    d = [0, -20, 20, 80, 0, 0]
    robot = SixDoFRobot(a, d)

    problem_angle = [-75, -20, 45, 50, 10, 30]
    actual_6DOF, figures = robot.Forward_kinematics(problem_angle)
    joints_angle = robot.Inverse_kinematics(actual_6DOF, figures)
    reproduce_6DOF, _figure = robot.Forward_kinematics(joints_angle)

    print("\n\n=========  Forward Kinematics ============")
    print("J1={}, J2={}, J3={}, J4={}, J5={}, J6={}".format(*[angle for angle in problem_angle]))
    print("6DOF: X={:.3f}, Y={:.3f}, Z={:.3f}, rx={:.3f}, ry={:.3f}, rz={:.3f}".format(*[pos for pos in actual_6DOF]))
    print("[{}, {}, {}]".format(figures[0], figures[1], figures[2]))
    print("==========================================")
    
    print("\n\n=========  Inverse Kinematics ============")
    print("J1={:.2f}, J2={:.2f}, J3={:.2f}, J4={:.2f}, J5={:.2f}, J6={:.2f}".format(*[angle for angle in joints_angle]))
    print("Answer: X={:.3f}, Y={:.3f}, Z={:.3f}, rx={:.3f}, ry={:.3f}, rz={:.3f}".format(*[pos for pos in reproduce_6DOF]))
    print("==========================================")