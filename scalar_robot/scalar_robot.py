import numpy as np

class ScalarRobot():
    def __init__(self, l1, l2):
        self.l1 = l1
        self.l2 = l2

    @staticmethod
    def Eval_figure(j2):
        if j2 >= 0:
            return "righty"
        else:
            return "lefty"

    @staticmethod
    def round_rad(rad):
        while np.pi < rad:
            rad -= 2 * np.pi
            
        while rad < -np.pi:
            rad += 2 * np.pi
    
        return rad

    def Forward_kinematics(self, joints_angle):
        j1 = np.deg2rad(joints_angle[0])
        j2 = np.deg2rad(joints_angle[1])
        j3 = joints_angle[2]
        j4 = np.deg2rad(joints_angle[3])

        x = self.l1 * np.cos(j1) + self.l2 * np.cos(j1 + j2)
        y = self.l1 * np.sin(j1) + self.l2 * np.sin(j1 + j2)
        z = j3
        theta = j1 + j2 + j4
        figure = self.Eval_figure(j2)

        return ([x, y, z, np.rad2deg(theta)], figure)

    def Inverse_kinematics(self, pos, figure):
        x = pos[0]
        y = pos[1]
        j3 = pos[2]
        theta = np.deg2rad(pos[3])
        
        tcp_distance = x**2 + y**2
        if tcp_distance < (self.l1 - self.l2) ** 2:
            raise ValueError("cannot calculate joints angle")
        if (self.l1 + self.l2 )**2 < tcp_distance:
            raise ValueError("cannot calculate joints angle")

        cos_j2 = (x**2 + y**2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        j2_1 = np.arccos(cos_j2)
        j2_2 = -j2_1

        if figure == "righty":
            if j2_1 >= 0:
                j2 = j2_1
            else:
                j2 = j2_2
        else:
            if j2_1 >= 0:
                j2 = j2_2
            else:
                j2 = j2_1
        
        j1 = np.arctan2(-self.l2 * np.sin(j2) * x + (self.l1 + self.l2 * np.cos(j2)) * y,  
                        (self.l1 + self.l2 * np.cos(j2)) * x + self.l2 * np.sin(j2) * y)
        j4 = theta - j1 - j2
        j1, j2, j4 = np.rad2deg([j1, j2, j4])
        return [j1, j2, j3, j4]
        
if __name__ == "__main__":
    l1 = 200
    l2 = 100
    robot = ScalarRobot(l1, l2)

    moter_angle = [0, 0, 0, 0]
    pos, figure = robot.Forward_kinematics(moter_angle)
    joints_result = robot.Inverse_kinematics(pos, figure)

    print("\n\n=========  Forward Kinematics ============")
    print("J1={}, J2={}, J3={}, J4={}".format(*moter_angle))
    print("X={:.2f}, Y={:.2f}, Z={:.2f}, θ={:.2f}".format(*pos))
    print("==========================================")

    print("\n\n=========  Inverse Kinematics ============")
    print("J1={:.2f}, J2={:.2f}, J3={:.2f}, J4={:.2f}".format(*joints_result))
    print("==========================================")
        