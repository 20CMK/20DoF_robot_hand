#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32MultiArray, Int16MultiArray

class Kalman1D:
    def __init__(self, Q=0.05, R=1.0, P0=10.0):
        self.Q = float(Q)
        self.R = float(R)
        self.P = float(P0)
        self.x = 0.0
        self.initialized = False

    def reset(self, x0=0.0, P0=None):
        self.x = float(x0)
        if P0 is not None:
            self.P = float(P0)
        self.initialized = True

    def update(self, z):
        z = float(z)
        if not self.initialized:
            self.reset(z)
            return self.x
        
        P_pred = self.P + self.Q

        K = P_pred / (P_pred + self.R)

        self.x = self.x + K * (z - self.x)
        self.P = (1.0 - K) * P_pred
        return self.x


class AngleSensorProcessor:
    zero  = [470, 489, 498, 461, 447, 413, 389, 451, 485, 385, 493, 449, 333, 471, 465, 341, 300, 450, 370, 370]
    final = [
        zero[0] + 300, zero[1] + 300, zero[2] + 300, zero[3] + 300, 
        zero[4] + 300, zero[5] + 300, zero[6] + 300, 
        zero[7] + 300, zero[8] + 300, zero[9] + 300,
        zero[10] + 300, zero[11] + 300, zero[12] + 300, 
        zero[13] + 300, zero[14] + 300, zero[15] + 300, 
        zero[16] + 150, zero[17] + 200, zero[18] + 300, zero[19] + 300
    ]

    def __init__(self):
        rospy.init_node("angle_sensor_processor")

        Q  = rospy.get_param("~Q", 0.05)   # 프로세스 노이즈(모델 변화량)
        R  = rospy.get_param("~R", 1.0)    # 측정 노이즈(센서 잡음)
        P0 = rospy.get_param("~P0", 10.0)  # 초기 공분산
        self.round_places = rospy.get_param("~round_places", 1)

        self.kf = [Kalman1D(Q=Q, R=R, P0=P0) for _ in range(20)]

        self.pub = rospy.Publisher("/angle_sensor", Float32MultiArray, queue_size=10)
        rospy.Subscriber("/angle_sensor_raw", Int16MultiArray, self.callback)

        rospy.loginfo("angle_sensor_processor with Kalman started (Q=%.3f, R=%.3f, P0=%.3f)" % (Q, R, P0))
        rospy.spin()

    def _norm_to_deg(self, val, z, f):
        if val <= z:
            return 0.0
        if val >= f:
            return 90.0
        return (val - z) / float(f - z) * 90.0

    def callback(self, msg):
        raw = list(msg.data)

        if len(raw) < 20:
            raw += [raw[-1]] * (20 - len(raw))
        elif len(raw) > 20:
            raw = raw[:20]

        degs = []
        for i in range(20):
            if i == 16: 
                if raw[i] < 300 : 
                    raw[i] = 300
                elif raw[i] > 450 :
                    raw[i] = 450
                raw[i] = (450 - raw[i]) +300
            deg = self._norm_to_deg(raw[i], self.zero[i], self.final[i])
            if i > 16 :
                deg = (-1) * deg
            deg_f = self.kf[i].update(deg)
            degs.append(round(deg_f, self.round_places))

        out = Float32MultiArray()
        out.data = degs
        self.pub.publish(out)

        rospy.loginfo_throttle(1.0, f"/angle_sensor (kalman): {degs}")

if __name__ == "__main__":
    try:
        AngleSensorProcessor()
    except rospy.ROSInterruptException:
        pass
