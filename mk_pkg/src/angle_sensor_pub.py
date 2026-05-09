#!/usr/bin/env python3
import rospy
from std_msgs.msg import Int16MultiArray

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
    """
    /angle_sensor_raw (Int16MultiArray, len=6) 를 받아서
    - 인덱스별 raw 범위로 클램프
    - 인덱스별 출력 범위로 선형 변환
      * 인덱스 4, 5는 반전 매핑 (min→max, max→min)
    - 인덱스 5는 더 강한 칼만 필터 적용
    후 /angle_sensor6 (Int16MultiArray, len=6) 으로 발행
    """

    # Raw 범위 (센서 값 범위)
    RAW_MIN = [180, 210, 200, 250, 400, 200]
    RAW_MAX = [470, 650-100, 550-50, 460-50, 180, 600]

    # 출력 범위 (변환 후 값 범위)
    OUT_MIN = [130, 130, 130, 110, 270, 215]
    OUT_MAX = [510, 515, 515, 500, 400, 400]

    def __init__(self):
        rospy.init_node("angle_sensor_processor")

        Q  = rospy.get_param("~Q", 0.05)
        R  = rospy.get_param("~R", 1.0)
        P0 = rospy.get_param("~P0", 10.0)

        # 기본 채널(0~4, 5 포함)용 Kalman 필터 생성
        self.kf = [Kalman1D(Q=Q, R=R, P0=P0) for _ in range(6)]

        # ── 마지막 채널(인덱스 5)만 더 강한 필터 적용 ──
        # 기본값: Q를 10배 줄이고, R을 5배 키움 → 훨씬 덜 흔들림
        Q_last = rospy.get_param("~Q_last", Q * 0.1)
        R_last = rospy.get_param("~R_last", R * 3.0)
        self.kf[5] = Kalman1D(Q=Q_last, R=R_last, P0=P0)
        rospy.loginfo("Last channel (index 5) Kalman: Q=%.5f, R=%.5f" % (Q_last, R_last))

        self.pub = rospy.Publisher("/angle_sensor6", Int16MultiArray, queue_size=10)
        rospy.Subscriber("/angle_sensor_raw", Int16MultiArray, self.callback)

        rospy.loginfo("angle_sensor_processor started (Q=%.3f, R=%.3f, P0=%.3f)" % (Q, R, P0))
        rospy.spin()

    def _clamp(self, val, a, b):
        low = min(a, b)
        high = max(a, b)
        if val < low:
            return low
        if val > high:
            return high
        return val

    def _map_range(self, val, in_min, in_max, out_min, out_max, invert=False):
        """
        in_min/in_max 순서와 상관없이 [min, max] 로 정규화 후
        - invert=False : 센서 min -> out_min, 센서 max -> out_max
        - invert=True  : 센서 min -> out_max, 센서 max -> out_min (반전)
        """
        lo = min(in_min, in_max)
        hi = max(in_min, in_max)

        if lo == hi:
            return out_min

        val = self._clamp(val, lo, hi)
        ratio = float(val - lo) / float(hi - lo)

        if invert:
            ratio = 1.0 - ratio

        return out_min + ratio * (out_max - out_min)

    def callback(self, msg):
        raw = list(msg.data)

        # 길이 보정: 항상 6개로 맞추기
        if len(raw) < 6:
            raw += [raw[-1]] * (6 - len(raw))
        elif len(raw) > 6:
            raw = raw[:6]

        out_vals = []

        for i in range(6):
            v_raw = raw[i]

            # 4, 5번 인덱스는 반전 매핑
            invert = (i in (0, 1, 2, 3, 4))

            v_mapped = self._map_range(
                v_raw,
                self.RAW_MIN[i], self.RAW_MAX[i],
                self.OUT_MIN[i], self.OUT_MAX[i],
                invert=invert
            )

            v_filtered = self.kf[i].update(v_mapped)
            out_vals.append(int(round(v_filtered)))

        out_msg = Int16MultiArray()
        out_msg.data = out_vals
        self.pub.publish(out_msg)

        rospy.loginfo_throttle(1.0, f"/angle_sensor6 (filtered): {out_vals}")


if __name__ == "__main__":
    try:
        AngleSensorProcessor()
    except rospy.ROSInterruptException:
        pass
