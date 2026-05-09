#!/usr/bin/env python3
import serial
import rospy
from std_msgs.msg import Int32

def make_eol(scheme: str) -> str:
    scheme = scheme.lower()
    if scheme == "cr":   return "\r"
    if scheme == "lf":   return "\n"
    if scheme == "crlf": return "\r\n"
    return ""

def main():
    rospy.init_node("imu_reset_node")

    port = rospy.get_param("~port", "/dev/ebium")
    baud = int(rospy.get_param("~baud", 115200))
    eol_scheme = rospy.get_param("~eol", "none")
    mode_topic = rospy.get_param("~mode_topic", "/imu_reset_mode")
    msg1 = rospy.get_param("~msg1", "<cmo2>")
    msg2 = rospy.get_param("~msg2", "<reset>")
    eol = make_eol(eol_scheme)

    ser = serial.Serial(port, baud, timeout=0.1)

    def send_ascii(txt: str):
        data = (txt + eol).encode("ascii")
        ser.write(data)
        ser.flush()

    def cb_mode(msg: Int32):
        v = int(msg.data)
        if v == 1:
            send_ascii(msg1)
            rospy.loginfo("IMU offset, send msg <cmo2>")
        elif v == 2:
            send_ascii(msg2)
            rospy.loginfo("IMU power reset, send msg <reset>")
        else:
            rospy.logwarn(f"Unsupported mode value: {v}")

    sub = rospy.Subscriber(mode_topic, Int32, cb_mode, queue_size=10)

    try:
        rospy.spin()  # ← 노드가 살아있도록 유지 (핸드셰이크 정상 완료)
    finally:
        try:
            ser.close()
        except Exception:
            pass

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
