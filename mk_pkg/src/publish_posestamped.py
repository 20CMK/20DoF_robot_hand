#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion, Point

class PoseFusionNode:
    def __init__(self):
        rospy.init_node("fusuion_pose_node")
        self.pose_pub = rospy.Publisher("/fusion_pose", PoseStamped, queue_size=10)

        # 최신 데이터 저장용 변수
        self.latest_pos = None
        self.latest_ori = None

        # 구독
        rospy.Subscriber("/wrist_point", PointStamped, self.pos_cb)
        rospy.Subscriber("/wrist_imu_data", Quaternion, self.imu_cb)

        # 10Hz (0.1초마다) 주기적으로 발행
        rate_hz = rospy.get_param("~publish_rate", 10.0)
        rospy.Timer(rospy.Duration(1.0 / rate_hz), self.timer_callback)

        rospy.loginfo(f"PoseFusionNode started — publishing at {rate_hz} Hz")

    def pos_cb(self, msg):
        self.latest_pos = msg

    def imu_cb(self, msg):
        self.latest_ori = msg

    def timer_callback(self, event):
        if self.latest_pos is None and self.latest_ori is None:
            return
        
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "base_link"
        
        if self.latest_pos is not None:
            pose.pose.position = self.latest_pos.point
        # else:
        #     pose.pose.position = Point(0.4, 0.0, 0.5)

        if self.latest_ori is not None:
            pose.pose.orientation = self.latest_ori
        else:
            # pose.pose.orientation = Quaternion(0.0, 0.7071068, 0.0, 0.7071068)
            pose.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)

        self.pose_pub.publish(pose)

if __name__ == "__main__":
    PoseFusionNode()
    rospy.spin()
