#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy
import tf2_ros
import PyKDL as kdl
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Bool


class FusionPoseToServo:
    def __init__(self):
        rospy.init_node("fusion_pose_to_moveit_servo", anonymous=True)

        # ===== 파라미터 =====
        self.base_frame  = rospy.get_param("~base_frame",  "base_link")
        self.ee_frame    = rospy.get_param("~ee_frame",    "palm_1")  # MoveIt에서 EE link 이름
        self.twist_topic = rospy.get_param("~twist_topic", "/servo_server/delta_twist_cmds")

        # 위치/자세 게인
        self.k_pos = rospy.get_param("~k_pos", 1.0)    # 위치 오차 gain [1/s] 느낌
        self.k_rot = rospy.get_param("~k_rot", 0.5)    # 회전 오차 gain [1/s]

        # 최대 선속도 / 각속도 제한 (MoveIt Servo config와 대략 맞추기)
        self.max_lin_speed = rospy.get_param("~max_lin_speed", 0.3)  # [m/s]
        self.max_rot_speed = rospy.get_param("~max_rot_speed", 0.8)  # [rad/s]

        # on/off 스위치
        self.switch_on = False

        # TF 버퍼/리스너
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Publisher (MoveIt Servo 입력)
        self.twist_pub = rospy.Publisher(self.twist_topic, TwistStamped, queue_size=10)

        # Subscriber
        rospy.Subscriber("/fusion_pose", PoseStamped, self.fusion_pose_callback)
        rospy.Subscriber("/switch", Bool, self.switch_callback)

        rospy.loginfo("fusion_pose_to_moveit_servo node started")
        rospy.loginfo(" base_frame  : %s", self.base_frame)
        rospy.loginfo(" ee_frame    : %s", self.ee_frame)
        rospy.loginfo(" twist_topic : %s", self.twist_topic)

        rospy.spin()

    def switch_callback(self, msg: Bool):
        self.switch_on = msg.data
        rospy.loginfo_throttle(1.0, "switch status: %s", self.switch_on)

        # 꺼질 때는 0 twist 한 번 보내서 Servo 멈추게
        if not self.switch_on:
            zero = TwistStamped()
            zero.header.stamp = rospy.Time.now()
            zero.header.frame_id = self.base_frame
            self.twist_pub.publish(zero)

    def fusion_pose_callback(self, msg: PoseStamped):
        if not self.switch_on:
            return

        # 1) 현재 EE pose (base_frame 기준) TF에서 읽기
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_frame,
                rospy.Time(0),
                rospy.Duration(0.01)
            )
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(1.0, "TF %s->%s 실패: %s",
                                   self.base_frame, self.ee_frame, str(e))
            return

        # 현재 pose (TF에서)
        px_cur = tf.transform.translation.x
        py_cur = tf.transform.translation.y
        pz_cur = tf.transform.translation.z
        qx_cur = tf.transform.rotation.x
        qy_cur = tf.transform.rotation.y
        qz_cur = tf.transform.rotation.z
        qw_cur = tf.transform.rotation.w

        # 2) 목표 pose (/fusion_pose) – 기본적으로 base_frame 기준이라고 가정
        target = msg

        if target.header.frame_id and target.header.frame_id != self.base_frame:
            rospy.logwarn_throttle(
                1.0,
                "fusion_pose frame_id=%s (기대값=%s). TF 변환 추가 구현 필요.",
                target.header.frame_id, self.base_frame
            )
            # TODO: 필요하면 여기서 TF transformPose 추가

        px_des = target.pose.position.x
        py_des = target.pose.position.y
        pz_des = target.pose.position.z
        qx_des = target.pose.orientation.x
        qy_des = target.pose.orientation.y
        qz_des = target.pose.orientation.z
        qw_des = target.pose.orientation.w

        # 3) 위치 오차 계산
        ex = px_des - px_cur
        ey = py_des - py_cur
        ez = pz_des - pz_cur

        # 4) 자세 오차 계산 (R_err = R_cur^T * R_des)
        R_cur = kdl.Rotation.Quaternion(qx_cur, qy_cur, qz_cur, qw_cur)
        R_des = kdl.Rotation.Quaternion(qx_des, qy_des, qz_des, qw_des)
        R_err = R_cur.Inverse() * R_des

        # rot_vec = axis * angle (방향=축, 노름=각도) 형태 Vector
        rot_vec = R_err.GetRot()
        wx = self.k_rot * rot_vec[0]
        wy = self.k_rot * rot_vec[1]
        wz = self.k_rot * rot_vec[2]

        # 5) 선형 속도 = k_pos * 위치 오차
        vx = self.k_pos * ex
        vy = self.k_pos * ey
        vz = self.k_pos * ez

        # 6) 속도 제한 (선속도 / 각속도 각각 클램프)
        # 선속도 노름 제한
        lin_norm = math.sqrt(vx*vx + vy*vy + vz*vz)
        if lin_norm > self.max_lin_speed and lin_norm > 1e-6:
            scale = self.max_lin_speed / lin_norm
            vx *= scale
            vy *= scale
            vz *= scale

        # 각속도 노름 제한
        rot_norm = math.sqrt(wx*wx + wy*wy + wz*wz)
        if rot_norm > self.max_rot_speed and rot_norm > 1e-6:
            scale = self.max_rot_speed / rot_norm
            wx *= scale
            wy *= scale
            wz *= scale

        # 7) TwistStamped 메시지 구성 후 퍼블리시
        twist_msg = TwistStamped()
        twist_msg.header.stamp = rospy.Time.now()
        twist_msg.header.frame_id = self.base_frame  # Servo config의 planning_frame과 일치해야 함

        twist_msg.twist.linear.x  = vx
        twist_msg.twist.linear.y  = vy
        twist_msg.twist.linear.z  = vz
        twist_msg.twist.angular.x = wx
        twist_msg.twist.angular.y = wy
        twist_msg.twist.angular.z = wz

        self.twist_pub.publish(twist_msg)

        rospy.loginfo_throttle(
            0.5,
            "Servo cmd: lin=[%.3f %.3f %.3f], ang=[%.3f %.3f %.3f]",
            vx, vy, vz, wx, wy, wz
        )


if __name__ == "__main__":
    FusionPoseToServo()
