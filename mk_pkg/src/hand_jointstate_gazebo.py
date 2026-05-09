#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
import math
import threading
import time
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class AngleToJointState:
    def __init__(self):
        rospy.init_node("angle_to_jointstate")

        # 전체 손 관절 이름 (URDF와 일치해야 함)
        self.joint_names = [
            "joint1","joint2","joint3","joint4","joint5","joint6",
            "joint_index_RT","joint_middle_RT","joint_ring_RT","joint_pinky_RT",
            "joint_index_MCP","joint_index_PIP","joint_index_DIP",
            "joint_middle_MCP","joint_middle_PIP","joint_middle_DIP",
            "joint_ring_MCP","joint_ring_PIP","joint_ring_DIP",
            "joint_pinky_MCP","joint_pinky_PIP","joint_pinky_DIP",
            "joint_thumb_CMC_1","joint_thumb_CMC_2","joint_thumb_MP","joint_thumb_IP"
        ]
        
        # 내부 상태(모든 조인트 초기값 0)
        self.pos = {name: 0.0 for name in self.joint_names}
        self.vel = {name: 0.0 for name in self.joint_names}
        self.lock = threading.Lock()
        
        
        # publisher: RViz JointState
        self.pub = rospy.Publisher("/joint_states", JointState, queue_size=10)
        
        
        ############## 가제보 핸드############
        # self.pub_hand = rospy.Publisher("/hand_controller/command", JointTrajectory, queue_size=10)
        
        self.sub_arm_state = rospy.Subscriber(
            "/arm_controller/state",
            JointTrajectoryControllerState,  # ★ 여기
            self.cb_arm_state,
            queue_size=10
        )
        
        
        self.sub_angle = rospy.Subscriber(
            "/angle_sensor",
            Float32MultiArray, self.cb_angle, queue_size=10
        )
        
        self.timer = rospy.Timer(rospy.Duration(0.1), self.publish_now)

        rospy.loginfo("AngleToJointState node started.")
        rospy.spin()

    def cb_angle(self, msg: Float32MultiArray):
        finger_list = [
            "joint_index_RT","joint_middle_RT","joint_ring_RT","joint_pinky_RT",
            "joint_index_MCP","joint_index_PIP","joint_index_DIP",
            "joint_middle_MCP","joint_middle_PIP","joint_middle_DIP",
            "joint_ring_MCP","joint_ring_PIP","joint_ring_DIP",
            "joint_pinky_MCP","joint_pinky_PIP","joint_pinky_DIP",
            "joint_thumb_CMC_1","joint_thumb_CMC_2","joint_thumb_MP","joint_thumb_IP"
        ]
        
        with self.lock:
            for i, jn in enumerate(finger_list):
                if i >= len(msg.data):
                    break
                angle_deg = float(msg.data[i])
                self.pos[jn] = math.radians(angle_deg)

        ################## 가제보에서 핸드 조작 ###################
        # jt = JointTrajectory()
        # # jt.header.stamp = rospy.Time.now()
        # jt.header.stamp = rospy.Time(0)
        # jt.joint_names = list(finger_list)

        # p = JointTrajectoryPoint()
        # p.positions = [self.pos[j] for j in finger_list]
        # p.time_from_start = rospy.Duration(3)
        # jt.points.append(p)

        # self.pub_hand.publish(jt)
        ####################################################
        

    def cb_arm_state(self, msg: JointTrajectoryControllerState):
        with self.lock:
            for name, val in zip(msg.joint_names, msg.actual.positions):
                if name in self.pos:
                    if val > math.pi:
                        val = math.pi
                    elif val < -math.pi:
                        val = -math.pi
                    self.pos[name] = round(float(val), 5)
                    
            for name, vel in zip(msg.joint_names, msg.actual.velocities):
                if name in self.vel:
                    self.vel[name] = round(float(vel), 5)



    def cb_fake_js(self, msg: JointState):
        with self.lock:
            for name, val in zip(msg.name, msg.position):
                if name in self.pos:
                    self.pos[name] = round(float(val), 5)

    def publish_now(self, event=None):
        js = JointState()
        js.header.stamp = rospy.Time.now()
        js.name = list(self.joint_names)
        with self.lock:
            js.position = [self.pos[n] for n in self.joint_names]
            js.velocity = [self.vel[n] for n in self.joint_names]
        # velocity/effort는 생략(빈 리스트면 RViz/MoveIt OK)
        self.pub.publish(js)


if __name__ == "__main__":
    try:
        AngleToJointState()
    except rospy.ROSInterruptException:
        pass
