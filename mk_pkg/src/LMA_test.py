#!/usr/bin/env python3

import math
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import PyKDL as kdl
from urdf_parser_py.urdf import URDF
import kdl_parser_py.urdf
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Bool

def normalize_quaternion(x, y, z, w, eps=1e-12):
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n < eps:
        return 0.0, 0.0, 0.0, 1.0
    return x/n, y/n, z/n, w/n

def wrap_array_to_limits(q_list, lower_limits, upper_limits):
    out = []
    for i, q in enumerate(q_list):
        lo = float(lower_limits[i]); hi = float(upper_limits[i])
        q_wrapped = q
        if (hi - lo) > 1.0:  
            q_wrapped = ((q + math.pi) % (2*math.pi)) - math.pi
        if q_wrapped < lo: q_wrapped = lo
        if q_wrapped > hi: q_wrapped = hi
        out.append(q_wrapped)
    return out

class KDL_IK_Solver:
    def __init__(self):
        rospy.init_node("kdl_ik_solver", anonymous=True)
        self.cmd_pub = rospy.Publisher("/arm_controller/command", JointTrajectory, queue_size=10)

        base_link = rospy.get_param("~base_link", "base_link")
        tip_link  = rospy.get_param("~tip_link",  "palm_1")
        self.max_speed = rospy.get_param("~max_speed", 1.0)

        robot = URDF.from_parameter_server()
        ok, self.kdl_tree = kdl_parser_py.urdf.treeFromUrdfModel(robot)
        if not ok:
            rospy.logerr("URDF → KDL tree 실패")
            return

        self.chain = self.kdl_tree.getChain(base_link, tip_link)

        # 움직이는 조인트만 추출
        self.joint_names = []
        for i in range(self.chain.getNrOfSegments()):
            joint = self.chain.getSegment(i).getJoint()
            if joint.getTypeName() != 'None':
                self.joint_names.append(joint.getName())

        nj = len(self.joint_names)
        if nj == 0:
            rospy.logerr("가동 조인트 0개")
            return

        # FK, IK 준비
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)

        # joint limits
        self.lower_limits = kdl.JntArray(nj)
        self.upper_limits = kdl.JntArray(nj)
        for i in range(nj):
            self.lower_limits[i] = -3.14
            self.upper_limits[i] =  3.14

        # LMA만 사용
        self.ik_lma = kdl.ChainIkSolverPos_LMA(self.chain)

        # joint state
        self.q_current = kdl.JntArray(nj)
        self.have_joint_states = False
        self.switch_on = False

        rospy.Subscriber("/joint_states", JointState, self.joint_state_callback)
        rospy.Subscriber("/fusion_pose", PoseStamped, self.fusion_pose_callback)
        rospy.Subscriber("/switch", Bool, self.switch_callback)

        rospy.loginfo(f"Using joints: {self.joint_names}")
        rospy.spin()

    def clamp_list(self, q):
        out = []
        for i, val in enumerate(q):
            lo = float(self.lower_limits[i]); hi = float(self.upper_limits[i])
            if val < lo: val = lo
            if val > hi: val = hi
            out.append(val)
        return out

    def switch_callback(self, msg):
        self.switch_on = msg.data
        rospy.loginfo_throttle(1.0, f"switch: {self.switch_on}")

    def joint_state_callback(self, msg):
        hit = 0
        for i, n in enumerate(self.joint_names):
            if n in msg.name:
                idx = msg.name.index(n)
                self.q_current[i] = msg.position[idx]
                hit += 1
        if hit == len(self.joint_names):
            self.have_joint_states = True

    def try_solve_with_seed(self, seed_q, target_frame):
        q_seed = kdl.JntArray(len(seed_q))
        for i, v in enumerate(seed_q):
            q_seed[i] = v

        q_sol = kdl.JntArray(len(seed_q))
        ret = self.ik_lma.CartToJnt(q_seed, target_frame, q_sol)

        if ret >= 0:
            q_list = [float(q_sol[i]) for i in range(q_sol.rows())]
            q_list = self.clamp_list(q_list)
            return True, q_list, "LMA"
        return False, None, "fail"

    def fusion_pose_callback(self, msg):
        if not self.switch_on:
            rospy.loginfo_throttle(1.0, "switch off")
            return
        if not self.have_joint_states:
            rospy.logwarn_throttle(1.0, "joint_states 미수신")
            return

        # target pose 생성 (그대로 사용)
        px, py, pz = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        qx, qy, qz, qw = normalize_quaternion(
            msg.pose.orientation.x, msg.pose.orientation.y,
            msg.pose.orientation.z, msg.pose.orientation.w
        )
        target_frame = kdl.Frame(
            kdl.Rotation.Quaternion(qx, qy, qz, qw),
            kdl.Vector(px, py, pz)
        )

        q_cur_list = [float(self.q_current[i]) for i in range(self.q_current.rows())]

        # 3가지 seed
        seed_A = wrap_array_to_limits(q_cur_list, self.lower_limits, self.upper_limits)
        seed_B = [0.0]*len(self.joint_names)
        seed_C = q_cur_list

        for tag, seed in [("wrap(cur)", seed_A), ("zero", seed_B), ("cur", seed_C)]:
            ok, q_list, how = self.try_solve_with_seed(seed, target_frame)
            if ok:
                deltas = [abs(q_list[i] - q_cur_list[i]) for i in range(len(q_list))]
                max_delta = max(deltas) if deltas else 0.0
                speed = max(self.max_speed, 0.1)
                duration = max(max_delta / speed, 0.8)

                vels = [(q_list[i] - q_cur_list[i]) / duration for i in range(len(q_list))]
                vels = [round(v, 4) for v in vels]

                traj = JointTrajectory()
                traj.joint_names = self.joint_names
                pt = JointTrajectoryPoint()
                pt.positions = [round(v,4) for v in q_list]
                pt.velocities = vels
                pt.accelerations = [0.1] * len(self.joint_names)
                pt.time_from_start = rospy.Duration(duration)
                traj.points.append(pt)

                self.cmd_pub.publish(traj)
                rospy.loginfo_throttle(
                    1.0,
                    f"IK OK ({how}, seed={tag}), duration={duration:.3f}s"
                )
                return

        rospy.logwarn_throttle(1.0, "IK 실패 (LMA only)")

if __name__ == "__main__":
    KDL_IK_Solver()
