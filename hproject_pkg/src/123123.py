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

def wrap_to_range(angle, lo, hi):
    width = hi - lo
    if width <= 0:
        return max(min(angle, hi), lo)
    return (angle - lo) % width + lo

def wrap_array_to_limits(q_list, lower_limits, upper_limits):
    out = []
    for i, q in enumerate(q_list):
        lo = float(lower_limits[i]); hi = float(upper_limits[i])
        # 최근접 동치각으로 보내고(2π wrap), 그 후 클램프
        q_wrapped = q
        if (hi - lo) > 1.0:  # 회전축 가정
            # [-pi, pi] 근처로 2π wrap
            q_wrapped = ( (q + math.pi) % (2*math.pi) ) - math.pi
        # 최종 리밋 적용
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
        self.max_speed = rospy.get_param("~max_speed", 1.5)

        robot = URDF.from_parameter_server()
        ok, self.kdl_tree = kdl_parser_py.urdf.treeFromUrdfModel(robot)
        if not ok:
            rospy.logerr("URDF → KDL tree 실패")
            return

        self.chain = self.kdl_tree.getChain(base_link, tip_link)

        # 가동 조인트 정확히 추출 (fixed 섞임 방지)
        self.joint_names = []
        for i in range(self.chain.getNrOfSegments()):
            joint = self.chain.getSegment(i).getJoint()
            self.joint_names = []
            for i in range(self.chain.getNrOfSegments()):
                joint = self.chain.getSegment(i).getJoint()
                if joint.getTypeName() != 'None':     # ✅ fixed 제외
                    self.joint_names.append(joint.getName())


        nj = len(self.joint_names)
        if nj == 0:
            rospy.logerr("가동 조인트가 0개입니다.")
            return

        # FK/Vel/IK 준비
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)
        self.ik_vel    = kdl.ChainIkSolverVel_pinv(self.chain)

        # 리밋: -3.14 ~ +3.14
        self.lower_limits = kdl.JntArray(nj)
        self.upper_limits = kdl.JntArray(nj)
        for i in range(nj):
            self.lower_limits[i] = -3.14
            self.upper_limits[i] =  3.14

        # Joint-Limited IK (NR_JL)
        self.ik_jl = kdl.ChainIkSolverPos_NR_JL(
            self.chain, self.lower_limits, self.upper_limits,
            self.fk_solver, self.ik_vel, 300, 1e-6
        )
        # 폴백: LMA (리밋 미반영 → 해 구하고 clamp)
        self.ik_lma = kdl.ChainIkSolverPos_LMA(self.chain)

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
        rospy.loginfo_throttle(1.0, f"switch status {self.switch_on}")

    def joint_state_callback(self, msg):
        hit = 0
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                idx = msg.name.index(name)
                self.q_current[i] = msg.position[idx]
                hit += 1
        if hit == len(self.joint_names):
            self.have_joint_states = True

    def try_solve_with_seed(self, seed_q, target_frame):
        """NR_JL → 실패 시 LMA+clamp 폴백"""
        q_seed = kdl.JntArray(len(self.joint_names))
        for i, v in enumerate(seed_q):
            q_seed[i] = v

        q_sol = kdl.JntArray(len(self.joint_names))
        ret = self.ik_jl.CartToJnt(q_seed, target_frame, q_sol)
        if ret >= 0:
            return True, [float(q_sol[i]) for i in range(q_sol.rows())], "NR_JL"

        # fallback: LMA
        ret2 = self.ik_lma.CartToJnt(q_seed, target_frame, q_sol)
        if ret2 >= 0:
            q_list = [float(q_sol[i]) for i in range(q_sol.rows())]
            q_list = self.clamp_list(q_list)
            return True, q_list, "LMA+clamp"

        return False, None, "fail"

    def fusion_pose_callback(self, msg):
        if not self.switch_on:
            rospy.loginfo_throttle(1.0, "switch off")
            return

        if not self.have_joint_states:
            rospy.logwarn_throttle(1.0, "joint_states 미수신 — IK 스킵")
            return

        # 목표 자세 구성 (quat 정규화)
        px, py, pz = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        qx, qy, qz, qw = normalize_quaternion(
            msg.pose.orientation.x, msg.pose.orientation.y,
            msg.pose.orientation.z, msg.pose.orientation.w
        )
        target_frame = kdl.Frame(kdl.Rotation.Quaternion(qx, qy, qz, qw),
                                 kdl.Vector(px, py, pz))

        # 씨드 3종 시도
        q_cur_list = [float(self.q_current[i]) for i in range(self.q_current.rows())]
        seed_A = wrap_array_to_limits(q_cur_list, self.lower_limits, self.upper_limits)  # 현재각 wrap
        seed_B = [0.0]*len(self.joint_names)                                            # 제로 씨드
        seed_C = q_cur_list                                                             # 원본 현재각

        for tag, seed in [("wrap(cur)", seed_A), ("zero", seed_B), ("cur", seed_C)]:
            ok, q_list, how = self.try_solve_with_seed(seed, target_frame)
            if ok:
                deltas = [abs(q_list[i] - q_cur_list[i]) for i in range(len(q_list))]
                max_delta = max(deltas) if deltas else 0.0
                speed = max(self.max_speed, 1e-6)
                duration = max(max_delta / speed, 0.1)

                traj = JointTrajectory()
                traj.joint_names = self.joint_names
                pt = JointTrajectoryPoint()
                pt.positions = [round(v, 4) for v in q_list]
                pt.time_from_start = rospy.Duration(duration)
                traj.points.append(pt)
                self.cmd_pub.publish(traj)

                rospy.loginfo_throttle(1.0, f"IK ok ({how}, seed={tag}), duration={duration:.3f}s")
                return

        rospy.logwarn_throttle(1.0, "IK failed (NR_JL & LMA fallback 모두 실패)")

if __name__ == "__main__":
    KDL_IK_Solver()
