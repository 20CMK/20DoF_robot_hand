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
    """쿼터니언 (x, y, z, w)를 단위 쿼터니언으로 정규화."""
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n < eps:
        return 0.0, 0.0, 0.0, 1.0
    return x/n, y/n, z/n, w/n


def wrap_array_to_limits(q_list, lower_limits, upper_limits):
    """
    q_list(조인트 각 리스트)를 조인트별 리밋 안으로 wrap + clamp.
    - 리밋 폭이 1rad보다 크면 회전축으로 보고 2π wrap → [-pi, pi] 근처로 보정
    - 이후 최종적으로 [lower, upper]로 clamp
    """
    out = []
    for i, q in enumerate(q_list):
        lo = float(lower_limits[i])
        hi = float(upper_limits[i])
        q_wrapped = q
        if (hi - lo) > 1.0:
            q_wrapped = ((q + math.pi) % (2*math.pi)) - math.pi
        if q_wrapped < lo:
            q_wrapped = lo
        if q_wrapped > hi:
            q_wrapped = hi
        out.append(q_wrapped)
    return out


class KDL_IK_Solver:
    def __init__(self):
        rospy.init_node("kdl_ik_wdls_solver", anonymous=True)
        self.cmd_pub = rospy.Publisher(
            "/arm_controller/command", JointTrajectory, queue_size=10
        )

        self.base_link = rospy.get_param("~base_link", "base_link")
        tip_link       = rospy.get_param("~tip_link",  "palm_1")  # 필요하면 palm_1 등으로 변경
        tip_link       = rospy.get_param("~tip_link",  "end_effector_1") 
        self.max_speed = rospy.get_param("~max_speed", 2.0)

        robot = URDF.from_parameter_server()
        ok, self.kdl_tree = kdl_parser_py.urdf.treeFromUrdfModel(robot)
        if not ok:
            rospy.logerr("URDF → KDL tree 실패")
            return

        self.chain = self.kdl_tree.getChain(self.base_link, tip_link)

        # 움직이는 조인트 이름만 추출
        self.joint_names = []
        for i in range(self.chain.getNrOfSegments()):
            joint = self.chain.getSegment(i).getJoint()
            if joint.getTypeName() != 'None':
                self.joint_names.append(joint.getName())

        rospy.loginfo("KDL chain joints: %s", self.joint_names)
        for i in range(self.chain.getNrOfSegments()):
            seg_name = self.chain.getSegment(i).getName()
            rospy.loginfo("segment: %s", seg_name)

        nj = len(self.joint_names)
        if nj == 0:
            rospy.logerr("가동 조인트 0개")
            return

        # FK solver
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)

        # 조인트 리밋 (임시로 전체 -3.14 ~ 3.14)
        self.lower_limits = kdl.JntArray(nj)
        self.upper_limits = kdl.JntArray(nj)
        for i in range(nj):
            self.lower_limits[i] = -3.14
            self.upper_limits[i] =  3.14

        # 🔹 WDLS 속도 IK solver 생성
        eps     = 1e-5   # Jacobian 계산 허용오차
        maxiter = 150    # 내부 반복 (vel solver 기준)
        lam     = 0.01   # damping 계수 (특이점 안정화)
        try:
            self.ik_vel = kdl.ChainIkSolverVel_wdls(self.chain, eps, maxiter, lam)
        except TypeError:
            # PyKDL 빌드에 따라 시그니처가 다를 수 있어서 fallback
            self.ik_vel = kdl.ChainIkSolverVel_wdls(self.chain)

        # 관절 weight 설정 (관절 민감도 조절용, 지원 안 하면 warning)
        try:
            weights = kdl.JntArray(nj)
            for i in range(nj):
                weights[i] = 1.0
            self.ik_vel.setWeight(weights)
        except AttributeError:
            rospy.logwarn("WDLS: setWeight 미지원 PyKDL 버전일 수 있습니다.")

        # 선호자세 (null-space에서 끌고갈 posture)
        try:
            self.q_rest = kdl.JntArray(nj)
            for i in range(nj):
                self.q_rest[i] = 0.0
            self.ik_vel.setOptPos(self.q_rest)
        except AttributeError:
            rospy.logwarn("WDLS: setOptPos 미지원 PyKDL 버전일 수 있습니다.")

        # 현재 조인트 상태
        self.q_current = kdl.JntArray(nj)
        self.have_joint_states = False
        self.switch_on = False

        # servo-style step 크기 / 게인
        # self.pos_dt = rospy.get_param("~pos_dt", 0.02)   # q_next = q + qdot * dt
        self.pos_dt = rospy.get_param("~pos_dt", 0.3)   # q_next = q + qdot * dt
        # self.k_pos  = rospy.get_param("~k_pos", 1.0)     # 위치 오차 gain
        self.k_pos  = rospy.get_param("~k_pos", 1.0)     # 위치 오차 gain
        # self.k_rot  = rospy.get_param("~k_rot", 0.5)     # 자세 오차 gain (처음엔 작게 추천)
        self.k_rot  = rospy.get_param("~k_rot", 0.5)     # 자세 오차 gain (처음엔 작게 추천)

        rospy.Subscriber("/joint_states", JointState, self.joint_state_callback)
        rospy.Subscriber("/fusion_pose", PoseStamped, self.fusion_pose_callback)
        rospy.Subscriber("/switch", Bool, self.switch_callback)

        rospy.loginfo("Using joints (WDLS): %s", self.joint_names)
        rospy.spin()

    def clamp_list(self, q):
        out = []
        for i, val in enumerate(q):
            lo = float(self.lower_limits[i])
            hi = float(self.upper_limits[i])
            if val < lo:
                val = lo
            if val > hi:
                val = hi
            out.append(val)
        return out

    def switch_callback(self, msg):
        self.switch_on = msg.data
        rospy.loginfo_throttle(1.0, "switch status: %s", self.switch_on)

    def joint_state_callback(self, msg):
        hit = 0
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                idx = msg.name.index(name)
                self.q_current[i] = msg.position[idx]
                hit += 1
        if hit == len(self.joint_names):
            self.have_joint_states = True

    def step_wdls_once(self, target_frame):
        """
        현재 q_current에서 target_frame 쪽으로
        WDLS 속도 IK를 '한 번만' 적용해서 q_next를 리턴.
        (servo-style: 콜백마다 1 step)
        """
        nj = len(self.joint_names)
        if not self.have_joint_states:
            return False, None, "no_joint_state"

        # 1) 현재 조인트 각 복사
        q = kdl.JntArray(nj)
        for i in range(nj):
            q[i] = self.q_current[i]

        # 2) 현재 EE pose 계산 (FK)
        f_cur = kdl.Frame()
        self.fk_solver.JntToCart(q, f_cur)

        # 3) 현재 → 목표 twist
        twist = kdl.diff(f_cur, target_frame)
        v = twist.vel
        w = twist.rot

        # 4) gain 적용 (필요하면 k_rot를 작게 해서 방향만 느끼게)
        v_scaled = kdl.Vector(v[0]*self.k_pos,  v[1]*self.k_pos,  v[2]*self.k_pos)
        w_scaled = kdl.Vector(w[0]*self.k_rot,  w[1]*self.k_rot,  w[2]*self.k_rot)
        twist_scaled = kdl.Twist(v_scaled, w_scaled)

        # 5) WDLS 속도 IK: qdot 계산
        qdot = kdl.JntArray(nj)
        ret = self.ik_vel.CartToJnt(q, twist_scaled, qdot)
        if ret < 0:
            return False, None, f"vel_fail(ret={ret})"

        # 6) 한 스텝만 전진: q_next = q + qdot * dt
        dt = self.pos_dt
        q_next = []
        for i in range(nj):
            q_next.append(float(q[i] + qdot[i] * dt))

        # 7) 조인트 리밋 clamp
        q_next = self.clamp_list(q_next)

        return True, q_next, "ok"

    def fusion_pose_callback(self, msg):
        if not self.switch_on:
            rospy.loginfo_throttle(1.0, "switch off")
            return

        if not self.have_joint_states:
            rospy.logwarn_throttle(1.0, "joint_states 미수신 — IK 스킵")
            return

        # frame_id 확인 (base_link 기준이어야 함)
        if msg.header.frame_id and msg.header.frame_id != self.base_link:
            rospy.logwarn_throttle(
                1.0,
                "fusion_pose frame_id=%s (기대값=%s). TF 변환 안 하면 오동작할 수 있음.",
                msg.header.frame_id, self.base_link
            )

        px, py, pz = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        
        dist = math.sqrt(px*px + py*py + (pz-0.05)*(pz-0.05))

        # 🔹 0.8m(80cm)보다 멀면 수행하지 않고 리턴
        if dist > 0.7:
            rospy.logwarn_throttle(
                1.0,
                "target too far: %.3fm > 0.80m, IK 스킵", dist
            )
            return
        
        
        qx, qy, qz, qw = normalize_quaternion(
            msg.pose.orientation.x, msg.pose.orientation.y,
            msg.pose.orientation.z, msg.pose.orientation.w
        )
        target_frame = kdl.Frame(
            kdl.Rotation.Quaternion(qx, qy, qz, qw),
            kdl.Vector(px, py, pz)
        )

        # WDLS 1-step
        ok, q_list, how = self.step_wdls_once(target_frame)
        if not ok:
            rospy.logwarn_throttle(1.0, "IK step 실패: %s", how)
            return

        # 현재 q와 q_next 차이로 duration 계산
        q_cur_list = [float(self.q_current[i]) for i in range(self.q_current.rows())]
        deltas = [abs(q_list[i] - q_cur_list[i]) for i in range(len(q_list))]
        max_delta = max(deltas) if deltas else 0.0
        
        speed =  self.max_speed

        # servo-style이니까 너무 길게 잡지 말고 최소 0.1초 정도
        # duration = max(max_delta / speed, 0.01)
        duration = max_delta / speed

        vels = [(q_list[i] - q_cur_list[i]) / duration for i in range(len(q_list))]
        vels = [round(v, 4) for v in vels]

        traj = JointTrajectory()
        traj.joint_names = self.joint_names

        pt = JointTrajectoryPoint()
        pt.positions = [round(v, 4) for v in q_list]
        pt.velocities = vels
        pt.accelerations = [0.0]*len(self.joint_names)
        pt.time_from_start = rospy.Duration(duration)
        traj.points.append(pt)

        self.cmd_pub.publish(traj)

        rospy.loginfo_throttle(
            0.5, "IK 1-step ok (%s), duration=%.3fs", how, duration
        )


if __name__ == "__main__":
    KDL_IK_Solver()
