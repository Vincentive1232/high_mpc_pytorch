import numpy as np
import os
import time

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

class Sim_Visual_Meshcat:
    def __init__(self, vis, pivot=np.array([2.0, 0.0, 3.0]), rope_length=2.0,
                 width=2.0, height=1.0, thickness=0.05):
        self.vis = vis
        self.pivot = pivot
        self.rope_length = rope_length
        self.width = width
        self.height = height
        self.thickness = thickness

        # 创建一个 group，整个摆放到这里
        self.root = self.vis["pendulum"]
        self.drone = self.vis["drone"]

        self._create_rope()
        self._create_gate()
        self._create_drone()

        self._setup_static_transforms()

    def _create_rope(self):
        self.root["rope"].set_object(
            g.Box([0.05, 0.05, self.rope_length]),
            g.MeshLambertMaterial(color=0x000000)
        )

    def _create_gate(self):
        w, h, t = self.width, self.height, self.thickness
        color = 0x000000

        self.root["gate/left"].set_object(
            g.Box([t, t, h+t]),
            g.MeshLambertMaterial(color=color)
        )
        self.root["gate/right"].set_object(
            g.Box([t, t, h+t]),
            g.MeshLambertMaterial(color=color)
        )
        self.root["gate/top"].set_object(
            g.Box([w, t, t]),
            g.MeshLambertMaterial(color=color)
        )
        self.root["gate/bottom"].set_object(
            g.Box([w, t, t]),
            g.MeshLambertMaterial(color=color)
        )

    def _create_drone(self):
        self.drone.set_object(g.ObjMeshGeometry.from_file(os.path.dirname(os.path.realpath(__file__)) + "/obj/cf2_assembly.obj"))

    def _setup_static_transforms(self):
        # 组件的相对位置：相对 group（pivot点）
        L, h, w, t = self.rope_length, self.height, self.width, self.thickness

        # 绳子放置：沿局部 z 轴向下，中心在绳子中点
        # z轴正方向是向上，我们想让绳子沿负z方向，因此绕x轴转180度再向下移动半根长度
        rope_tf = tf.concatenate_matrices(
            tf.translation_matrix(self.pivot),
            # tf.rotation_matrix(np.pi, [1, 0, 0]),  # 旋转180度使z轴向下
            tf.translation_matrix([0, 0, -L / 2])
        )
        self.root["rope"].set_transform(rope_tf)

        # 门相对 pivot 的位置：
        # 门顶端应该正好接在绳子底端，也就是 z = -L 处，
        # 门中心比门顶端低 h/2
        z_gate_center = -L - h / 2

        self.root["gate/left"].set_transform(tf.concatenate_matrices(tf.translation_matrix(self.pivot), tf.translation_matrix([-(w / 2), 0, z_gate_center])))
        self.root["gate/right"].set_transform(tf.concatenate_matrices(tf.translation_matrix(self.pivot), tf.translation_matrix([(w / 2), 0, z_gate_center])))
        self.root["gate/top"].set_transform(tf.concatenate_matrices(tf.translation_matrix(self.pivot), tf.translation_matrix([0, 0, -L])))
        self.root["gate/bottom"].set_transform(tf.concatenate_matrices(tf.translation_matrix(self.pivot), tf.translation_matrix([0, 0, -L - h])))

    def update(self, theta_pend, position: np.ndarray, quat: np.ndarray):
        # 整体group绕pivot点旋转
        # 先平移pivot点到原点，再绕y轴旋转，再平移回pivot位置
        trans_to_origin = tf.translation_matrix(-self.pivot)
        rot = tf.rotation_matrix(theta_pend, [0, 1, 0])  # 绕y轴转theta
        trans_back = tf.translation_matrix(self.pivot)

        tf_total = tf.concatenate_matrices(trans_back, rot, trans_to_origin)

        trans_drone = tf.concatenate_matrices(
            tf.translation_matrix(position),
            tf.quaternion_matrix(quat)
        )
        self.drone.set_transform(trans_drone)
        self.root.set_transform(tf_total)


if __name__ == "__main__":
    vis = meshcat.Visualizer().open()
    pendulum = RigidPendulumVisualizer(vis)

    t = 0.0
    while True:
        theta = 0.6 * np.sin(t)
        pendulum.update(theta)
        t += 0.03
        time.sleep(0.01)
