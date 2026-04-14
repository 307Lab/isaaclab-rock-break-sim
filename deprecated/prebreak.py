import numpy as np
import trimesh
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# =========================
# 1. 生成随机石头（凸多面体）
# =========================
def generate_random_rock(num_points=50, scale=1.0):
    points = np.random.randn(num_points, 3)

    # 拉伸 → 更像石头
    points *= np.array([1.0, 0.8, 0.6])

    hull = trimesh.convex.convex_hull(points)
    hull.apply_scale(scale)

    return hull


# =========================
# 2. 在石头内部采样点
# =========================
def sample_points_in_mesh(mesh, num_points=3000):
    points = []

    min_b, max_b = mesh.bounds

    while len(points) < num_points:
        p = np.random.uniform(min_b, max_b)
        if mesh.contains([p])[0]:
            points.append(p)

    return np.array(points)


# =========================
# 3. 聚类生成碎块（核心）
# =========================
def fracture_by_clustering(mesh, num_fragments=25):
    print("Sampling points inside mesh...")
    points = sample_points_in_mesh(mesh, num_points=20000)

    print("Clustering...")
    kmeans = KMeans(n_clusters=num_fragments, n_init=5)
    labels = kmeans.fit_predict(points)

    fragments = []

    for i in range(num_fragments):
        cluster_pts = points[labels == i]

        if len(cluster_pts) < 20:
            continue

        try:
            frag = trimesh.convex.convex_hull(cluster_pts)
            frag.apply_scale(1.02)
            fragments.append(frag)
        except:
            continue

    return fragments


# =========================
# 4. 导出 USD
# =========================
def export_to_usd_with_physics(meshes, file_path="fractured_rock2.usd"):
    try:
        from pxr import Usd, UsdGeom, Gf, UsdPhysics, PhysxSchema
    except ImportError:
        print("pxr not found, please run inside Isaac Sim python.sh")
        return

    stage = Usd.Stage.CreateNew(file_path)

    # 设置up axis
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    # 根节点
    world = UsdGeom.Xform.Define(stage, "/World")

    for i, mesh in enumerate(meshes):
        prim_path = f"/World/rock_{i}"

        # =========================
        # 1. 创建 Mesh
        # =========================
        usd_mesh = UsdGeom.Mesh.Define(stage, prim_path)

        vertices = mesh.vertices
        faces = mesh.faces

        usd_mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in vertices])
        usd_mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        usd_mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())

        # =========================
        # 2. 添加 RigidBody
        # =========================
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(usd_mesh.GetPrim())
        rigid_api.CreateRigidBodyEnabledAttr(True)

        # 初始为动态物体
        rigid_api.CreateKinematicEnabledAttr(False)

        # =========================
        # 3. 添加 Collider
        # =========================
        collision_api = UsdPhysics.CollisionAPI.Apply(usd_mesh.GetPrim())

        # =========================
        # 4. 设置 Convex Hull（关键！）
        # =========================
        physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(usd_mesh.GetPrim())

        physx_collision.CreateApproximationAttr("convexHull")

        # =========================
        # 5. 设置质量（避免异常）
        # =========================
        mass_api = UsdPhysics.MassAPI.Apply(usd_mesh.GetPrim())
        mass_api.CreateMassAttr(1.0)

    stage.GetRootLayer().Save()
    print(f"USD with physics saved to {file_path}")


# =========================
# 5. 可视化
# =========================
def plot_mesh(ax, mesh, color='gray', alpha=0.5):
    ax.plot_trisurf(
        mesh.vertices[:, 0],
        mesh.vertices[:, 1],
        mesh.vertices[:, 2],
        triangles=mesh.faces,
        color=color,
        alpha=alpha
    )


def visualize(original, fragments):
    fig = plt.figure(figsize=(12, 6))

    # 原始石头
    ax1 = fig.add_subplot(121, projection='3d')
    plot_mesh(ax1, original, color='gray', alpha=0.8)
    ax1.set_title("Original Rock")

    # 破碎后
    ax2 = fig.add_subplot(122, projection='3d')

    for frag in fragments:
        color = np.random.rand(3,)
        plot_mesh(ax2, frag, color=color, alpha=0.7)

    ax2.set_title("Fractured Rock")

    plt.show()


# =========================
# 6. 主函数
# =========================
def main():
    print("Generating rock...")
    rock = generate_random_rock(num_points=20000, scale=1.0)

    print("Fracturing...")
    fragments = fracture_by_clustering(rock, num_fragments=100)

    print(f"Generated {len(fragments)} fragments")

    print("Exporting USD...")
    export_to_usd_with_physics(fragments, "fractured_rock.usd")

    print("Visualizing...")
    visualize(rock, fragments)


if __name__ == "__main__":
    main()