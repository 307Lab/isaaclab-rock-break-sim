import json
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from scipy.spatial import Voronoi, cKDTree
import impact_model
import matplotlib.colors as mcolors
import pickle

def generate_random_rock(num_points=30, scale=1.0, seed=None):
    """
    用随机点 + Convex Hull 生成石头

    Parameters:
        num_points: 点数量
        scale: 缩放比例
        seed: 随机种子（None 表示完全随机）
    """
    rng = np.random.default_rng(seed)

    # 生成随机点
    points = rng.standard_normal((num_points, 3))

    # 拉伸形状（让它更像石头）
    points *= np.array([1.0, 0.8, 0.6])

    hull = trimesh.convex.convex_hull(points)

    hull.apply_scale(scale)

    return hull


def halfspace_to_mesh(planes, bounds=10.0):
    """
    将半空间集合转为封闭凸多面体
    planes: [(origin, normal), ...]
    """
    # 创建一个大盒子作为初始体
    box = trimesh.creation.box(extents=[bounds]*3)

    cell = box

    for origin, normal in planes:
        # 用一个大平面构造切割体
        plane = trimesh.creation.box(extents=[bounds]*3)

        plane.apply_translation(origin)

        # 用法向裁掉一半（关键 hack）
        plane = plane.slice_plane(origin, normal)

        cell = trimesh.boolean.intersection([cell, plane], engine='scad')

        if cell is None or cell.is_empty:
            return None

    return cell


def voronoi_fracture(mesh, num_cells=20, surface_ratio=1.0,seed=None):
    """
    Voronoi fracture + 表面密度匹配采样
    surface_ratio: 表面点权重（1.0=严格匹配密度）
    """

    bounds = mesh.bounds
    min_bound, max_bound = bounds

    # ------------------------
    # 1. 体内采样
    # ------------------------
    rng = np.random.default_rng(seed)
    volume_seeds = rng.uniform(min_bound, max_bound, size=(num_cells*2, 3))
    # print(len(volume_seeds))

    # ------------------------
    # 2. 计算密度
    # ------------------------
    volume = mesh.volume
    surface_area = mesh.area

    density = num_cells / volume

    # 等效厚度
    d = volume ** (1.0 / 3.0)

    num_surface = int(density * surface_area * d * surface_ratio*0.5)

    # ------------------------
    # 3. 表面采样
    # ------------------------
    surface_points, _ = trimesh.sample.sample_surface(mesh, num_surface,seed=seed)

    # ------------------------
    # 4. 合并 seeds
    # ------------------------
    seeds = np.vstack([volume_seeds, surface_points])
    # print(len(seeds))

    vor = Voronoi(seeds)

    fragments = []

    for i, point in enumerate(seeds):
        try:
            # 从原 mesh 开始裁剪
            cell = mesh.copy()
            

            # 找所有邻居
            neighbors = set()
            for ridge_points in vor.ridge_points:
                if i in ridge_points:
                    neighbors.update(ridge_points)
            neighbors.discard(i)

            for j in neighbors:
                p1 = seeds[i]
                p2 = seeds[j]

                # 中点
                mid = (p1 + p2) / 2

                # 法向（指向 p1）
                normal = p1 - p2
                normal /= np.linalg.norm(normal)

                # 半空间裁剪（保留靠近 p1 的一侧）
                plane_origin = mid
                plane_normal = normal

                cell = trimesh.intersections.slice_mesh_plane(
                    mesh=cell,
                    plane_normal=plane_normal,
                    plane_origin=plane_origin,
                    cap=True, 
                    ngine="earcut"
                )

                if cell.is_empty:
                    break

            if not cell.is_empty:
                fragments.append(cell)

        except Exception as e:
            print(e)
            continue
    print(f"[INFO]: Generated {len(fragments)} fragments after Voronoi fracture.")

    return fragments

def compute_mass(fragments, density=2500):
    masses = []
    for mesh in fragments:
        
        if not mesh.is_watertight:
            mesh = mesh.convex_hull  # fallback
        vol = mesh.volume  # 体积（单位取决于你的坐标单位）
        mass = vol * density 

        masses.append(mass)
    print(f"[INFO]: Computed mass for fragments. Total mass: {sum(masses)}")
    return masses



def export_to_usd(meshes, masses, file_path="rock.usd"):
    stage = Usd.Stage.CreateNew(file_path)

    # 设置物理场景（必须）
    scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, -1, 0))
    scene.CreateGravityMagnitudeAttr(9.81)

    root = UsdGeom.Xform.Define(stage, "/World")

    for i, (mesh, mass) in enumerate(zip(meshes, masses)):
        prim_path = f"/World/rock_{i}"
        usd_mesh = UsdGeom.Mesh.Define(stage, prim_path)

        vertices = mesh.vertices
        faces = mesh.faces

        # 几何
        usd_mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in vertices])
        usd_mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        usd_mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())

        # ✅ 添加刚体
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(usd_mesh.GetPrim())
        rigid_api.CreateRigidBodyEnabledAttr(True)

        # ✅ 添加质量
        mass_api = UsdPhysics.MassAPI.Apply(usd_mesh.GetPrim())
        mass_api.CreateMassAttr(mass)

    stage.GetRootLayer().Save()
    print(f"[INFO]: USD saved to {file_path}")
    



    

def mesh_distance(a, b):
    tree = cKDTree(a.vertices)
    dist, _ = tree.query(b.vertices, k=1)
    return dist.min()



def contact_area(mesh_a, mesh_b, threshold=5e-4):
    # 加速结构
    prox = trimesh.proximity.ProximityQuery(mesh_b)

    contact_area = 0.0

    # 三角面中心
    centers = mesh_a.triangles_center
    areas = mesh_a.area_faces

    # 计算每个中心到 mesh_b 的最近距离
    dist = prox.signed_distance(centers)

    # 选取接触区域
    contact_faces = np.abs(dist) < threshold

    contact_area = areas[contact_faces].sum()

    return contact_area
    
    


def build_connectivity(fragments, threshold=1e-3,scale=1,base_HP=0.2):
    
    centers = np.array([f.centroid for f in fragments])

    N = len(fragments)
    adj = np.zeros((N, N), dtype=float)
    
    node_ids = list(range(N))
    print('[INFO]: calculating connectivity')
    # todo: add a process bar 
    
    for i in range(N):

        for j in range(N):
            if i == j:
                continue
            if not np.isfinite(fragments[i].vertices).all():
                continue
            if not np.isfinite(fragments[j].vertices).all():
                continue

            # 精确判断：是否接触
            dist = mesh_distance(fragments[i], fragments[j])

            if dist < threshold:
                area = contact_area(fragments[i], fragments[j])
                # HP=math.dist(centers[i],centers[j])*scale
                # print(HP)
                adj[i, j] = (area+base_HP)*scale
                adj[j, i] = (area+base_HP)*scale
    print('[INFO]: calculated connectivity')
    # print(adj)
    return adj,centers,node_ids

def save_per_fragment(adj, base_path):
    N = adj.shape[0]

    for i in range(N):
        data = {
            "id": i,
            "neighbors": np.where(adj[i] == 1)[0].tolist()
        }

        with open(f"{base_path}_frag_{i}.json", "w") as f:
            json.dump(data, f)
    


def plot_mesh(ax, mesh, color='gray', alpha=0.5):
    ax.plot_trisurf(
        mesh.vertices[:, 0],
        mesh.vertices[:, 1],
        mesh.vertices[:, 2],
        triangles=mesh.faces,
        color=color,
        alpha=alpha
    )
    
def visualize(original, fragments, adj_matrix=None, impacted_adj=None, impact_idx=0):
    fig = plt.figure(figsize=(18, 6))

    # ------------------------
    # 原始石头
    # ------------------------
    ax1 = fig.add_subplot(131, projection='3d')
    plot_mesh(ax1, original, color='gray', alpha=0.8)
    ax1.set_title("Original Rock")

    # ------------------------
    # 破碎后（原始连接）
    # ------------------------
    ax2 = fig.add_subplot(132, projection='3d')

    for frag in fragments:
        color = np.random.rand(3,)
        plot_mesh(ax2, frag, color=color, alpha=0.05)

    centers = np.array([frag.centroid for frag in fragments])
    N = len(fragments)
    valueMax=adj_matrix.max()

    if adj_matrix is not None:
        cmap = cm.get_cmap('RdYlGn')

        norm = mcolors.Normalize(
            vmin=0,
            vmax=valueMax
        )

        for i in range(N):
            for j in range(i + 1, N):
                if adj_matrix[i, j] > 0:
                    p1, p2 = centers[i], centers[j]
                    color = cmap(norm(adj_matrix[i, j]))

                    ax2.plot(
                        [p1[0], p2[0]],
                        [p1[1], p2[1]],
                        [p1[2], p2[2]],
                        color=color,
                        linewidth=1
                    )

        ax2.set_title("Before Impact")

    # ------------------------
    # 冲击后连接
    # ------------------------
    ax3 = fig.add_subplot(133, projection='3d')

    for frag in fragments:
        color = np.random.rand(3,)
        plot_mesh(ax3, frag, color=color, alpha=0.05)

    if impacted_adj is not None:
        cmap = cm.get_cmap('RdYlGn')

        norm = mcolors.Normalize(
            vmin=0,
            vmax=valueMax
        )

        for i in range(N):
            for j in range(i + 1, N):

                p1, p2 = centers[i], centers[j]

                if impacted_adj[i, j] > 0:
                    # 正常连接 → colormap
                    color = cmap(norm(impacted_adj[i, j]))
                elif impacted_adj[i, j] < 0:
                    # ❗断裂连接 → 黑色
                    color = 'black'
                else:
                    continue

                ax3.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    color=color,
                    linewidth=1
                )

        ax3.set_title("After Impact (Black = Broken)")

    # ------------------------
    # 画中心点（两个图共用）
    # ------------------------
    for ax in [ax2, ax3]:
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            color='red',
            s=1
        )
        
    ax3.scatter(
            centers[impact_idx, 0],
            centers[impact_idx, 1],
            centers[impact_idx, 2],
            color='blue',
            s=10
        )

    plt.show()

    
def main():
    
    # 1. 生成石头
    rock = generate_random_rock(num_points=40)

    # 2. Voronoi破碎
    fragments = voronoi_fracture(rock, num_cells=10)
    
    mass = compute_mass(fragments, density=2500)

    print(f"Generated {len(fragments)} fragments")
    # print(f"mass: {mass}")
    print(f"Total mass: {sum(mass)}")


    # 3. 导出USD（给Isaac Sim用）
    export_to_usd(fragments, mass, "fractured_rock.usd")
    
    adj,centers,ids=build_connectivity(fragments)
    idx = np.argmax(centers[:, 2])
    
    # save_per_fragment(adj,"fractured_rock")
    impact=impact_model.apply_impact(centers,adj,idx,[0, -1, 0])
    print(impact)
    # subgraph = impact_model.get_damaged_subgraphes(impact,centers,ids)
    groups_subgraphs, groups_centers, groups_ids = impact_model.group_by_subgraphs(impact, centers, ids)
    print(f"Number of subgraphes after impact: {len(groups_subgraphs)}")

    # 4. 可视化
    visualize(rock, fragments,adj,impact,idx)
    
"""
生成一个预破碎石头，并保存到文件中
Returns:
    Rock: 包含石头信息和Isaac Lab对象的Rock实例
"""
def generate_prebroken_rock_and_save(
    num_points=40,
    scale=0.5,
    num_cells=25,
    file_name="rock_data.pkl",
    seed=None
):
    
    # 原始石头
    rock = generate_random_rock(num_points=num_points,scale=scale,seed=seed)

    # Voronoi 破碎
    fragments = voronoi_fracture(rock,num_cells=num_cells,seed=seed)

    # 质量
    masses = compute_mass(fragments)

    # 连接关系
    adj, centers, ids = build_connectivity(fragments)

    # 保存
    data = {
        "fragments": fragments,
        "masses": masses,
        "adj": adj,
        "centers": centers,
        "ids": ids,
    }
    with open(file_name, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"[INFO]: rock saved to {file_name}")


if __name__ == "__main__":
    main()