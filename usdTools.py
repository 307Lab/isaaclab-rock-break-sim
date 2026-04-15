from pxr import UsdGeom, Gf, UsdPhysics
import omni.usd
import numpy as np
import isaaclab.sim as sim_utils
import impact_model
from isaaclab.assets import RigidObject, RigidObjectCfg
from pxr import UsdGeom, Usd
import prebreakv2
import torch
import pickle

class Rock_state:
    def __init__(self, adj, centers, ids, masses, connections=None, base_translation=(0, 0, 0)):
        self.adj = adj
        self.centers = centers
        self.ids = ids
        self.masses = masses
        self.connections = connections
        self.base_translation = base_translation

class Rock:
    def __init__(self, rock_objects,rockname, root, adj, centers, ids, masses,connections=None,base_translation=(0, 0, 0),rock_initstate=None):
        self.rock_objects = rock_objects
        self.rockname = rockname
        self.root = root
        self.root_path = root.GetPath().pathString                                                                    
        self.adj = adj
        self.centers = centers
        self.ids = ids
        self.masses = masses
        self.connections = connections
        self.base_translation = base_translation
        self.rock_initstate = rock_initstate
        
def load_meshes_to_isaaclab(
    meshes,
    masses,
    root_path="/World/Objects",
    base_translation=(0, 0, 0),      # 整体位置
    per_mesh_translations=None,      # 每个mesh单独位置（list）
):
    stage = omni.usd.get_context().get_stage()
    
    root = UsdGeom.Xform.Define(stage, root_path)
    root_prim = root.GetPrim()
    sim_utils.standardize_xform_ops(root_prim)

    rock_objects = []
    for i, (mesh, mass) in enumerate(zip(meshes, masses)):
        mesh.face_normals  # 使用面法线
        mesh.vertex_normals = None
        prim_path = f"{root_path}/rock_{i}"

        usd_mesh = UsdGeom.Mesh.Define(stage, prim_path)

        vertices = mesh.vertices
        faces = mesh.faces

        usd_mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in vertices])
        usd_mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        usd_mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())

        prim = usd_mesh.GetPrim()

        # 刚体
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(prim)
        rigid_api.CreateRigidBodyEnabledAttr(True)

        # 质量
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr(mass)

        # 碰撞
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")    

        # =========================
        # 位置控制
        # =========================
        base = np.array(base_translation)

        if per_mesh_translations is not None:
            local = np.array(per_mesh_translations[i])
        else:
            local = np.zeros(3)

        final_pos = base + local 
        UsdGeom.XformCommonAPI(prim).SetTranslate(tuple(final_pos))
        
        # 创建刚体对象，获取位置用
        rockRigid_cfg = RigidObjectCfg(
            prim_path=prim_path,
            init_state=RigidObjectCfg.InitialStateCfg(),
        )
        rock_object = RigidObject(cfg=rockRigid_cfg)
        rock_objects.append(rock_object)  
    
    print(f"[INFO]: Loaded {len(meshes)} meshes into Isaac Lab.")
    return root_prim,rock_objects

def create_attachment_between_prims(rockname,actor0_id,actor1_id,actor0_path: str = "/World/BoxA",actor1_path: str = "/World/BoxB"):
    # Get current USD stage
    stage: Usd.Stage = omni.usd.get_context().get_stage()

    actor0_prim = stage.GetPrimAtPath(actor0_path)
    actor1_prim = stage.GetPrimAtPath(actor1_path)
    if not actor0_prim or not actor1_prim:
        raise RuntimeError(f"Rigid body prim not found at {actor0_path} or {actor1_path}")


    # Define a PhysxPhysicsAttachment under actor0
    attachment_path = f"/World/Objects/{rockname}/attachment_from_{actor0_id}_to_{actor1_id}"
    attachment = UsdPhysics.FixedJoint.Define(stage, attachment_path)

    # Set the two actors to be attached
    attachment.CreateBody0Rel().SetTargets([actor0_prim.GetPath()])
    attachment.CreateBody1Rel().SetTargets([actor1_prim.GetPath()])
    
def break_attachment_between_prims(rockname,actor0_id,actor1_id):
    stage: Usd.Stage = omni.usd.get_context().get_stage()
    attachment_path = f"/World/Objects/{rockname}/attachment_from_{actor0_id}_to_{actor1_id}"
    attachment_prim = stage.GetPrimAtPath(attachment_path)
    if attachment_prim:
        sim_utils.delete_prim(attachment_path)
        prim_i = f"/World/Objects/{rockname}/rock_{actor0_id}"
        prim_j = f"/World/Objects/{rockname}/rock_{actor1_id}"
        for path in [prim_i, prim_j]:
            prim = stage.GetPrimAtPath(path)
            collision_api = UsdPhysics.CollisionAPI.Apply(prim)
            collision_api.GetCollisionEnabledAttr().Set(True)



def update_break_meshes_new_new(impact_adj,rock,velocities=None):   
    # 从rock.connections中提取每个group的连接关系
    for connection in rock.connections:
        i, j = connection
        if impact_adj[rock.ids.index(i)][rock.ids.index(j)] <= 0:
            break_attachment_between_prims(rock.rockname, i, j)


"""
生成一个预破碎石头，并加载到 Isaac Lab 场景中
Returns:
    Rock: 包含石头信息和Isaac Lab对象的Rock实例
"""
def generate_prebroken_rock(
    num_points=40,
    scale=0.5,
    num_cells=25,
    rock_name="base_rock_0",
    root_path="/World/Objects/base_rock_0",
    base_translation=(0, 0, 0),
    seed=None
):
    
    # 原始石头
    rock = prebreakv2.generate_random_rock(num_points=num_points,scale=scale,seed=seed)

    # Voronoi 破碎
    fragments = prebreakv2.voronoi_fracture(rock,num_cells=num_cells,seed=seed)

    # 质量
    masses = prebreakv2.compute_mass(fragments)

    # 连接关系
    adj, centers, ids = prebreakv2.build_connectivity(fragments)

    # 加载到 IsaacLab
    root, rock_objects = load_meshes_to_isaaclab(fragments,masses,root_path=root_path,base_translation=base_translation)
    N = adj.shape[0]
    
    # 创建连接关系
    connections=[]
    for i in range(N):
        for j in range(i , N):  # 只遍历上三角，避免重复
            if adj[i][j] > 0:
                prim_i = f"/World/Objects/{rock_name}/rock_{i}"
                prim_j = f"/World/Objects/{rock_name}/rock_{j}"

                create_attachment_between_prims(rock_name,i,j,prim_i, prim_j)
                connections.append((i, j))
                
    Rock_initstate = Rock_state(adj=adj,centers=centers,ids=ids,masses=masses,connections=connections,base_translation=base_translation)
    
    return Rock(rock_objects=rock_objects,rockname=rock_name, root=root, adj=adj, centers=centers, ids=ids, masses=masses,connections=connections,base_translation=base_translation,rock_initstate=Rock_initstate)


"""
从文件中加载已生成的石头
Returns:
    Rock: 包含石头信息和Isaac Lab对象的Rock实例
"""   
def load_rock_from_file(
    file_name=None,
    rock_name="base_rock_0",
    root_path="/World/Objects/base_rock_0",
    base_translation=(0, 0, 0)
):
    if file_name is None:
        return None
    
    with open(file_name, "rb") as f:
        data = pickle.load(f)

    fragments = data["fragments"]
    masses = data["masses"]
    adj = data["adj"]
    centers = data["centers"]
    ids = data["ids"]
    # 加载到 IsaacLab
    root, rock_objects = load_meshes_to_isaaclab(fragments,masses,root_path=root_path,base_translation=base_translation)
    N = adj.shape[0]
    
    # 创建连接关系
    connections=[]
    for i in range(N):
        for j in range(i , N):  # 只遍历上三角，避免重复
            if adj[i][j] > 0:
                prim_i = f"/World/Objects/{rock_name}/rock_{i}"
                prim_j = f"/World/Objects/{rock_name}/rock_{j}"

                create_attachment_between_prims(rock_name,i,j,prim_i, prim_j)
                connections.append((i, j))
    
    Rock_initstate = Rock_state(adj=adj,centers=centers,ids=ids,masses=masses,connections=connections,base_translation=base_translation)
    
    return Rock(rock_objects=rock_objects,rockname=rock_name, root=root, adj=adj, centers=centers, ids=ids, masses=masses,connections=connections,base_translation=base_translation,rock_initstate=Rock_initstate)
    


"""
应用冲击模型
Returns:
    List[Rock]: 破碎后的岩石对象列表
"""

def apply_impact(
    rock,
    impact_idx,
    impact_dir,
    I0=250.0,     # 冲击力
    gamma=0.8,     # 传播衰减
    lamb=0.8,      # 距离衰减
    k=1.0,          # 材料系数
    alpha=1.0,     # 非线性结构强度
    beta=0.5,       # 损伤后损伤强度
    min_I=1e-3,     # 传播停止阈值
    k_c = 1.0,   # 压缩强度
    k_t = 0.2   # 抗拉强度只有20%
):
    impact_id=rock.ids.index(impact_idx)
    
    # 判断是否可破碎
    if not is_breakable(rock):
        print("[WARN]: Rock is not breakable.")
        return rock
    
    # 应用冲击模型
    impacted,node_forces,edge_force=impact_model.apply_impact(rock.centers,rock.adj,impact_id,impact_dir,I0,gamma,lamb,k,alpha,beta,min_I,k_c,k_t)
    groups_subgraphs, groups_centers, groups_ids, groups_Object = impact_model.group_by_subgraphs(impacted, rock.centers, rock.ids, rock.rock_objects)
    groups_masses=[]
    for idx, group_ids in enumerate(groups_ids):
        group_masses = [rock.masses[rock.ids.index(gid)] for gid in group_ids]
        groups_masses.append(group_masses)

    velocities = impact_model.compute_node_velocities(node_forces, rock.masses)

    print(f"[INFO]: Created {len(groups_subgraphs)} broken groups.")
    update_break_meshes_new_new(impacted, rock)
    
    # 根据子图连接性拆解石块
    rocks = []    
    for idx, group_ids in enumerate(groups_ids):
        connections=[]
        N=len(groups_ids[idx])
        for i in range(N):
            for j in range(0 , N):  # 只遍历上三角，避免重复
                if groups_subgraphs[idx][i][j] > 0:
                    connections.append((groups_ids[idx][i], groups_ids[idx][j]))

        rocks.append(Rock(rock_objects=groups_Object[idx],rockname=rock.rockname, root=rock.root, adj=groups_subgraphs[idx], centers=groups_centers[idx], ids=groups_ids[idx], masses=groups_masses[idx],connections=connections,base_translation=rock.base_translation,rock_initstate=rock.rock_initstate))
    
    return rocks

"""
检查石头是否可以破碎
"""
def is_breakable(rock):
    if (len(rock.adj) == 0):
        return False   
    return True

def quat_apply(q, v):
    # q: (N, 4)  [x, y, z, w]
    # v: (N, 3)
    q = q.float()
    v = v.float()
    q_w = q[:, 0:1]
    q_vec = q[:, 1:4]

    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    v_world = v + q_w * t + torch.cross(q_vec, t, dim=-1)
    return v_world

def transform_point(root_pos_w, root_quat_w, center_local):
    """
    root_pos_w: (N, 3)
    root_quat_w: (N, 4)
    center_local: (N, 3) or (3,)
    """
    if center_local.dim() == 1:
        center_local = center_local.unsqueeze(0).repeat(root_pos_w.shape[0], 1)

    rotated = quat_apply(root_quat_w, center_local)
    center_world = rotated + root_pos_w
    return center_world

"""
获取距离冲击点最近的岩石ID和距离
Returns:
    closest_id: 最近的岩石对象的内部岩石ID
    closest_rock: 最近的岩石对象
    min_dist: 最近距离
"""

def get_closest_rock_and_id(rocks, impact_point):
    min_dist = float('inf')
    closest_id = -1
    closest_rock = None
    if isinstance(rocks, list):
        for rock in rocks:
            for idx, rock_object in enumerate(rock.rock_objects):
                root_pose_w = rock_object.data.root_state_w
                root_pos_w = root_pose_w[:, 0:3]
                root_quat_w = root_pose_w[:, 3:7]
                cent = torch.tensor(rock.centers[idx]+rock.base_translation, device=root_pos_w.device)
                center_to_world = transform_point(root_pos_w, root_quat_w, cent)
                p = torch.tensor(impact_point, device=root_pos_w.device)
                dist = torch.norm(center_to_world - p, dim=1)
                # print(f"Rock {rock.rockname}.{rock.ids[idx]} - Distances: {dist.cpu().numpy()}")
                if dist < min_dist:
                    min_dist = dist
                    closest_id = rock.ids[idx]
                    closest_rock = rock
    else:
        for idx, rock_object in enumerate(rocks.rock_objects):
            root_pose_w = rock_object.data.root_state_w
            root_pos_w = root_pose_w[:, 0:3]
            root_quat_w = root_pose_w[:, 3:7]
            cent = torch.tensor(rocks.centers[idx]+rocks.base_translation, device=root_pos_w.device)
            center_to_world = transform_point(root_pos_w, root_quat_w, cent)
            p = torch.tensor(impact_point, device=root_pos_w.device)
            dist = torch.norm(center_to_world - p, dim=1)
            # print(f"Rock {rocks.rockname}.{rocks.ids[idx]} - Distances: {dist.cpu().numpy()}")
            if dist < min_dist:
                min_dist = dist
                closest_id = rocks.ids[idx]
                closest_rock = rocks
    return closest_id, closest_rock, min_dist

def update_rock_state(rock, dt=0.02):
    for rock_object in rock.rock_objects:
        rock_object.update(dt)
        
"""
更新状态，isaaclab需要这个函数来更新每个rock对象的状态，从而更新石块的位置信息
"""
        
def update_rocks_state(rocks, dt=0.02):
        # 如果rocks是一个列表，遍历更新每个rock的状态
    if isinstance(rocks, list):
        for rock in rocks:
            update_rock_state(rock, dt)
    else:
        # 如果rocks是一个单一的rock对象，直接更新它的状态
        update_rock_state(rocks, dt)
        
        
"""
重置石头状态
"""

def reset_rock(rocks : list, rock_name : str):
    if not isinstance(rocks, list):
        print("[WARNING]: No rock can be reset")
        return rocks
    stage = omni.usd.get_context().get_stage()
    for rock in rocks[:]:
        if rock.rockname == rock_name:
            initstate=rock.rock_initstate
            break
            
    N=len(initstate.ids)
    for i in range(N):
        for j in range(i , N):  # 只遍历上三角，避免重复
            if initstate.adj[i][j] > 0:
                prim_i = f"/World/Objects/{rock_name}/rock_{i}"
                prim_j = f"/World/Objects/{rock_name}/rock_{j}"
                break_attachment_between_prims(rock_name,i,j)
    
    rock_objects=[]
    for rock in rocks[:]:    
        if rock.rockname == rock_name:
            initstate=rock.rock_initstate
            root=rock.root
            for rock_object in rock.rock_objects:
                # rock_object.reset()
                rock_object.write_root_state_to_sim(rock_object.data.default_root_state)
                rock_objects.append(rock_object)
            rocks.remove(rock) 

    rock_org= Rock(rock_objects=rock_objects,rockname=rock_name,root=root,adj=initstate.adj,centers=initstate.centers, ids=initstate.ids, masses=initstate.masses, connections= initstate.connections, base_translation=initstate.base_translation, rock_initstate=initstate)
    rocks.append(rock_org)
    for idx in range(len(rock_org.ids)):
        prim = stage.GetPrimAtPath(f"{rock_org.root_path}/rock_{idx}")
        collision_api = UsdPhysics.CollisionAPI.Apply(prim)
        collision_api.GetCollisionEnabledAttr().Set(True)
    for i in range(N):
        for j in range(i , N):  # 只遍历上三角，避免重复
            if initstate.adj[i][j] > 0:
                prim_i = f"/World/Objects/{rock_name}/rock_{i}"
                prim_j = f"/World/Objects/{rock_name}/rock_{j}"
                create_attachment_between_prims(rock_name,i,j,prim_i, prim_j)
    
    return rocks
            
    