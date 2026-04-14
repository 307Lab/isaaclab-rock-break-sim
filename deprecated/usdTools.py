from turtle import position

from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema
from omni.kit.primitive import mesh
import omni.usd
import numpy as np
import omni.kit.commands
from warp import pos
# from isaaclab.sim.views import XformPrimView
# from isaaclab.sim.views. import RigidPrimView
# from isaacsim.core.prims import RigidPrim,GeometryPrim
import isaaclab.sim as sim_utils
from omni.physx.scripts import physicsUtils
class Rock:
    def __init__(self, rock_objects,rockname, root, adj, centers, ids, masses,connections=None,base_translation=(0, 0, 0)):
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
        
def load_meshes_to_isaaclab(
    meshes,
    masses,
    root_path="/World/Objects",
    base_translation=(0, 0, 0),      # 整体位置
    per_mesh_translations=None,      # 每个mesh单独位置（list）
    random_offset_scale=0.0          # 随机扰动强度
):
    stage = omni.usd.get_context().get_stage()
    
    root = UsdGeom.Xform.Define(stage, root_path)
    root_prim = root.GetPrim()
    
    # rigid_api = UsdPhysics.RigidBodyAPI.Apply(root_prim)
    # rigid_api.CreateRigidBodyEnabledAttr(True)
    sim_utils.standardize_xform_ops(root_prim)

    # mass_api = UsdPhysics.MassAPI.Apply(root_prim)
    # mass_api.CreateMassAttr(sum(masses))  # 总质量

    # UsdPhysics.CollisionAPI.Apply(root_prim)
    # UsdPhysics.MeshCollisionAPI.Apply(root_prim).CreateApproximationAttr("convexHull")
    # mesh_path = []  # 用于记录每个mesh的prim
    # mesh_prims = []
    # meshviews = []
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
        # sim_utils.standardize_xform_ops(prim)
        # meshview=GeometryPrim(prim_paths_expr=prim_path, name="rigid_prim_view_rock_"+str(i))
        
        

        # =========================
        # 位置控制
        # =========================
        base = np.array(base_translation)

        if per_mesh_translations is not None:
            local = np.array(per_mesh_translations[i])
        else:
            local = np.zeros(3)

        # 随机扰动（防止完全重叠爆炸）
        # rand = np.random.uniform(-1, 1, 3) * random_offset_scale

        final_pos = base + local # + rand
        # meshview = XformPrimView(prim_path=prim_path, device="cuda:0")
        
        # meshview.set_world_poses(positions=torch.tensor(final_pos, dtype=torch.float32, device="cuda:0").unsqueeze(0),orientations=torch.tensor([[1, 0, 0, 0]], dtype=torch.float32, device="cuda:0"))

        UsdGeom.XformCommonAPI(prim).SetTranslate(tuple(final_pos))
        
        # mesh_prims.append(prim)  # 记录每个mesh的prim
        rockRigid_cfg = RigidObjectCfg(
            prim_path=prim_path,
            # spawn=cfg_rock,
            init_state=RigidObjectCfg.InitialStateCfg(),
        )
        rock_object = RigidObject(cfg=rockRigid_cfg)
        rock_objects.append(rock_object)
        # mesh_path.append(prim_path)
        # meshviews.append(meshview)
    
    
    
    print(f"[INFO]: Loaded {len(meshes)} meshes into Isaac Lab.")
    return root_prim,rock_objects

def create_attachment_between_prims(rockname,actor0_id,actor1_id,actor0_path: str = "/World/BoxA",actor1_path: str = "/World/BoxB",):
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

    # Let PhysX automatically compute the joint frames from current poses.
    # This is the key "easy" step: no manual joint transform math. [web:12][web:41]
    # PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment.GetPrim())

    # print(f"[INFO] Created attachment at {attachment_path} between {actor0_path} and {actor1_path}")
    
def break_attachment_between_prims(rockname,actor0_id,actor1_id):
    stage: Usd.Stage = omni.usd.get_context().get_stage()
    attachment_path = f"/World/Objects/{rockname}/attachment_from_{actor0_id}_to_{actor1_id}"
    attachment_prim = stage.GetPrimAtPath(attachment_path)
    if attachment_prim:
        sim_utils.delete_prim(attachment_path)
        # print(f"[INFO] Deleted attachment at {attachment_path}")
    # else:
        # print(f"[WARNING] Attachment prim not found at {attachment_path}, cannot delete.")

from isaaclab.assets import RigidObject, RigidObjectCfg
def spawn_rock(sim, rock_usd_path="rock.usd", root_path="/World/Objects/base_rock_0",base_translation=(0, 0, 10)):
    """Spawn a USD rock file into the scene, then restore Rock object."""
    # 已废弃

    # 导入 rock.usd
    cfg_rock = sim_utils.UsdFileCfg(usd_path=rock_usd_path)
    rock_obj=cfg_rock.func(root_path, cfg_rock, translation=base_translation)
    rockRigid_cfg = RigidObjectCfg(
        prim_path=root_path,
        # spawn=cfg_rock,
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    rock_object = RigidObject(cfg=rockRigid_cfg)

    stage = sim._stage
    root_prim = stage.GetPrimAtPath(root_path+"/rock_0")

    # 遍历子 prim 获取碎块 Mesh
    mesh_prims = []
    for prim in stage.Traverse():
        if prim.GetPath().pathString.startswith(root_path) and prim.GetTypeName() == "Mesh":
            mesh_prims.append(prim)
            # 避免子刚体继承父 transform 冲突
            # UsdGeom.XformCommonAPI(prim).SetResetXformStack(True)

    # 从 root prim 的自定义属性恢复 Rock 数据
    adj_attr = root_prim.GetAttribute("adjacency_matrix")
    adj = np.array(adj_attr.Get()).reshape(-1, int(np.sqrt(len(adj_attr.Get())))) if adj_attr else None

    centers_attr = root_prim.GetAttribute("centers")
    centers = np.array(centers_attr.Get()).reshape(-1, 3) if centers_attr else None

    ids_attr = root_prim.GetAttribute("ids")
    ids = np.array(ids_attr.Get()).tolist() if ids_attr else None

    masses_attr = root_prim.GetAttribute("masses")
    masses = np.array(masses_attr.Get()).tolist() if masses_attr else None

    rock = Rock(rock_object,root_prim, adj, centers, ids, masses, mesh_prims)
    return rock



from pxr import UsdGeom, Usd

# from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim.utils import move_prim,delete_prim
import torch

def update_break_meshes_new(groups_ids,rock,velocities=None):
    rock_prims = {}
    stage = omni.usd.get_context().get_stage()
    root_path = rock.root.GetPath().pathString
    
    # 先收集所有 rock prim
    rock_prims = {}
    for child in rock.root.GetChildren():
        name = child.GetName()
        if name.startswith("rock_"):
            idx = int(name.split("_")[1])
            rock_prims[idx] = child
            
    # 获取root的世界变换
    root_pose_w = rock.rock_obj.data.root_state_w
    # root_pos_w = root_pose_w[:, 0:3]
    # root_quat_w = root_pose_w[:, 3:7]
    # root_lin_vel_w = root_pose_w[:, 7:10]
    # root_ang_vel_w = root_pose_w[:, 10:13]
    # print("Root world pos:", root_pos_w)
    # print("Root world quat:", root_quat_w)

    
    # 创建 group 并把对应的 rock 移动到 group 下    
    groups_prims = []
    groups_Object=[] 
    for gid, group in enumerate(groups_ids):
        # print(f"Creating group {gid} with rock ids: {group}")
        group_path = f"/World/Objects/{rock.rockname}_group_{gid}"

        group_xform = UsdGeom.Xform.Define(stage, group_path)
        group_prim = group_xform.GetPrim()
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(group_prim)
        rigid_api.CreateRigidBodyEnabledAttr(True)
        sim_utils.standardize_xform_ops(group_prim)

        
        
        # 移动 rock 到 group 下
        for rid in group:
            if rid not in rock_prims:
                continue
            
            rock_prim = rock_prims[rid]
            
            old_path = rock_prim.GetPath()
            new_path = f"{group_path}/rock_{rid}"
            # move_prim(path_from=str(old_path), path_to=new_path, keep_world_transform=True, stage=stage)
            usd_mesh = UsdGeom.Mesh.Define(stage, new_path)

            vertices = rock.meshes[rid].vertices
            faces = rock.meshes[rid].faces

            usd_mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in vertices])
            usd_mesh.CreateFaceVertexCountsAttr([3] * len(faces))
            usd_mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())

            prim = usd_mesh.GetPrim()

            # 刚体
            # rigid_api = UsdPhysics.RigidBodyAPI.Apply(prim)
            # rigid_api.CreateRigidBodyEnabledAttr(True)

            # 质量
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr(rock.masses[rid])

            # 碰撞
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
            
            
        # 设为刚体
        
        # UsdPhysics.CollisionAPI.Apply(group_prim)
        # print(rigid_api.GetRigidBodyEnabledAttr())
        # sim_utils.standardize_xform_ops(group_prim)
        
            
        groups_prims.append(group_prim)
        
    
    
    # 删除旧的 rockprim
    # delete_prim(root_path)
    # rock.rock_obj.set_visibility(False)
    # print("Deleted old rock prim at:", root_path)
    
    for gid, group in enumerate(groups_ids):
        # print(f"Creating group {gid} with rock ids: {group}")
        group_path = f"/World/Objects/{rock.rockname}_group_{gid}"
        # 为 group 创建刚体对象    
        rockRigid_cfg = RigidObjectCfg(
            prim_path=group_path,
            init_state=RigidObjectCfg.InitialStateCfg(),
        )
        rock_object = RigidObject(cfg=rockRigid_cfg)
        # rock_object.write_root_state_to_sim(root_pose_w)
        
        groups_Object.append(rock_object)
    
    
    print(f"[INFO]: Created {len(groups_prims)} broken groups.")
    
    return groups_prims, groups_Object

def update_break_meshes_new_new(impact_adj,rock,velocities=None):
    
    # 从rock.connections中提取每个group的连接关系
    for connection in rock.connections:
        i, j = connection
        if impact_adj[rock.ids.index(i)][rock.ids.index(j)] <= 0:
            break_attachment_between_prims(rock.rockname, i, j)
            # print(f"Broken attachment between rock_{i} and rock_{j}")

# from omni.physx import get_physx_interface
def update_break_meshes(groups_ids,rock,velocities=None):
    """
    groups_ids: List[List[int]]
        每个子列表是一组 rock 的 id，例如 [[0,2],[1,3]]
    root_prim: Usd.Prim
        load_meshes_to_isaaclab 返回的 root
    """

    stage = omni.usd.get_context().get_stage()
    root_path = rock.root.GetPath().pathString
    # curr_transform = get_physx_interface().get_rigidbody_transformation(str(root_path))
    # rv_success = curr_transform["ret_val"]
    # if rv_success:
    #     curr_pos_f3 = curr_transform["position"]
    #     curr_pos = Gf.Vec3d(curr_pos_f3[0], curr_pos_f3[1], curr_pos_f3[2])
    #     curr_rot_f4 = curr_transform["rotation"]
    #     curr_rot_quat = Gf.Quatd(curr_rot_f4[3], curr_rot_f4[0], curr_rot_f4[1], curr_rot_f4[2])
        
    # print(curr_pos,curr_rot_quat)
    
    
    # prim=RigidPrim(prim_paths_expr=root_path, name="rigid_prim_view")
    # prim
    # pose=prim.get_world_poses()
    # print(pose)
    # rootview.destroy()
    # =========================
    # 关闭整体刚体
    # =========================
    
    # if rock.view is not None:
        # rock.view.destroy()
    
    if rock.root.HasAPI(UsdPhysics.RigidBodyAPI):
        rigid_api = UsdPhysics.RigidBodyAPI(rock.root)
        rigid_api.CreateRigidBodyEnabledAttr(False)
    
    # print("Destroyed root rigid view.")
    # =========================
    # 2收集所有 rock prim
    # =========================
    rock_prims = {}
    for child in rock.root.GetChildren():
        name = child.GetName()
        if name.startswith("rock_"):
            idx = int(name.split("_")[1])
            rock_prims[idx] = child

    # =========================
    # 创建 group
    # =========================
    group_prims = []
    groups_view = []
    groups_meshviews = []
    groups_meshviews_path = []   
    for gid, group in enumerate(groups_ids):
        # print(f"Creating group {gid} with rock ids: {group}")
        group_path = f"{root_path}/group_{gid}"

        group_xform = UsdGeom.Xform.Define(stage, group_path)
        group_prim = group_xform.GetPrim()

        
        # 设为刚体
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(group_prim)
        rigid_api.CreateRigidBodyEnabledAttr(True)
        sim_utils.standardize_xform_ops(group_prim)
        # UsdPhysics.CollisionAPI.Apply(group_prim)
        # UsdPhysics.MeshCollisionAPI.Apply(group_prim).CreateApproximationAttr("convexHull")

        # =========================
        # 把 rock 移到 group 下
        # =========================
        # xform_cache = UsdGeom.XformCache()
        
        group_paths=[]
        group_meshviews = []
        for rid in group:
            if rid not in rock_prims:
                continue
            
            rock_prim = rock_prims[rid]
            
            # world_pos = world_transform.ExtractTranslation()
            # print(f"Moving rock_{rid} to group_{gid} with world position {world_pos}")
            # xform_cache = UsdGeom.XformCache(Usd.TimeCode.defa())
            # print("PHYSX:", xform_cache.GetLocalToWorldTransform(rock_prim).ExtractTranslation())
            old_path = rock_prim.GetPath()
            new_path = f"{group_path}/rock_{rid}"
            group_paths.append(new_path)

            
            # ⚠️ USD 重命名 = 移动
            # meshview=rock.meshviews[rid]
            # meshview
            

            # positions, orientations = meshview.get_world_poses(usd=False)
            # print(rid,meshview,positions, orientations)
            # group_meshviews.append(meshview)
            # meshview.destroy()

            # omni.kit.commands.execute(
            #     "MovePrim",
            #     path_from=str(old_path),
            #     path_to=new_path
            # )
            move_prim(path_from=str(old_path), path_to=new_path, keep_world_transform=False, stage=stage)
            
            # print("ddd")
            # positions, orientations = meshviews.get_world_poses([rid])
            # moved_prim = stage.GetPrimAtPath(new_path)
            #set pos to correct pose
            # UsdGeom.XformCommonAPI(moved_prim).SetTranslate(tuple(positions[0].cpu().tolist()))
            # from pxr import Gf

            # quat = orientations[0].cpu().tolist()  # [w, x, y, z]

            # UsdGeom.XformCommonAPI(moved_prim).SetRotate(Gf.Quatf(quat[0], quat[1], quat[2], quat[3]))
            
            # sim_utils.standardize_xform_ops(moved_prim)
            # prim_view = XformPrimView(prim_path=new_path, device="cuda:0")
            
            # prim_view.set_world_poses(
            #     positions=positions,
            #     orientations=orientations  # quaternion OK
            # )
            
            # moved_prim = stage.GetPrimAtPath(new_path)
            # xform = UsdGeom.Xformable(moved_prim)
            # world_transform = xform.ComputeLocalToWorldTransform(rock_prim)
            # xform.ClearXformOpOrder()

            # xform.AddTransformOp().Set(world_transform)
            # UsdGeom.XformCommonAPI(moved_prim).SetTranslate(world_pos)
            
        group_prims.append(group_prim)
        groups_meshviews.append(group_meshviews)
       

        
        
        
        # print(group_path)
        # group_view=RigidPrim(prim_paths_expr=group_path, name=f"rigid_prim_view_group_{gid}")
        # group_view.initialize()
        # print("Group view:", group_view.initialized,group_view.is_physics_handle_valid(),group_view.is_valid())
        
        # groups_view.append(group_view)
        # groups_meshviews.append(group_meshviews)
    

    print(f"[INFO]: Created {len(group_prims)} broken groups.")

    return group_prims,groups_view, groups_meshviews

import prebreakv2

def generate_prebroken_rock(
    num_points=40,
    scale=0.5,
    num_cells=25,
    rock_name="base_rock_0",
    root_path="/World/Objects/base_rock_0",
    base_translation=(0, 0, 10),
    seed=None
):
    """
    生成一个预破碎石头，并加载到 Isaac Lab 场景中

    Returns:
        dict:
            {
                "rock": 原始mesh,
                "fragments": 碎块列表,
                "root": USD根prim,
                "mesh_prims": 每个碎块的prim,
                "adj": 邻接矩阵,
                "centers": 碎块中心,
                "ids": 节点ID,
                "masses": 质量列表
            }
    """

    # 1. 原始石头
    rock = prebreakv2.generate_random_rock(
        num_points=num_points,
        scale=scale,
        seed=seed
    )

    # 2. Voronoi 破碎
    fragments = prebreakv2.voronoi_fracture(
        rock,
        num_cells=num_cells,
        seed=seed
    )

    # 3. 质量
    masses = prebreakv2.compute_mass(fragments)

    # 4. 连接关系
    adj, centers, ids = prebreakv2.build_connectivity(fragments)

    # 5. 加载到 IsaacLab
    root, rock_objects = load_meshes_to_isaaclab(
        fragments,
        masses,
        root_path=root_path,
        base_translation=base_translation
    )
    N = adj.shape[0]
    connections=[]
    for i in range(N):
        for j in range(i , N):  # 只遍历上三角，避免重复
            if adj[i][j] > 0:
                prim_i = f"/World/Objects/{rock_name}/rock_{i}"
                prim_j = f"/World/Objects/{rock_name}/rock_{j}"

                create_attachment_between_prims(rock_name,i,j,prim_i, prim_j)
                connections.append((i, j))
    # create_attachment_between_prims("/World/Objects/base_rock_0/rock_0","/World/Objects/base_rock_0/rock_1")
    # rockRigid_cfg = RigidObjectCfg(
    #     prim_path=root_path,
    #     # spawn=cfg_rock,
    #     init_state=RigidObjectCfg.InitialStateCfg(),
    # )
    # rock_object = RigidObject(cfg=rockRigid_cfg)
    # root_view=RigidPrim(prim_paths_expr=root_path, name=f"rigid_prim_view_rock_1")
    # root_view.initialize()
    
    return Rock(rock_objects=rock_objects,rockname=rock_name, root=root, adj=adj, centers=centers, ids=ids, masses=masses,connections=connections,base_translation=base_translation)





import impact_model
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
    if not is_breakable(rock):
        print("[WARN]: Rock is not breakable.")
        return rock
    impacted,node_forces,edge_force=impact_model.apply_impact(
        rock.centers,
        rock.adj,
        impact_id,
        impact_dir,
        I0,
        gamma,
        lamb,
        k,
        alpha,
        beta,
        min_I,
        k_c,
        k_t
    )
    groups_subgraphs, groups_centers, groups_ids, groups_Object = impact_model.group_by_subgraphs(impacted, rock.centers, rock.ids, rock.rock_objects)
    # cluster_forces=impact_model.compute_cluster_force(node_forces, edge_force, groups_subgraphs)
    # print("node forces:", node_forces)
    groups_masses=[]
    for idx, group_ids in enumerate(groups_ids):
        group_masses = [rock.masses[rock.ids.index(gid)] for gid in group_ids]
        # group_meshviews = [rock.meshviews[rock.ids.index(gid)] for gid in group_ids]
        # total_mass = sum(group_masses)
        groups_masses.append(group_masses)
    # velocities = impact_model.compute_cluster_velocities(cluster_forces, groups_masses)
    velocities = impact_model.compute_node_velocities(node_forces, rock.masses)
    # print("node velocities:", velocities)
    # print("Cluster forces:", cluster_forces)
    # print("Cluster masses:", groups_masses)
    # print("Cluster velocities:", velocities)
    print(f"[INFO]: Created {len(groups_subgraphs)} broken groups.")
    update_break_meshes_new_new(impacted, rock)
    # 设置碎块速度
    for idx,rock_object in enumerate(rock.rock_objects):
        speed = velocities[idx]*5
        # add rotational velocity to the end of speed
        speed_with_rot = np.concatenate([speed, np.zeros(3)], axis=0)
        # convert to torch tensor and set to sim
        speed_torch = torch.tensor(speed_with_rot, dtype=torch.float32, device="cuda:0").unsqueeze(0)
        # force = node_forces[idx]
        # force_torch = torch.tensor(force, dtype=torch.float32, device="cuda:0").reshape(1, 3)
        # print(force_torch)
        # torques=torch.zeros(0, 0, 3)
        # print(torques)
        # rock_object.set_external_force_and_torque(forces=force_torch,torques=torch.zeros(0, 0, 3))
        # rock_object.write_data_to_sim()

    
    rocks = []    
    for idx, group_ids in enumerate(groups_ids):
        connections=[]
        N=len(groups_ids[idx])
        for i in range(N):
            for j in range(0 , N):  # 只遍历上三角，避免重复
                if groups_subgraphs[idx][i][j] > 0:
                    connections.append((groups_ids[idx][i], groups_ids[idx][j]))

        rocks.append(Rock(rock_objects=groups_Object[idx],rockname=rock.rockname, root=rock.root, adj=groups_subgraphs[idx], centers=groups_centers[idx], ids=groups_ids[idx], masses=groups_masses[idx],connections=connections,base_translation=rock.base_translation))
    
    return rocks


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
        
def update_rocks_state(rocks, dt=0.02):
        # 如果rocks是一个列表，遍历更新每个rock的状态
    if isinstance(rocks, list):
        for rock in rocks:
            update_rock_state(rock, dt)
    else:
        # 如果rocks是一个单一的rock对象，直接更新它的状态
        update_rock_state(rocks, dt)