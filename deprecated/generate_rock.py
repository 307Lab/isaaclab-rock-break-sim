

from pxr import UsdGeom, Gf, UsdPhysics, Sdf, Usd, Gf
import numpy as np
import prebreakv2

def export_meshes_to_usd(
    fragments,
    masses,
    rock_data,
    usd_path="rock.usd",
    root_path="/World/rock_0",
    base_translation=(0, 0, 0)
):
    """
    将碎石和Rock信息导出到USD文件
    """
    
    stage = Usd.Stage.CreateNew(usd_path)
    root_xform = UsdGeom.Xform.Define(stage, "/World")

    root = UsdGeom.Xform.Define(stage, root_path)
    root_prim = root.GetPrim()
    
    rigid_api = UsdPhysics.RigidBodyAPI.Apply(root_prim)
    rigid_api.CreateRigidBodyEnabledAttr(True)

    # 可以给root加上自定义属性存储Rock数据
    root_prim.CreateAttribute("adjacency_matrix", Sdf.ValueTypeNames.FloatArray).Set(rock_data['adj'].flatten().tolist())
    root_prim.CreateAttribute("centers", Sdf.ValueTypeNames.FloatArray).Set(rock_data['centers'].flatten().tolist())
    root_prim.CreateAttribute("ids", Sdf.ValueTypeNames.IntArray).Set(rock_data['ids'])
    root_prim.CreateAttribute("masses", Sdf.ValueTypeNames.FloatArray).Set(rock_data['masses'])

    mesh_prims = []
    for i, (mesh, mass) in enumerate(zip(fragments, masses)):
        prim_path = f"{root_path}/rock_{i}"
        usd_mesh = UsdGeom.Mesh.Define(stage, prim_path)

        vertices = mesh.vertices
        faces = mesh.faces

        usd_mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in vertices])
        usd_mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        usd_mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())

        prim = usd_mesh.GetPrim()
        UsdGeom.XformCommonAPI(prim).SetResetXformStack(True)

        # 刚体和质量
        # UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)

        # 碰撞
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")

        # 设置位置


        # 随机扰动（防止完全重叠爆炸）
        # rand = np.random.uniform(-1, 1, 3) * random_offset_scale

        # final_pos = np.array(base_translation)
        # UsdGeom.XformCommonAPI(prim).SetTranslate(tuple(final_pos))

        mesh_prims.append(prim)
    

    # 设置默认 Prim
    stage.SetDefaultPrim(root_xform.GetPrim())
    # # 显式指定 prim 路径（不依赖 defaultPrim）
    # root_prim.GetReferences().AddReference(
    #     assetPath="rock.usd",
    #     primPath="/Rock"  # 明确指定 rock.usd 里的具体 prim 路径
    # )
    
    flattened_stage = stage.Flatten()
    flattened_stage.Export(usd_path)
    print(f"[INFO] Exported {len(fragments)} meshes and rock data to {usd_path}")
    return root_prim, mesh_prims

def generate_and_export_prebroken_rock(
    num_points=40,
    scale=0.5,
    num_cells=25,
    usd_path="rock.usd",
    root_path="/World/rock_0",
    base_translation=(0, 0, 10),
    seed=None
):
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

    rock_data = {
        "rock": rock,
        "fragments": fragments,
        "adj": adj,
        "centers": centers,
        "ids": ids,
        "masses": masses
    }

    root_prim, mesh_prims = export_meshes_to_usd(
        fragments,
        masses,
        rock_data,
        usd_path=usd_path,
        root_path=root_path,
        base_translation=base_translation
    )
    
    return True



if __name__ == "__main__":
    generate_and_export_prebroken_rock()