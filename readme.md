# 石头破碎模拟

---

## 函数用法

### 石头定义

石头的可操作对象

usdTools.Rock(
    rock_objects,
    rockname, 
    root, 
    adj, 
    centers, 
    ids, 
    masses,
    connections,
    base_translation
)

#### 参数列表

- rock_objects : isaaclab.assets.RigidObject list 石头中每个碎块的 RigidObject，用于获取实时位置
- rockname : str                石块名称
- root : Prim                   石块根prim
- root_path : str               石块根prim的路径
- adj : (N,N)                   碎块的邻接矩阵，数值大于 0 表示相连，小于等于 0 表示断开，N 为碎块数量
- centers : (3,N)               每个碎块的中心位置
- ids : tuple(N)                每个碎块的id                 
- masses : tuple(N)             每个碎块的质量
- connections ： (2,N)          记录连接在一起的边
- base_translation : tuple(3)   初始偏移

---

### 生成石头

在场景中生成一个预破碎后的石块

usdTools.generate_prebroken_rock(
    num_points=40,
    scale=0.5,
    num_cells=25,
    rock_name="base_rock_0",
    root_path="/World/Objects/base_rock_0",
    base_translation=(0, 0, 10),
    seed=None
)

#### 参数列表

- num_points : int          生成的多边体石头的端点数量，默认为 40
- scale : float             生成比例，默认比例下平均半径为3 m，默认为 1
- num_cells : int           预破碎碎块的数量基数，通常生成的碎块数量为3-6倍基数，默认为 25
- rock_name : str           石头名称，需要与 root_path 中 Objects 后的名字一致，默认为 "base_rock_0"
- root_path : str           石头在 isaaclab 中生成的 prim 位置，默认为 "/World/Objects/base_rock_0"
- base_translation : tuple  石头生成位置，默认(0, 0, 0)
- seed : int                种子，None为随机生成

#### 返回值

- usdTools.Rock 单个石头对象

---

#### 应用冲击模型

应用一个冲击力到指定石块上的对应点，首先会检测石头是否还可以继续破碎，如果不能则返回石头自身，破碎完成后返回碎裂开的石头列表

usdTools.apply_impact(
    rock,
    impact_idx,
    impact_dir,
    I0=250.0,
    gamma=0.8,
    lamb=0.8,
    k=1.0,
    alpha=1.0,
    beta=0.5,
    min_I=1e-3,
    k_c = 1.0,
    k_t = 0.2
)

#### 参数列表

- rock : usdTools.Rock  需要破碎的石头对象
- impact_idx : int      破碎的子石块的id
- impact_dir : (0,3)    冲击力方向
- I0 : float            初始冲击力
- gamma : float         冲击随传播节点衰减倍率，默认为 0.8
- lamb : float          冲击随传播距离衰减倍率，默认为 0.8
- K : float             材料抗冲击系数，默认为 1
- alpha : float         非线性结构强度，默认为 1
- beta : float          冲击伤害倍率，默认为 0.5
- min_I : float         冲击力小于该值时停止计算，默认为 1e-3
- k_c : float           材料受到压缩时的强度系数，数值越低受到压缩时损伤越大，默认为 1
- k_t : float           材料受到拉伸时的强度系数，数值越低受到拉伸时损伤越大，默认为 0.2 （石块抗拉伸强度远低于抗压缩强度）

#### 返回值

- usdTools.Rock : list 碎裂后的石头列表

---

### 获取最近的石头

从石块中获取最近的一个碎块

get_closest_rock_and_id(
    rocks, 
    impact_point
)

#### 参数列表

- rocks : usdTools.Rock 单个石头或所有石头的列表
- impact_point ： (0,3) 点位置

#### 返回值

- closest_id : int              对应rock中的石块id
- closest_rock ：usdTools.Rock  最近的石块对象
- min_dist ： tensor            最近距离

---

### 状态更新

更新状态，isaaclab需要这个函数来更新每个rock对象的状态，从而更新石块的位置信息，无返回值，每个 sim.step() 后都需要执行
        
update_rocks_state(
    rocks, 
    dt=0.02
)

#### 参数列表

- rocks ： usdTools.Rock        单个石头或所有石头的列表
- dt=0.02 : float               时间步长，需要等于设置的 sim_dt

---


### 离线生成石头

生成石块到指定文件

prebreakv2.generate_prebroken_rock_and_save(num_points=40,
    scale=0.5,
    num_cells=25,
    file_name="rock_data.pkl",
    seed=None
) 

#### 参数列表

- num_points : int          生成的多边体石头的端点数量，默认为 40
- scale : float             生成比例，默认比例下平均半径为3 m，默认为 1
- num_cells : int           预破碎碎块的数量基数，通常生成的碎块数量为3-6倍基数，默认为 25
- file_name : str           要保存的石块的文件名
- seed : int                种子，None为随机生成

---

### 加载离线生成的石头

用法与 usdTools.generate_prebroken_rock 基本一致

usdTools.load_rock_from_file(
    file_name=None,
    rock_name="base_rock_0",
    root_path="/World/Objects/base_rock_0",
    base_translation=(0, 0, 0)
)

#### 参数列表

- file_name : str           石头文件的位置
- rock_name : str           石头名称，需要与 root_path 中 Objects 后的名字一致，默认为 "base_rock_0"
- root_path : str           石头在 isaaclab 中生成的 prim 位置，默认为 "/World/Objects/base_rock_0"
- base_translation : tuple  石头生成位置，默认(0, 0, 0)

#### 返回值

- usdTools.Rock 单个石头对象

---

### 重置石头

usdTools.reset_rock(
    rocks : list, 
    rock_name : str
):

#### 参数列表

- rocks : list          现有石头的列表
- rock_name : str       需要重置的石头的名称

#### 返回值

- usdTools.Rock : list  碎裂后的石头列表

---

## 使用方法

完整方法参见 start_sim.py

### 生成石头

在 isaaclab 中生成石头，需要在 sim.reset() 之前执行

```
......

def design_scene():
    # 创建基础场景
    ......

    # 生成石头
    rock=usdTools.generate_prebroken_rock(num_points=50, scale=0.5, num_cells=10, root_path="/World/Objects/base_rock_0", base_translation=(0, 0, 0), seed=None)
    # 从预先生成的石头中加载
    # rock=usdTools.load_rock_from_file(file_name="rock_data.pkl",rock_name="base_rock_0",root_path="/World/Objects/base_rock_0",base_translation=(0, 0, 0))

    return rock

# 初始化仿真环境
sim_cfg = SimulationCfg(dt=0.02,physx= PhysxCfg(enable_external_forces_every_iteration=True,))
sim = SimulationContext(sim_cfg)
sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

# 场景初始化
rock=design_scene()
    
sim.reset()
......
```

### 实时更新石头参数

```

sim_dt = sim.get_physics_dt()
while simulation_app.is_running():
        
    # 执行仿真步
    sim.step()
    # 更新石头参数
    usdTools.update_rocks_state(rock, sim_dt)

```

### 寻找离预计冲击点最近的石块

```
    closest_id, rock, distance = usdTools.get_closest_rock_and_id(rock, impact_point=[0, 0, 10])
```

### 冲击最近的石块

```
    rock=usdTools.apply_impact(rock, impact_idx=closest_id, impact_dir=[0, -1, 0])
```


### 重置破碎后的石头

```
    rock=usdTools.reset_rock(rock,"base_rock_0")
```
