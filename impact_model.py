import numpy as np
from collections import deque

def apply_impact(
    centers,
    adj_matrix,
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
    n = len(centers)
    d = impact_dir / np.linalg.norm(impact_dir)

    new_adj = adj_matrix.copy()

    visited = np.zeros(n, dtype=bool)

    # BFS队列： (节点, 当前冲击强度)
    queue = deque()
    queue.append((impact_idx, I0))
    visited[impact_idx] = True
    node_force = np.zeros((n, 3))
    edge_force = {}

    while queue:
        i, I_i = queue.popleft()

        if I_i < min_I:
            continue

        # 找邻居（只考虑有效边）
        neighbors = np.where(new_adj[i] > 0)[0]

        for j in neighbors:

            A = new_adj[i, j]
            if A <= 0:
                continue

            vec = centers[j] - centers[i]
            dist = np.linalg.norm(vec)
            if dist < 1e-6:
                continue

            e = vec / dist

            # 方向因子
            cos_theta = np.dot(d, e)

            if cos_theta > 0:
            # 压缩
                dir_factor = cos_theta
                T = k_c * (A ** alpha)
            else:
            # 拉伸
                dir_factor = abs(cos_theta)
                T = k_t * (A ** alpha)   # 更容易断

            # 冲击传播
            I_ij = I_i * dir_factor * np.exp(-lamb * dist) * gamma


            # 破坏连接
            if I_ij > T:
                damage = beta * (I_ij - T)

                new_adj[i, j] -= damage
                new_adj[j, i] -= damage
                
            else:
                # 没有破坏，但也会衰减
                new_adj[i, j] *= (1 - beta * I_ij / T)
                new_adj[j, i] *= (1 - beta * I_ij / T)

            # 如果还有冲击，就继续传播
            if not visited[j] and I_ij > min_I:
                queue.append((j, I_ij))
                visited[j] = True
                
            # ===== 3. 边力（关键新增）=====
            # 正：压缩 / 负：拉伸
            edge_F_scalar = I_ij - T
            edge_F_vec = edge_F_scalar * e

            edge_force[(i, j)] = edge_F_vec

            # ===== 4. 节点力累加 =====
            node_force[i] += -edge_F_vec
            node_force[j] += edge_F_vec

    return new_adj, node_force, edge_force

# find the clostest point in centers to the impact point
def find_closest_node(centers, impact_point):
    min_dist = float('inf')
    closest_idx = -1
    for i, c in enumerate(centers):
        dist = np.linalg.norm(c - impact_point)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
    return closest_idx  

# generate the connective subgraphes of the damaged graph
def get_damaged_subgraphes(adj_matrix,centers,ids):
    subgraphs = []
    n = len(adj_matrix)
    visited = np.zeros(n, dtype=bool)

    def dfs(i, subgraph):
        visited[i] = True
        for j in range(n):
            if adj_matrix[i, j] > 0 and not visited[j]:
                subgraph[i, j] = adj_matrix[i, j]
                subgraph[j, i] = adj_matrix[j, i]
                dfs(j, subgraph)

    for i in range(n):
        if not visited[i]:
            subgraph = np.zeros_like(adj_matrix)
            dfs(i, subgraph)
            subgraphs.append(subgraph)

    return subgraphs

import numpy as np

def group_by_subgraphs(adj_matrix, centers, ids, rock_objects):
    n = len(adj_matrix)
    visited = np.zeros(n, dtype=bool)

    groups_centers = []
    groups_ids = []
    groups_Object = []
    groups_subgraphs = []

    def dfs(i, comp_nodes):
        visited[i] = True
        comp_nodes.append(i)
        for j in range(n):
            if adj_matrix[i, j] > 0 and not visited[j]:
                dfs(j, comp_nodes)

    for i in range(n):
        if not visited[i]:
            comp_nodes = []
            dfs(i, comp_nodes)

            comp_nodes = np.array(comp_nodes)

            # 提取子图
            subgraph = adj_matrix[np.ix_(comp_nodes, comp_nodes)]

            # 重组 centers 和 ids
            groups_centers.append(centers[comp_nodes])
            groups_ids.append([ids[k] for k in comp_nodes])
            groups_subgraphs.append(subgraph)
            groups_Object.append([rock_objects[k] for k in comp_nodes])

    return groups_subgraphs, groups_centers, groups_ids, groups_Object

def compute_cluster_force(node_forces, edge_forces, subgraphs):
    cluster_forces = []

    for nodes in subgraphs:
        nodes = np.asarray(nodes).flatten().astype(int)

        F = np.zeros(3)

        node_set = set(nodes)

        # 节点力
        for i in nodes:
            F += node_forces[i]

        # 边力
        for (i, j), ef in edge_forces.items():
            if i in node_set and j in node_set:
                F += ef

        cluster_forces.append(F)

    return cluster_forces

def compute_cluster_velocities(cluster_forces, masses, dt=0.02):
    """
    cluster_forces: list[(3,)] 每个碎块合力
    masses: list[float]       每个碎块质量
    dt: 时间尺度（冲击持续时间）
    """
    velocities = []

    for F, m in zip(cluster_forces, masses):

        if m <= 1e-6:
            velocities.append(np.zeros(3))
            continue

        v = (F / m) * dt
        velocities.append(v)

    return velocities

def compute_node_velocities(node_forces, masses, dt=0.02):
    node_velocities = []
    for F, m in zip(node_forces, masses):
        if m <= 1e-6:
            node_velocities.append(np.zeros(3))
            continue
        v = (F / m) * dt
        node_velocities.append(v)
    return node_velocities