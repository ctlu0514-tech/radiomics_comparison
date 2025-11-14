import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# 1. 加载图数据
def load_graph(path):
    """从边列表文件中加载图"""
    G = nx.read_edgelist(path, nodetype=int)
    return G

# 2. 预处理邻居信息
def precompute_neighbors(G):
    """为每个节点存储其邻居节点,键是图中的节点编号，值是该节点的邻居节点集合（即与该节点有边连接的其他节点）。"""
    neighbors = {node: set(G.neighbors(node)) for node in G.nodes()}
    return neighbors

# 3. 计算节点相似度
def sim(neighbors, v_i, v_j):
    """计算两个节点的相似度（共同邻居数）"""
    return len(neighbors[v_i] & neighbors[v_j])

# 4. 选择初始代表节点（Exemplars）
def select_exemplars(G, degrees, k_max, neighbors):
    """选择最多 k_max 个初始代表节点，并返回每个代表节点的 possibility 值
    输入参数
        G: 网络图 (networkx 图对象）
        k_max: 最多选择的代表节点数量
        neighbors: 预处理的邻居信息（由 precompute_neighbors 函数生成）
    输出
        exemplars: 选定的代表节点列表
        possibility_values: 每个代表节点的可能性值列表
    """
    exemplars = []  # 存储代表节点
    possibility_values = []  # 存储每个代表节点的 possibility 值
    max_sim_dict = {}  # 维护每个节点与已选代表节点的最大相似度

    for l in range(k_max):  # 最多选择 k_max 个代表节点
        # print(l)
        if l == 0:
            # 第一个代表节点选择度数最大的节点
            e_l = max(degrees, key=degrees.get)
            P_l = degrees[e_l]  # 第一个节点的 possibility 值就是其度数
            exemplars.append(e_l)
            possibility_values.append(P_l)
            # 初始化max_sim_dict，记录所有节点与第一个代表节点的相似度
            for node in G.nodes():
                if node == e_l or degrees[node] == 0:
                    continue
                current_sim = sim(neighbors, node, e_l)
                max_sim_dict[node] = current_sim
        else:
            # 后续代表节点基于度数和与已选代表节点的相似度
            P_l_dict = {}
            for node in G.nodes():  # 遍历所有的节点，计算当前节点的可能值
                if node in exemplars or degrees[node] == 0:
                    continue
                max_sim = max_sim_dict.get(node, 0)
                P_l = degrees[node] / (max_sim + 1)
                P_l_dict[node] = P_l
                
            e_l = max(P_l_dict, key=P_l_dict.get)  # 选择 possibility 值最大的节点
            P_l_value = P_l_dict[e_l]  # 获取该节点的 possibility 值
            exemplars.append(e_l)  # 将选择的代表节点加入列表
            possibility_values.append(P_l_value)  # 记录该节点的 possibility 值

            # 更新max_sim_dict：计算候选节点与新代表节点的相似度并更新最大值
            for node in G.nodes():
                if node in exemplars or degrees[node] == 0:
                    continue
                current_sim = sim(neighbors, node, e_l)
                if current_sim > max_sim_dict.get(node, 0):
                    max_sim_dict[node] = current_sim

    # print('代表节点与相似度：' , exemplars, possibility_values)
    return exemplars, possibility_values  # 返回代表节点和 possibility 值

# 5. 初始化社区划分
def initial_partition(G, degrees, exemplars, neighbors, k):
    """
    根据前k个代表节点初始化社区划分
    输出:
        partition: 存储了每个节点的社区分配情况，键是节点编号，值是该节点所属的社区编号。
    """
    partition = {}  # 存储每个节点的社区分配
    for node in G.nodes():  # 遍历图中的每个节点
        if degrees[node] == 0:
            continue
        max_sim = -1
        assigned_l = 0
        for l in range(k):  # 仅遍历估计的前k个代表节点
            e = exemplars[l]  # 获取第 l 个代表节点
            current_sim = sim(neighbors, node, e)  # 计算节点与代表节点的相似度
            if current_sim > max_sim:
                max_sim = current_sim
                assigned_l = l  # 分配到相似度最高的社区
        partition[node] = assigned_l
    return partition

# 辅助函数：计算 delta_i
def compute_delta_i(R, node, k):
    """计算节点 node 的重要性集中度 delta_i"""
    f_li_all = [R[l][node] for l in range(k)]  # 是列表，存储了节点 node 在各个社区中的局部重要性值
    sum_f = sum(f_li_all)
    if sum_f == 0:
        delta_i = 0
    else:
        delta_i = np.sqrt(sum((f_li / sum_f) ** 2 for f_li in f_li_all))
    return delta_i

def estimate_k(possibility_values, drop_threshold):
    """
    根据可能性值的下降幅度估计社区数量 k
    参数:
        possibility_values (list): 每个代表节点的可能性值列表
        drop_ratio_threshold (float): 下降比率阈值，默认0.5（下降超过50%）
    返回:
        int: 估计的社区数量 k
    """
    candidates = []
    for i in range(1, len(possibility_values)):
        # 计算当前可能性值与上一个的比率
        ratio = possibility_values[i] / possibility_values[i-1]
        # 如果下降幅度超过阈值，记录当前i作为候选k值
        if ratio < drop_threshold:
            candidates.append(i)  # 注意：i对应第i+1个节点，例如i=0对应第一个节点后的下降点
    
    # 如果没有候选值，返回全部可能性值的数量（即未检测到明显下降）
    if not candidates:
        return len(possibility_values)
    else:
        # 选择最大的候选k值（例如图2(b)中的k=6）
        return max(candidates)

# 6. 计算社区描述模型 R
def compute_R(G, degrees, partition, k, neighbors):
    """
    计算社区描述模型 R
    参数：
        G (networkx.Graph): 输入的图数据结构。
        partition (dict): 初始的社区划分，字典的键是节点编号，值是对应的社区编号。
        k (int): 估计的社区数量。
        neighbors (dict): 预处理得到的每个节点的邻居节点集合，字典的键是节点编号，值是邻居节点的集合。
    返回值：
        R (numpy.ndarray): 社区描述模型矩阵，形状为 (k, n)，其中 k 是社区数量，n 是图中节点的数量。
                          R[l][i] 表示节点 i 属于社区 l 的重要性描述，基于局部重要性和重要性集中度的综合考量。
    """
    n = G.number_of_nodes()  # 获取图中节点的总数
    R = np.zeros((k, n))  # R 是一个 k x n 的矩阵
    community_members = {}  # 用于按社区存储节点，键是社区编号，值是属于该社区的节点列表。
    for node, comm in partition.items():
        community_members.setdefault(comm, []).append(node)  # 将同一个社区的节点收集到一起
    # 计算局部重要性 f_li
    for comm in range(k):  # 这个循环遍历所有社区索引，从 0 到 k−1
        members = community_members.get(comm, [])  # 对于每个社区 comm，获取属于该社区的节点列表 members
        n_l = len(members)  # 社区内的节点数
        for node in members:  # 遍历社区 comm 中的每个节点
            d_li = len(neighbors[node] & set(members))  # 节点与社区内的边数
            f_li = d_li / n_l  # 计算局部重要性
            R[comm][node] = f_li  # 初始化 R
    # 计算 delta_i for each node
    for node in G.nodes():
        if degrees[node] == 0:
            continue
        delta_i = compute_delta_i(R, node, k)
        for l in range(k):
            R[l][node] *= delta_i  # Update R with f_li * delta_i
    return R, community_members

# 7. 计算节点与社区的相似度 M
def compute_M(R, neighbors, degrees):
    """
    计算节点与社区的相似度 M
    输入参数：
        R: 社区描述模型矩阵, R[l][i] 表示节点 i 属于社区 l 的重要性描述。
        neighbors: 键是节点编号，值是邻居节点的集合
    输出：
        M: 一个 (n, k) 的矩阵，矩阵的元素 M[node][l] 表示节点 node 与社区 l 的相似度。
    """
    n = R.shape[1]  # 获取图中节点的总数
    k = R.shape[0]  # 获取矩阵 R 的行数，即社区的数量 k
    M = np.zeros((n, k))  # 用于存储每个节点与每个社区的相似度量
    # 遍历所有节点
    for node in neighbors:
        if degrees[node] == 0:
            continue
        # 对于每个节点，计算它与所有社区的相似度
        for l in range(k):
            # 值越大，说明 vi 的邻居节点在社区 Vl 中的代表性越强，因此 vi 归属于社区 Vl 的可能性也就越高。
            M[node][l] = np.sum(R[l, list(neighbors[node])])  # 使用numpy加速计算
    return M

# 8. 分配节点到社区
def assign_nodes(M, degrees):
    """将节点分配到相似度最高的社区"""
    n, k = M.shape  # 获取 M 的维度，n 是节点数，k 是社区数
    partition = {}  # 存储节点的社区分配
    
    for node in range(n):  # 遍历所有节点
        if degrees[node] == 0:
            continue
        max_l = np.argmax(M[node])  # 找到相似度最高的社区，np.argmax 返回最大值的索引
        partition[node] = max_l  # 将该节点分配给相似度最高的社区
    
    return partition

# 10. 计算 F(Ω) 
def compute_F(G, degrees, community_members, k, neighbors, R):
    """计算目标函数 F(Ω)"""
    n = G.number_of_nodes()  # 获取图中节点的总数
    F = 0  # 初始化目标函数值
    for node in G.nodes():
        if degrees[node] == 0:
            continue
        # 计算节点在所属社区的贡献度 I_i
        I_i = 0
        for l in range(k):
            members = community_members.get(l, [])
            d_li = len(neighbors[node] & set(members))  # 计算节点与社区内节点的共享邻居数
            f_li = d_li / len(members) if len(members) > 0 else 0  # 局部重要性 f_li
            I_i += d_li * f_li  # 累加贡献度
        # 计算节点的集中度 delta_i
        delta_i = compute_delta_i(R, node, k)
        # 更新目标函数值
        F += delta_i * I_i  # 对每个节点的贡献度与集中度的乘积求和
    return F

# 11. 检查是否收敛
def has_converged(old_F, new_F, tolerance=1e-6):
    """检查目标函数 F(Ω) 是否收敛"""
    return abs(new_F - old_F) < tolerance

# 辅助函数 打印社区
def format_community_partition(partition):
    """
    格式化输出社区划分
    输入:
        partition: 一个字典，键是节点编号，值是该节点所属的社区编号
    输出:
        打印每个社区及其包含的节点
    """
    community_nodes = {}  # 存储社区及其对应的节点
    for node, comm in partition.items():  # 遍历所有节点及其所属社区
        community_nodes.setdefault(comm, []).append(node)  # 将节点添加到相应社区

    # 输出每个社区的节点，按社区编号和节点编号排序
    for comm, nodes in sorted(community_nodes.items()):
        print(f"社区 {comm}: {sorted(nodes)}")

# 12. ISCD 算法主函数
def ISCD(G, degrees, k, t_m, neighbors, exemplars):
    """
    ISCD 算法主函数，用于迭代优化社区划分并计算社区描述模型和目标函数值。
    参数：
        neighbors (dict): 预处理得到的每个节点的邻居节点集合，字典的键是节点编号，值是邻居节点集合。
        exemplars (list): 选择的初始代表节点列表，用于初始化社区划分。
    返回值：
        partition (dict): 最终的社区划分，字典的键是节点编号，值是对应的社区编号。
        R (numpy.ndarray): 社区描述模型矩阵，形状为 (k, n)，其中 k 是社区数量，n 是图中节点的数量。
                          R[l][i] 表示节点 i 属于社区 l 的重要性描述，基于局部重要性和重要性集中度的综合考量。
        F_values (list): 目标函数值的变化历史，包含每次迭代后的目标函数值，用于评估社区划分质量。
    """
    partition = initial_partition(G, degrees, exemplars, neighbors, k)  # 初始化社区划分
    # print("初始社区划分:")
    # format_community_partition(partition)
    R, community_members = compute_R(G, degrees, partition, k, neighbors)  # 初始化社区描述模型 R
    F_values = []
    F_prev = compute_F(G, degrees, community_members, k, neighbors, R) # 目标函数值 F，表示当前社区划分的质量
    F_values.append(F_prev) # 将初始的 F_prev 添加到 F_values，用于跟踪每次迭代的优化效果
    for t in range(t_m):  # 迭代优化
        M = compute_M(R, neighbors, degrees)  # 计算节点间相似度 M
        new_partition = assign_nodes(M, degrees)  # 分配节点到社区
        R, community_members = compute_R(G, degrees, new_partition, k, neighbors)  # 更新 R
        F_current = compute_F(G, degrees, community_members, k, neighbors, R)
        F_values.append(F_current)
        if has_converged(F_prev, F_current):  # 检查是否收敛
            break
        F_prev = F_current
        partition = new_partition  # 更新社区划分
    return partition, R, F_values

# 主程序
def iscd_algorithm_auto_k(G):
    """
    运行 ISCD 算法并返回每个社区包含的节点列表。

    Args:
        G (networkx.Graph): 输入的图数据。

    Returns:
        dict: 每个节点所属的社区编号，格式为 {node_id: community_id}。
    """
    degrees = dict(G.degree())  # 获取每个节点的度数，键是节点编号，值是对应的度数
    neighbors = precompute_neighbors(G)  # 预处理邻居信息
    print("非零节点数：", sum(1 for degree in degrees.values() if degree > 0))  # 设置最大社区数量为非零度数节点的数量
    k_max = int(np.sqrt(G.number_of_nodes()))  # 设置最大社区数量为 sqrt(n)
    # k_max = 40
    # 选择代表节点并获取 possibility 值
    exemplars, possibility_values = select_exemplars(G, degrees, k_max, neighbors)
    
    # 估计社区数量 k
    k = estimate_k(possibility_values, drop_threshold=0.85)
    print(f"估计的社区数量 k = {k}")
    
    # 绘制可能性值曲线图
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(possibility_values)+1 ), possibility_values, marker='o', linestyle='-', color='b')
    plt.axvline(x=k, color='r', linestyle='--', label=f'Estimated k = {k}')  # 标记估计的 k 值
    plt.xlabel('Number of Exemplars (l)')
    plt.ylabel('Possibility Value (P_l)')
    plt.title('Possibility Values for Exemplar Selection')
    plt.legend()
    plt.grid(True)
    plt.savefig('/data/qh_20T_share_file/lct/CDGAFS/可能性值曲线.png', format='png')

    t_m = 100  # 最大迭代次数
    partition, R, F_values = ISCD(G, degrees, k, t_m, neighbors, exemplars)  # 运行 ISCD 算法
    
    # 格式化输出社区划分
    print("最终社区划分:")
    format_community_partition(partition)

    # # 格式化输出社区描述模型 R
    # print("\n社区描述模型 R:")
    # for l in range(k):
    #     print(f"社区 {l}: {R[l]}")
    
    # 格式化输出目标函数 F(Ω) 的变化
    print("\n目标函数 F(Ω) 的变化:")
    for i, value in enumerate(F_values):
        print(f"迭代 {i}: F(Ω) = {value:.4f}")
    
    # 返回节点到社区的映射
    return partition

    