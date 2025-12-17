import taichi as ti
ti.init(arch=ti.cuda)  # Alternatively, ti.init(arch=ti.cpu)

n = 128 
quad_size = 1.0 / n # distance between x or z 相邻点
dt = 4e-2 / n
substeps = int(1 / 60 // dt)

gravity = ti.Vector([0, -9.8, 0]) # 重力被定义在y轴负方向
spring_Y = 3e4 # 弹簧弹性系数
dashpot_damping = 1e4 # 两个质点相对移动导致的阻尼系数
# 一个给定点最多可能受到12个邻点的影响，其他质点的影响不计，通过弹簧对给定点施加内力
# 内力宽泛地表示弹簧弹性形变导致的内力，以及两点间相对运动导致的阻尼
drag_damping = 1 # 弹簧阻尼系数

avg_speed = ti.field(dtype=float, shape=())


table_y = 0.0
table_mu = 2.0
table_bounds = 0.8

ball_radius = 0.1 # 球中心是一维field，唯一元素是三维浮点数向量
ball_center = ti.Vector.field(3, dtype=float, shape=(1, ))
ball_center[0] = [0, table_y+ball_radius, 0] # 放在桌子上

# ---- clamps (tool) ----
num_clamps = 2
clamp_pos = ti.Vector.field(3, dtype=float, shape=(num_clamps,))
clamp_prev = ti.Vector.field(3, dtype=float, shape=(num_clamps,))
clamp_vel = ti.Vector.field(3, dtype=float, shape=(num_clamps,))

clamp_on = ti.field(dtype=ti.i32, shape=())
clamp_on[None] = 0

clamp_k = 5e4   # 夹持刚度（越大越“夹死”）
clamp_c = 6e3   # 阻尼（越大越稳，但太大可能拖不动）

clamp_half_x = 0.06   # 夹板半宽（x方向）
clamp_half_y = 0.02   # 夹板半厚（y方向）
clamp_half_z = 0.06   # 夹板半深（z方向）
clamp_color  = (0.1, 0.1, 0.9)

clamp_vertices = ti.Vector.field(3, dtype=float, shape=num_clamps * 16)
clamp_indices  = ti.field(int, shape=num_clamps * 72)
clamp_gap_open  = 0.12   # 初始张开缝隙（明显）
clamp_gap_close = 0.02   # 夹住后的缝隙
clamp_gap = ti.field(dtype=ti.f32, shape=())
clamp_gap[None] = clamp_gap_open

clamp_R = 3
clamp_dpos = ti.Vector.field(3, dtype=ti.f32, shape=(num_clamps, 2*clamp_R+1, 2*clamp_R+1))

tear_ratio = 2.8   # 越小越容易撕（1.6~2.0 常用）
enable_tear = ti.field(dtype=ti.i32, shape=())
enable_tear[None] = 0

tear_ratio_dyn = ti.field(dtype=ti.f32, shape=())
tear_ratio_dyn[None] = 3.0   # 抬起阶段阈值大

# 夹住的两个布点 index（固定夹两侧）
clamp_ij = ti.Vector.field(2, dtype=ti.i32, shape=(num_clamps,))

# x(position), v(velocities)是nxn的field，有3d漂浮点的向量
x = ti.Vector.field(3, dtype=float, shape=(n, n))
v = ti.Vector.field(3, dtype=float, shape=(n, n))

num_triangles = (n - 1) * (n - 1) * 2
# cloth作为三角网格，由两个field表示，vertices是顶点
DUMMY = n * n
num_verts = n * n + 1
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=float, shape=num_verts)
colors = ti.Vector.field(3, dtype=float, shape=num_verts)

# table mesh
table_vertices = ti.Vector.field(3, dtype=float, shape=8)
table_indices = ti.field(int, shape = 12 * 3)
table_color = (0.2,0.2,0.2)
table_half = 1.0
table_thick = 0.02

bending_springs = False
# kernel装饰器可以并行计算
@ti.kernel
def initialize_mass_points():
    # 随机的offset
    random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.1
    # 轻微移动cloth，随机偏离
    for i, j in x:
        x[i, j] = [
            i * quad_size - 0.5 + random_offset[0], 0.9,
            j * quad_size - 0.5 + random_offset[1]
        ]
        v[i, j] = [0, 0, 0] # 初速度为0

# cloth由nxn的质点网络表示
# 或者是由小正方形组成的n-1 x n-1的网络
# 每个正方形作为两个三角形渲染
# 共有 (n-1) * (n-1) * 2个三角形
# 每个三角形都由vertices field中的三个整数表示
# 记录了vertices field中三角形的顶点索引
# 该函数只需要被调用一次，三角顶点的索引不变，实际上只有位置改变
@ti.kernel
def initialize_mesh_indices():
    for i, j in ti.ndrange(n - 1, n - 1):
        quad_id = (i * (n - 1)) + j
        # 1st triangle of the square
        indices[quad_id * 6 + 0] = i * n + j
        indices[quad_id * 6 + 1] = (i + 1) * n + j
        indices[quad_id * 6 + 2] = i * n + (j + 1)
        # 2nd triangle of the square
        indices[quad_id * 6 + 3] = (i + 1) * n + j + 1
        indices[quad_id * 6 + 4] = i * n + (j + 1)
        indices[quad_id * 6 + 5] = (i + 1) * n + j

    for i, j in ti.ndrange(n, n):
        if (i // 4 + j // 4) % 2 == 0:
            colors[i * n + j] = (0.22, 0.72, 0.52)
        else:
            colors[i * n + j] = (1, 0.334, 0.52)

# M1:table
@ti.kernel
def init_table_mesh():
    y0 = table_y - table_thick
    y1 = table_y
    table_vertices[0] = [-table_half, y0, -table_half]
    table_vertices[1] = [table_half, y0, -table_half]
    table_vertices[2] = [table_half, y0, table_half]
    table_vertices[3] = [-table_half, y0, table_half]
    table_vertices[4] = [-table_half, y1, -table_half]
    table_vertices[5] = [ table_half, y1, -table_half]
    table_vertices[6] = [ table_half, y1, table_half]
    table_vertices[7] = [ -table_half, y1, table_half]

    idx = ti.Vector([
        # bottom
        0, 2, 1, 0, 3, 2,
        # top
        4, 5, 6, 4, 6, 7,
        # front(-z)
        0, 1, 5, 0, 5, 4,
        # back(+z)
        3, 7, 6, 3, 6, 2,
        # left(-x)
        0, 4, 7, 0, 7, 3,
        # right(+x)
        1, 2, 6, 1, 6, 5
    ])
    for k in range(36):
        table_indices[k] = idx[k]

# M2:speed and velocity
@ti.kernel
def compute_avg_speed():
    s = 0.0
    for I in ti.grouped(v):
        s += v[I].norm()
    avg_speed[None] = s / (n * n)

# M3:add clamps
@ti.kernel
def init_clamps():
    clamp_ij[0] = ti.Vector([2,      n // 2])
    clamp_ij[1] = ti.Vector([n - 3,  n // 2])

    for k in range(num_clamps):
        clamp_pos[k]  = ti.Vector([0.0, table_y + 2.0, 0.0])  # 初始丢到高处（不生效）
        clamp_prev[k] = clamp_pos[k]
        clamp_vel[k]  = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def update_clamp_velocity(inv_dt: float):
    for k in range(num_clamps):
        clamp_vel[k] = (clamp_pos[k] - clamp_prev[k]) * inv_dt
        clamp_prev[k] = clamp_pos[k]

# 关键：进入第二阶段时，把夹板“瞬移”到布点位置，避免一夹就暴力抽飞
@ti.kernel
def attach_clamps_to_cloth():
    for ck in range(num_clamps):
        ij = clamp_ij[ck]
        ci, cj = ij[0], ij[1]
        cpos = x[ci, cj]

        clamp_pos[ck]  = cpos
        clamp_prev[ck] = cpos
        clamp_vel[ck]  = ti.Vector([0.0, 0.0, 0.0])

        for di, dj in ti.ndrange((-clamp_R, clamp_R+1), (-clamp_R, clamp_R+1)):
            ii = ti.max(0, ti.min(n-1, ci + di))
            jj = ti.max(0, ti.min(n-1, cj + dj))
            clamp_dpos[ck, di + clamp_R, dj + clamp_R] = x[ii, jj] - cpos

@ti.kernel
def init_clamp_mesh_indices():
    base = ti.Vector([
        0, 2, 1, 0, 3, 2,       # bottom
        4, 5, 6, 4, 6, 7,       # top
        0, 1, 5, 0, 5, 4,       # front(-z)
        3, 7, 6, 3, 6, 2,       # back(+z)
        0, 4, 7, 0, 7, 3,       # left(-x)
        1, 2, 6, 1, 6, 5        # right(+x)
    ])
    for c in range(num_clamps):
        # jaw0 vertices offset = c*16 + 0
        # jaw1 vertices offset = c*16 + 8
        for jaw in range(2):
            v0 = c * 16 + jaw * 8
            i0 = c * 72 + jaw * 36
            for t in range(36):
                clamp_indices[i0 + t] = v0 + base[t]


@ti.kernel
def update_clamp_mesh_vertices():
    for c in range(num_clamps):
        p = clamp_pos[c]
        g = clamp_gap[None]

        # 两片夹板的中心分别在 p.y ± g/2
        for jaw in range(2):
            sign = 1.0 if jaw == 0 else -1.0
            pc = ti.Vector([p.x, p.y + sign * (g * 0.5 + clamp_half_y), p.z])

            x0 = pc.x - clamp_half_x
            x1 = pc.x + clamp_half_x
            y0 = pc.y - clamp_half_y
            y1 = pc.y + clamp_half_y
            z0 = pc.z - clamp_half_z
            z1 = pc.z + clamp_half_z

            b = c * 16 + jaw * 8
            clamp_vertices[b + 0] = [x0, y0, z0]
            clamp_vertices[b + 1] = [x1, y0, z0]
            clamp_vertices[b + 2] = [x1, y0, z1]
            clamp_vertices[b + 3] = [x0, y0, z1]
            clamp_vertices[b + 4] = [x0, y1, z0]
            clamp_vertices[b + 5] = [x1, y1, z0]
            clamp_vertices[b + 6] = [x1, y1, z1]
            clamp_vertices[b + 7] = [x0, y1, z1]


initialize_mesh_indices()
init_table_mesh()
init_clamps()
init_clamp_mesh_indices()

# cloth作为一个有质量的弹簧网格进行建模，假设质点的相对index是(0,0)至少被12个周围点影响
# spring offsets，用来存储受影响点的相对index的列表
spring_offsets = []
for i in (-1, 0, 1):
    for j in (-1, 0, 1):
        if (i, j) != (0, 0):
            spring_offsets.append(ti.Vector([i, j]))

ns = len(spring_offsets)
import math
rest_len = []
opp = []
for k, off in enumerate(spring_offsets):
    dx, dz = int(off[0]), int(off[1])
    rest_len.append(quad_size * math.sqrt(dx*dx + dz*dz))

for k, off in enumerate(spring_offsets):
    dx, dz = int(off[0]), int(off[1])
    # 找相反方向在 spring_offsets 里的 index
    target = (-dx, -dz)
    j = None
    for t, of2 in enumerate(spring_offsets):
        if int(of2[0]) == target[0] and int(of2[1]) == target[1]:
            j = t
            break
    assert j is not None
    opp.append(j)

# 每个点、每种offset对应的弹簧是否“还活着”
# ---- Taichi 侧存储：kernel 内可动态索引 ----
spring_off = ti.Vector.field(2, dtype=ti.i32, shape=(ns,))
spring_L0  = ti.field(dtype=ti.f32, shape=(ns,))
spring_opp = ti.field(dtype=ti.i32, shape=(ns,))

for k, off in enumerate(spring_offsets):
    spring_off[k] = [int(off[0]), int(off[1])]
    spring_L0[k]  = rest_len[k]
    spring_opp[k] = opp[k]

# ---- 每个点、每种 offset 的弹簧是否存活 ----
spring_alive = ti.field(dtype=ti.i32, shape=(n, n, ns))

@ti.kernel
def init_springs():
    for i, j, k in spring_alive:
        spring_alive[i, j, k] = 1

def find_off(dx, dz):
    for k, off in enumerate(spring_offsets):
        if int(off[0]) == dx and int(off[1]) == dz:
            return k
    raise RuntimeError("offset not found")

k_x  = find_off(1, 0)    # (i,j) -> (i+1,j)
k_z  = find_off(0, 1)    # (i,j) -> (i,j+1)
k_d1 = find_off(1, 1)    # diag
k_d2 = find_off(-1, 1)
k_d_p = find_off(1,  1)   # (i,j) -> (i+1,j+1)
k_d_m = find_off(1, -1)   # (i,j) -> (i+1,j-1)


@ti.kernel
def make_notch():
    mid = n // 2
    i = mid - 1
    j0 = n // 2 - 4
    L  = 12
    for t in range(L):
        j = j0 + t
        if 0 <= j < n:
            # 1) 断中线结构边： (mid-1,j) <-> (mid,j)
            spring_alive[i, j, k_x] = 0
            spring_alive[i + 1, j, spring_opp[k_x]] = 0

            # 2) 断跨缝对角（上方那条）：(mid-1,j) <-> (mid,j+1)
            if j + 1 < n:
                spring_alive[i, j, k_d_p] = 0
                spring_alive[i + 1, j + 1, spring_opp[k_d_p]] = 0

            # 3) 断跨缝对角（下方那条）：(mid-1,j+1) <-> (mid,j)
            if j + 1 < n:
                spring_alive[i, j + 1, k_d_m] = 0
                spring_alive[i + 1, j, spring_opp[k_d_m]] = 0


@ti.kernel
def update_mesh_indices_by_tears():

    for i, j in ti.ndrange(n - 1, n - 1):  # 遍历网格中的每个四边形（除了边界）
        quad_id = i * (n - 1) + j
        v00 = i * n + j
        v10 = (i + 1) * n + j
        v01 = i * n + (j + 1)
        v11 = (i + 1) * n + (j + 1)

        # # 共有边：v10 <-> v01，用 k_d2 = (-1, +1) 表示从 v10 指向 v01
        # diag_alive = spring_alive[i + 1, j, k_d2]
        diag_a = spring_alive[i + 1, j, k_d2]       # v10<->v01  (-1,+1)
        diag_b = spring_alive[i, j, k_d_p]          # v00<->v11  (+1,+1)
        diag_alive = 1 if (diag_a != 0 or diag_b != 0) else 0

        # tri1: (v00, v10, v01)
        alive1 = 1
        if spring_alive[i, j, k_x] == 0: alive1 = 0      # v00-v10
        if spring_alive[i, j, k_z] == 0: alive1 = 0      # v00-v01
        if diag_alive == 0: alive1 = 0                   # v10-v01

        if alive1 == 1:
            indices[quad_id * 6 + 0] = v00
            indices[quad_id * 6 + 1] = v10
            indices[quad_id * 6 + 2] = v01
        else:
            indices[quad_id * 6 + 0] = DUMMY
            indices[quad_id * 6 + 1] = DUMMY
            indices[quad_id * 6 + 2] = DUMMY

        # tri2: (v11, v01, v10)
        alive2 = 1
        if spring_alive[i, j + 1, k_x] == 0: alive2 = 0  # v01-v11
        if spring_alive[i + 1, j, k_z] == 0: alive2 = 0  # v10-v11
        if diag_alive == 0: alive2 = 0                   # v01-v10

        if alive2 == 1:
            indices[quad_id * 6 + 3] = v11
            indices[quad_id * 6 + 4] = v01
            indices[quad_id * 6 + 5] = v10
        else:
            indices[quad_id * 6 + 3] = DUMMY
            indices[quad_id * 6 + 4] = DUMMY
            indices[quad_id * 6 + 5] = DUMMY

# 弹簧质点系统受到的重力、内力、阻尼、碰撞等效果的积累
@ti.kernel
def substep():
    # 0) 重力
    for i in ti.grouped(x):
        v[i] += gravity * dt # 对field里所有元素施加重力循环
        # for循环将x作为一个一维数组自动遍历其中所有元素，不论形状
    # 1) 内力：弹簧 + dashpot + 撕裂
    for i in ti.grouped(x):
        force = ti.Vector([0.0, 0.0, 0.0])

        for k in range(ns):
            if spring_alive[i[0], i[1], k] == 0:
                continue

            off2 = spring_off[k]     # (dx, dz)
            L0   = spring_L0[k]
            ok   = spring_opp[k]

            j = i + off2
            if 0 <= j[0] < n and 0 <= j[1] < n:
                x_ij = x[i] - x[j]
                v_ij = v[i] - v[j]
                dist = x_ij.norm() + 1e-6
                d = x_ij / dist

                stretch = dist / (L0 + 1e-6)


                # ====== “中线竖缝”撕裂（从中间撕成两半）======
                mid = n // 2

                # 只考虑结构边 (±1, 0)，也就是 i 方向的相邻点弹簧
                is_struct_x = (off2[1] == 0) and (ti.abs(off2[0]) == 1)

                # 只允许跨过中线那条缝：i = mid-1 <-> mid
                # 两个方向都要覆盖：从 mid-1 指向 mid（dx=+1），以及从 mid 指向 mid-1（dx=-1）

                is_mid_seam = is_struct_x and (
                    (i[0] == mid - 1 and off2[0] == 1) or
                    (i[0] == mid     and off2[0] == -1)
                )

                if enable_tear[None] == 1 and is_mid_seam and stretch > tear_ratio_dyn[None]:
                    # 断结构边（双向）
                    spring_alive[i[0], i[1], k] = 0
                    spring_alive[j[0], j[1], ok] = 0
                # 额外：断掉该段附近跨缝对角（两条）
                    # 统一用 L 表示中线左侧点 (mid-1, j)
                    Lp = i if (i[0] == mid - 1) else j
                    li, lj = Lp[0], Lp[1]

                    # (mid-1,lj) <-> (mid,lj+1)
                    if lj + 1 < n:
                        spring_alive[li, lj, k_d_p] = 0
                        spring_alive[li + 1, lj + 1, spring_opp[k_d_p]] = 0

                        # (mid-1,lj+1) <-> (mid,lj)
                        spring_alive[li, lj + 1, k_d_m] = 0
                        spring_alive[li + 1, lj, spring_opp[k_d_m]] = 0
                    continue

                # ================================================

                # 弹簧力 + dashpot
                force += -spring_Y * d * (stretch - 1.0)
                force += -v_ij.dot(d) * d * dashpot_damping * quad_size

        v[i] += force * dt

    

    for i in ti.grouped(x):
        # 1) 空气阻尼
        v[i] *= ti.exp(-drag_damping * dt)

        # 2) 积分
        x[i] += dt * v[i]

        # M2升级碰撞模型
        # 3）球体碰撞：位置投影+正常速度投影
        offset = x[i] - ball_center[0]
        dist = offset.norm()
        if dist < ball_radius:
            nrm = offset / (dist + 1e-6)
            # push point to sphere surface
            x[i] = ball_center[0] + ball_radius * nrm
            # remove inward normal velocity
            vn = v[i].dot(nrm)
            if vn < 0:
                v[i] -= vn * nrm
    
        # 4) 桌面碰撞=位置投影+正常速度+tan摩擦
        # M1：桌面碰撞
        eps = 1e-4
        if x[i].y < table_y + eps:
            x[i].y = table_y + eps
            if v[i].y < 0:
                v[i].y = 0

            v[i].x *= ti.exp(-table_mu * dt)
            v[i].z *= ti.exp(-table_mu * dt)

        # 5）额外的碰撞边界
        x[i].x = ti.max(-table_bounds, ti.min(table_bounds, x[i].x))
        x[i].z = ti.max(-table_bounds, ti.min(table_bounds, x[i].z))

    if clamp_on[None] == 1:
        for ck in range(num_clamps):
            ci = clamp_ij[ck][0]
            cj = clamp_ij[ck][1]
            for di, dj in ti.ndrange((-clamp_R, clamp_R+1), (-clamp_R, clamp_R+1)):
                ii = ti.max(0, ti.min(n-1, ci + di))
                jj = ti.max(0, ti.min(n-1, cj + dj))
                x[ii, jj] = clamp_pos[ck] + clamp_dpos[ck, di + clamp_R, dj + clamp_R]
                v[ii, jj] = clamp_vel[ck]


# 每个帧都要调用update_vertices，顶点在模拟中持续更新
@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = x[i, j]
    vertices[DUMMY] = ti.Vector([0.0, -100.0, 0.0])  # 远离相机和桌面
    colors[DUMMY]   = ti.Vector([0.0, 0.0, 0.0])

window = ti.ui.Window("Taichi Cloth Simulation | [P] pause  [N] step  [R] reset", (1024, 1024), vsync=True)
print("[Controls] P pause | N step | R reset | Auto: drop -> attach -> lift -> pull -> tear")

# canvas = window.get_canvas()
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
# scene = ti.ui.Scene()
scene = window.get_scene()
camera = ti.ui.Camera()

current_t = 0.0
initialize_mass_points()
init_springs()
# make_notch()
paused = False
settled = False
p_prev = False
r_prev = False
n_prev = False
phase = 0          # 0=drop 1=attach 2=lift 3=pull 4=done
auto_run = True
t_phase = 0.0      # 每个阶段内部计时
phase = 0
auto_run = True
clamp_on[None] = 0
frame_id = 0
notch_done = False
TEAR_RATIO_SAFE = 3.0   # 抬起/移动阶段：不希望乱裂
TEAR_RATIO_TEAR = 1.2   # 真撕裂阶段：希望容易裂
last_phase = -999

while window.running:
    # --- hotkeys (inside loop) ---
    # R reset (debounced)
    r_now = window.is_pressed('r')
    if r_now and (not r_prev):
        initialize_mass_points()
        init_springs()
 
        current_t = 0.0
        settled = False
        paused = False

        phase = 0
        auto_run = True
        clamp_on[None] = 0
        enable_tear[None] = 0
        tear_ratio_dyn[None] = TEAR_RATIO_SAFE
        notch_done = False
        last_phase = -999

    r_prev = r_now
    if frame_id % 30 == 0:
        ij0 = clamp_ij[0].to_numpy()
        y_cloth = x[int(ij0[0]), int(ij0[1])].to_numpy()[1]
        y_clamp = clamp_pos[0].to_numpy()[1]
        print("frame", frame_id, "phase", phase, "on", clamp_on[None], "y_clamp", y_clamp, "y_cloth", y_cloth)

    # P pause toggle (debounced)
    p_now = window.is_pressed('p')
    if p_now and (not p_prev):
        paused = not paused
    p_prev = p_now

    # N single-step (debounced): only one frame step when paused
    n_now = window.is_pressed('n')
    step_once = (n_now and (not n_prev))
    n_prev = n_now


    # --- settle detection ---
    compute_avg_speed()
    if (not settled) and current_t > 1.0 and avg_speed[None] < 0.02:
        settled = True

    if (not paused) or step_once:
        # --- auto two-stage script ---
        if phase != last_phase:
            # 进入新阶段时只做一次配置
            if phase in (0, 1, 15, 2, 25):
                enable_tear[None] = 0
                tear_ratio_dyn[None] = TEAR_RATIO_SAFE
            elif phase == 3:
                enable_tear[None] = 1
                tear_ratio_dyn[None] = TEAR_RATIO_TEAR

            last_phase = phase
            
        if auto_run:
            frame_dt = substeps * dt

            # 目标参数
            y_lift = table_y + 0.40      # 抬起高度
            y_pull = table_y + 0.30      # 拉开时高度
            pull_speed = 0.90            # 每秒拉开多少（单位：世界坐标）
            lift_speed = 0.90            # 每秒抬起多少
            pull_limit = 1.20            # 拉到多开停止（左右 clamp 的 x 距离的一半）
            close_speed = 0.256
            if phase == 0:
                # 阶段1：纯掉落，不夹
                clamp_on[None] = 0
                clamp_gap[None] = clamp_gap_open
                if settled or (current_t > 2.0):
                    phase = 1
                    # t_phase = 0.0
                
            elif phase == 1:
                # 阶段2开始：attach（把夹板瞬移到布点上，避免暴力）
                attach_clamps_to_cloth()
                clamp_on[None] = 1
                clamp_gap[None] = clamp_gap_open
                phase = 15
                # phase = 2
                # t_phase = 0.0
            elif phase == 15:
                clamp_gap[None] = max(clamp_gap_close, clamp_gap[None] - close_speed * frame_dt)
                if clamp_gap[None] <= clamp_gap_close + 1e-4:
                    phase = 2    

            elif phase == 2:
                # 抬起：两侧一起抬到 y_lift
                for k in range(num_clamps):
                    p = clamp_pos[k].to_numpy()
                    p[1] = min(y_lift, p[1] + lift_speed * frame_dt)
                    clamp_pos[k] = p
                if clamp_pos[0].to_numpy()[1] >= y_lift - 1e-3:
                    phase = 25
                    
            elif phase == 25:
                for k in range(num_clamps):
                    p = clamp_pos[k].to_numpy()
                    p[1] = max(y_pull, p[1] - 0.60 * frame_dt)
                    clamp_pos[k] = p
                if clamp_pos[0].to_numpy()[1] <= y_pull + 1e-3:
                    if not notch_done:
                        make_notch()
                        notch_done = True
                    phase = 3


            elif phase == 3:
                # 拉开：左右向两侧拉，同时压低到 y_pull
                p0 = clamp_pos[0].to_numpy()
                p1 = clamp_pos[1].to_numpy()
                # 对边
                p0[0] -= pull_speed * frame_dt
                p1[0] += pull_speed * frame_dt

                p0[1] = y_pull
                p1[1] = y_pull

                # 防止飞出桌面范围
                p0[0] = max(-table_bounds, p0[0])
                p1[0] = min( table_bounds, p1[0])

                clamp_pos[0] = p0
                clamp_pos[1] = p1

                # 达到拉开上限就结束展示
                if (p1[0] - p0[0]) * 0.5 > pull_limit:
                    phase = 4

            elif phase == 4:
                paused = True
                auto_run = False

        frame_dt = substeps * dt
        update_clamp_velocity(1.0 / frame_dt)

        # --- simulation stepping ---
        for _ in range(substeps):
            substep()
            current_t += dt

    update_vertices()
    update_mesh_indices_by_tears()
    update_clamp_mesh_vertices()

    target = (0.0, table_y + 0.10, 0.0)  # 盯住布/球的上方一点
    d = 1.8                           # 相机距离（越大越远）
    h = 1                              # 相机高度（决定俯视角）
    camera.position(0, target[1] + h, target[2] + d)
    camera.lookat(*target)
    camera.up(0.0, 1.0, 0.0)
    # camera.track_user_inputs(window, movement_speed=0.02, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))

    # draw table
    scene.mesh(table_vertices, indices=table_indices, color=table_color, two_sided=True)
    scene.mesh(vertices, indices=indices, per_vertex_color=colors, two_sided=True)
    # Draw a smaller ball to avoid visual penetration
    scene.particles(ball_center, radius=ball_radius * 0.95, color=(0.5, 0.42, 0.8))
    scene.mesh(clamp_vertices, indices=clamp_indices, color=clamp_color, two_sided=True)

    canvas.scene(scene)
    window.show()
    frame_id += 1