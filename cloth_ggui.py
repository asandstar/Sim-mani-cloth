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

ball_radius = 0.3 # 球中心是一维field，唯一元素是三维浮点数向量
ball_center = ti.Vector.field(3, dtype=float, shape=(1, ))
ball_center[0] = [0, 0, 0] # 放在原点
# x(position), v(velocities)是nxn的field，有3d漂浮点的向量
x = ti.Vector.field(3, dtype=float, shape=(n, n))
v = ti.Vector.field(3, dtype=float, shape=(n, n))

num_triangles = (n - 1) * (n - 1) * 2
# cloth作为三角网格，由两个field表示，vertices是顶点
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=float, shape=n * n)
colors = ti.Vector.field(3, dtype=float, shape=n * n)

bending_springs = False
# kernel装饰器可以并行计算
@ti.kernel
def initialize_mass_points():
    # 随机的offset
    random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.1
    # 轻微移动cloth，随机偏离
    for i, j in x:
        x[i, j] = [
            i * quad_size - 0.5 + random_offset[0], 0.6,
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

initialize_mesh_indices()
# cloth作为一个有质量的弹簧网格进行建模，假设质点的相对index是(0,0)至少被12个周围点影响
# spring offsets，用来存储受影响点的相对index的列表
spring_offsets = []
if bending_springs:
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i, j) != (0, 0):
                spring_offsets.append(ti.Vector([i, j]))

else:
    for i in range(-2, 3):
        for j in range(-2, 3):
            if (i, j) != (0, 0) and abs(i) + abs(j) <= 2:
                spring_offsets.append(ti.Vector([i, j]))
# 弹簧质点系统受到的重力、内力、阻尼、碰撞等效果的积累
@ti.kernel
def substep():
    for i in ti.grouped(x):
        v[i] += gravity * dt # 对field里所有元素施加重力循环
        # for循环将x作为一个一维数组自动遍历其中所有元素，不论形状

    for i in ti.grouped(x):
        # 施加在特定质点的初始的力
        force = ti.Vector([0.0, 0.0, 0.0])
        # 穿越周围的质点
        for spring_offset in ti.static(spring_offsets):
            # j是受影响的点的绝对index，是二维向量
            j = i + spring_offset
            # 如果受影响的点在 nxn的网格中，算出内力并且把它施加到当前的质点上
            if 0 <= j[0] < n and 0 <= j[1] < n:
                # 两点的相对位移，内力与之有关
                x_ij = x[i] - x[j]
                v_ij = v[i] - v[j]
                # d是正则化的向量
                d = x_ij.normalized()
                current_dist = x_ij.norm()
                original_dist = quad_size * float(i - j).norm()
                # Spring force
                # 弹簧的内力
                force += -spring_Y * d * (current_dist / original_dist - 1)
                # Dashpot damping
                # 由于两点相对运动而产生的阻尼力
                force += -v_ij.dot(d) * d * dashpot_damping * quad_size
        # 由于有内力作用，当前速度会因此增加
        v[i] += force * dt

    for i in ti.grouped(x):
        # 穿越场v里的元素
        v[i] *= ti.exp(-drag_damping * dt)
        offset_to_center = x[i] - ball_center[0]
        if offset_to_center.norm() <= ball_radius:
            # Velocity projection
            normal = offset_to_center.normalized()
            # 算出速度的积累
            v[i] -= min(v[i].dot(normal), 0) * normal
            # 算出每个质点的最终位置
        x[i] += dt * v[i]

# 每个帧都要调用update_vertices，顶点在模拟中持续更新
@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = x[i, j]

window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 1024),
                      vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

current_t = 0.0
initialize_mass_points()

while window.running:
    if current_t > 1.5:
        # Reset
        initialize_mass_points()
        current_t = 0

    for i in range(substeps):
        substep()
        current_t += dt
    update_vertices()

    camera.position(0.0, 0.0, 3)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(vertices,
               indices=indices,
               per_vertex_color=colors,
               two_sided=True)

    # Draw a smaller ball to avoid visual penetration
    scene.particles(ball_center, radius=ball_radius * 0.95, color=(0.5, 0.42, 0.8))
    canvas.scene(scene)
    window.show()