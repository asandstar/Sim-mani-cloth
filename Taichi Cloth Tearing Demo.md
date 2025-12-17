# Taichi Cloth Tearing Demo (Mass–Spring + Tool Grasp + Mid-Seam Tear)

本项目演示了一个可实时运行和渲染的布料操控与撕裂场景：一块布料先落在桌面（并与球体发生碰撞），随后两侧夹子夹住布料、抬起并向两侧拉开，布料沿中线发生可控的撕裂。

与“纯自然撕裂”不同，本项目的撕裂采用了**工程化的门控策略（mid seam gate）**：只有跨越中线的结构边允许断裂，以保证演示稳定性与可复现性。


## 开始

### 1) 导入并初始化 Taichi

```python
import taichi as ti
ti.init(arch=ti.cuda)  # 推荐 CUDA；GGUI 3D 渲染需要 CUDA/Vulkan 等后端
```

说明：

* 本项目使用 Taichi GGUI 进行 3D 渲染（桌面/布料三角网格 + 球体粒子 + 夹子网格）。
* `ti.cuda` 可获得更高的帧率；如果本机不支持可切换为 CPU，但渲染/性能会受影响。

## 建模

本节内容包括：

* 布料：弹簧质点系统（mass–spring）
* 场景：桌面 + 球体
* 工具：双夹子（clamps）与“局部贴附”抓取
* 撕裂：弹簧存活状态 + 渲染拓扑更新

## 模型简化与核心假设

### 布料：mass–spring 网格

布料被建模为一个 `n × n` 质点网格：

* `x[i, j]`：质点位置（3D 向量）
* `v[i, j]`：质点速度（3D 向量）

相互作用只考虑局部邻域弹簧连接（8 邻域 / 12 影响点的思想），并用：

* 弹簧力（Hooke-like）
* dashpot 阻尼（沿弹簧方向的相对速度阻尼）
* 空气阻尼（整体速度指数衰减）

来保证数值稳定性和可控性。

### 场景：桌面 + 球体

* 桌面使用一个平面（`y = table_y`）做投影式碰撞，并在切向方向加入摩擦衰减。
* 球体用中心 `ball_center` 与半径 `ball_radius` 表示，对穿透质点做投影回球面，并移除“向内”的法向速度分量。

### 工具：双夹子抓取（Tool Constraints）

夹子不是通过刚体接触力求解来抓布，而是采用一种**可控的“局部贴附约束”**：

* 选定两个布点 `clamp_ij[0], clamp_ij[1]` 作为夹持中心；
* 抓取开启时，对每个夹子中心周围一个 `(2R+1)×(2R+1)` 的 patch：

  * 直接把该 patch 的 `x` 设为 `clamp_pos + clamp_dpos`
  * 把 `v` 设为 `clamp_vel`（同步工具速度）

该方法的优点是稳定、易控制、可复现；缺点是牺牲了部分真实接触力学。

### 撕裂：可控的中线门控策略（mid seam gate）

为了避免“全局应力触发导致随机开裂”，撕裂采用了门控：

* 只有跨越中线 `i = mid-1 <-> mid` 的结构边（`(±1,0)`）允许断裂；
* 断裂判据为：`stretch > tear_ratio_dyn` 且 `enable_tear == 1`

撕裂阶段之外通过设置较大的 `tear_ratio_dyn` 和关闭 `enable_tear` 来避免误裂。

## 数据结构

### 1) 布料 Field：位置与速度

```python
n = 128
x = ti.Vector.field(3, dtype=float, shape=(n, n))
v = ti.Vector.field(3, dtype=float, shape=(n, n))

quad_size = 1.0 / n
```

初始化布料：

```python
@ti.kernel
def initialize_mass_points():
    random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.1
    for i, j in x:
        x[i, j] = [i * quad_size - 0.5 + random_offset[0], 0.9,
                   j * quad_size - 0.5 + random_offset[1]]
        v[i, j] = [0, 0, 0]
```

### 2) 球体 Field：中心 + 半径

```python
ball_radius = 0.1
ball_center = ti.Vector.field(3, dtype=float, shape=(1,))
ball_center[0] = [0, table_y + ball_radius, 0]
```

### 3) 弹簧拓扑与存活状态

本项目的弹簧连接通过 `spring_offsets` 定义（8 邻域），并在 Taichi 侧存储为 field，方便 kernel 内动态索引：

* `spring_off[k]`：第 k 条 offset
* `spring_L0[k]`：该 offset 的原长
* `spring_opp[k]`：相反方向 offset 的索引（用于双向断裂）
* `spring_alive[i, j, k]`：从点 `(i,j)` 沿 offset k 的弹簧是否存在

### 4) 夹子（clamps）与抓取 patch

* `clamp_pos[k]` / `clamp_prev[k]` / `clamp_vel[k]`：夹子状态
* `clamp_ij[k]`：夹子夹持的布点索引
* `clamp_dpos[k, :, :]`：夹持 patch 内各点相对中心的偏移

抓取开启时：patch 内点被强制附着到工具坐标系。

## 模拟

模拟通过 `substep()` kernel 完成一次时间步的更新，其结构可概括为：

1. 重力更新速度
2. 计算弹簧内力 + dashpot 阻尼，并累积到速度
3. 空气阻尼
4. 积分更新位置
5. 碰撞处理（球体 + 桌面 + 边界）
6. 若夹持开启，对 patch 执行贴附约束（覆盖 x, v）

### 1) 重力

```python
for i in ti.grouped(x):
    v[i] += gravity * dt
```

### 2) 弹簧内力 + dashpot 阻尼

对每个点遍历所有 offset 弹簧：

* `stretch = dist / L0`
* 弹簧力：沿方向 `d`，与 `(stretch - 1)` 成正比
* dashpot：沿方向 `d`，与相对速度投影 `v_ij·d` 成正比

```python
force += -spring_Y * d * (stretch - 1.0)
force += -v_ij.dot(d) * d * dashpot_damping * quad_size
```

### 3) 撕裂判据：mid seam gate + stretch threshold

撕裂不对全局弹簧开放，而只允许中线跨越边：

* `mid = n // 2`
* `is_struct_x`：结构边 `(±1,0)`
* `is_mid_seam`：仅当边跨越 `mid-1 <-> mid`

触发后断双向弹簧，并额外断跨缝对角弹簧（减少“视觉粘连”）。

### 4) 空气阻尼 + 积分

```python
v[i] *= ti.exp(-drag_damping * dt)
x[i] += dt * v[i]
```

### 5) 碰撞：球体与桌面（投影式）

球体：

* 若进入球内部，投影回球面
* 去掉“向内”的法向速度

桌面：

* 若低于桌面，投影到桌面上方 `eps`
* 法向速度截断
* 切向速度做摩擦衰减

### 6) 夹子抓取：局部贴附约束

当 `clamp_on == 1`：

* patch 内点位置与速度直接被覆盖为工具刚体坐标系下的值
* 这是一个稳定且可控的“抓取”近似

## 阶段脚本：Drop–Settle–Manipulate

为了让演示行为可复现，主循环中使用 `phase` 状态机驱动夹子动作：

* Phase 0：Drop（自然落下）
* settle 检测：平均速度低于阈值认为稳定
* Phase 1/15：Attach + Close（对齐布点并夹紧）
* Phase 2：Lift（抬起到目标高度）
* Phase 25：Lower + Notch（下降到撕裂高度并切预裂口）
* Phase 3：Pull + Tear（向两侧拉开，开启撕裂）
* Phase 4：Stop（暂停展示）

同时，撕裂阈值在阶段切换时统一配置：

* 非撕裂阶段：`enable_tear = 0`，`tear_ratio_dyn = TEAR_RATIO_SAFE`
* 撕裂阶段：`enable_tear = 1`，`tear_ratio_dyn = TEAR_RATIO_TEAR`

该逻辑也用于解决 reset 后阈值残留导致的异常行为。

## 渲染

本项目渲染三类对象：

1. 桌面：三角网格（静态 indices）
2. 布料：三角网格（vertices 每帧更新；indices 会根据断裂动态失效）
3. 球体：粒子（GGUI particles）
4. 夹子：两块夹板的三角网格（vertices 每帧更新）

### 布料网格：vertices 更新

```python
@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = x[i, j]
```

### 撕裂可视化：动态更新 indices

仅断弹簧并不足以让网格“视觉断开”。为此，`update_mesh_indices_by_tears()` 会检查每个 quad 组成的两个三角形：

* 若三角形依赖的边（含对角边）已断，则把该三角形 indices 指向 `DUMMY`
* `DUMMY` 顶点被放在远离相机的位置，从而实现“该三角面片不再渲染”的效果

这一步是“看起来真的裂开”的关键。

## 已实现里程碑（Milestones）

* **M0**: Taichi cloth demo baseline (mass–spring + GGUI)
* **M1**: Table contact (projection + friction)
* **M2**: Projection-based collision (sphere + stability)
* **M3**: Two clamps (dual tool primitives + visualization)
* **M4**: Attachment grasp (patch-based attachment, stable pickup)
* **M5**: Drop–Settle–Manipulate + gated tearing (mid-seam tear + notch + phase control)

## 已知限制与折衷

* 撕裂不是材料损伤模型，而是“中线门控 + 阈值断弹簧 + 网格拓扑失效”的工程化实现；
* 抓取不是接触力求解，而是 patch 贴附约束；
* 这些折衷换来的是：稳定、可复现、可控、适合课堂演示与实验迭代。
