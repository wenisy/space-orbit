import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# 引力常数和三颗星的质量
G = 1.0
m1 = 1.0
m2 = 1.0
m3 = 1.0

# 定义三体系统的微分方程
def three_body_ode(t, state):
    # 提取位置和速度
    r1 = state[0:2]
    r2 = state[2:4]
    r3 = state[4:6]
    v1 = state[6:8]
    v2 = state[8:10]
    v3 = state[10:12]
    
    # 计算相互之间的距离
    d12 = np.linalg.norm(r1 - r2)
    d13 = np.linalg.norm(r1 - r3)
    d23 = np.linalg.norm(r2 - r3)
    
    # 根据牛顿万有引力定律计算加速度
    a1 = G * m2 * (r2 - r1) / d12**3 + G * m3 * (r3 - r1) / d13**3
    a2 = G * m1 * (r1 - r2) / d12**3 + G * m3 * (r3 - r2) / d23**3
    a3 = G * m1 * (r1 - r3) / d13**3 + G * m2 * (r2 - r3) / d23**3
    
    # 返回速度和加速度组合成的导数向量
    return np.concatenate([v1, v2, v3, a1, a2, a3])

# 设置初始条件（可通过微调扰动值来展示混沌敏感性）
r1_0 = np.array([-1.0, 0.0])
r2_0 = np.array([1.0, 0.0])
r3_0 = np.array([0.0, 1.0])
v1_0 = np.array([0.0, 0.3])
v2_0 = np.array([0.0, -0.3])
v3_0 = np.array([-0.3, 0.0])
state0 = np.concatenate([r1_0, r2_0, r3_0, v1_0, v2_0, v3_0])

# 模拟时间设置
t_span = (0, 30)
t_eval = np.linspace(t_span[0], t_span[1], 2000)

# 求解微分方程（数值积分）
sol = solve_ivp(three_body_ode, t_span, state0, t_eval=t_eval, rtol=1e-9, atol=1e-9)
r1 = sol.y[0:2].T
r2 = sol.y[2:4].T
r3 = sol.y[4:6].T

# 设置绘图和动画
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("三体问题的混沌运动示例")

# 分别为三颗星设置轨迹绘图（用不同颜色区分）
line1, = ax.plot([], [], 'ro-', label="星体 1")
line2, = ax.plot([], [], 'bo-', label="星体 2")
line3, = ax.plot([], [], 'go-', label="星体 3")
ax.legend()

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    return line1, line2, line3

def update(frame):
    # 更新各星体的运动轨迹（显示从起始到当前时刻的路径）
    line1.set_data(r1[:frame, 0], r1[:frame, 1])
    line2.set_data(r2[:frame, 0], r2[:frame, 1])
    line3.set_data(r3[:frame, 0], r3[:frame, 1])
    return line1, line2, line3

ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True, interval=20)
plt.show()
