import math

# ===== your params =====
A = 0.5
omega = 0.3
v = 0.36
k = omega / v          # 5/6
z = 0.25
yaw = 0.0

# ===== choose a SAFE x-range inside your map =====
# inner wall approx: [-2.35, 2.35], pick a margin
x_min = -2.0
x_max =  0.0           # centered around x=-1
dx = 0.05              # smaller -> smoother

def y_of_x(x):
    return A * math.sin(k * (x + 1.0))

t = 0.0
dt = dx / v            # keep approx constant x-speed magnitude

print("<trajectory id='0' type='swing'>")

# forward
x = x_min
while x <= x_max + 1e-9:
    y = y_of_x(x)
    print(f"  <waypoint><time>{t:.3f}</time><pose>{x:.3f} {y:.3f} {z:.2f} 0 0 {yaw:.3f}</pose></waypoint>")
    x += dx
    t += dt

# backward (skip the end point once to avoid duplication)
x = x_max - dx
while x >= x_min - 1e-9:
    y = y_of_x(x)
    print(f"  <waypoint><time>{t:.3f}</time><pose>{x:.3f} {y:.3f} {z:.2f} 0 0 {yaw:.3f}</pose></waypoint>")
    x -= dx
    t += dt

print("</trajectory>")