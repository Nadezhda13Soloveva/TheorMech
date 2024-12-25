import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Константы
R = 0.5  # Радиус цилиндра A
r = 0.1  # Радиус цилиндра B
l = 1.0  # Длина стержня
m1 = 2.0  # Масса стержня
m2 = 5.0  # Масса цилиндра A
m3 = 3.0  # Масса цилиндра B
g = 9.81  # Ускорение свободного падения

# Уравнения движения с использованием метода Крамера
def equations(t, y):
    phi, phi_dot, psi, psi_dot = y

    # Коэффициенты уравнений
    A11 = (m1/3)*l**2 + (m2 + m3)*(R + l)**2 + (m2 + m3/2)*R**2
    A12 = m3*(R - r)*(R + l)*np.cos(phi - psi)
    A21 = (R + l)*np.cos(phi - psi) - R/2
    A22 = (3/2)*(R - r)

    B1 = -((m1/2)*l + (m2 + m3)*(R + l)) * g * np.sin(phi) - m3*(R - r)*(R + l)*phi_dot**2*np.sin(phi - psi)
    B2 = -g*np.sin(psi) + (R + l)*phi_dot**2*np.sin(phi - psi)

    # Метод Крамера для решения системы
    det = A11 * A22 - A12 * A21
    phi_ddot = (B1 * A22 - B2 * A12) / det
    psi_ddot = (B2 * A11 - B1 * A21) / det

    return [phi_dot, phi_ddot, psi_dot, psi_ddot]

# Начальные условия
phi0 = np.pi / 6
psi0 = np.pi / 3
phi_dot0 = 0.0
psi_dot0 = 0.0
y0 = [phi0, phi_dot0, psi0, psi_dot0]

# Временной интервал
t_span = (0, 10)
t_eval = np.linspace(*t_span, 600)

# Решение системы уравнений
sol = solve_ivp(equations, t_span, y0, t_eval=t_eval)

# Вычисление R_Ox и R_Oy
phi_ddot_values = np.gradient(sol.y[1], t_eval)
psi_ddot_values = np.gradient(sol.y[3], t_eval)

R_Ox = -((m1/2)*l + (m2 + m3)*(R + l)) * (phi_ddot_values * np.sin(sol.y[0]) + sol.y[1]**2 * np.cos(sol.y[0])) \
       - m3*(R - r)*(psi_ddot_values * np.sin(sol.y[2]) + sol.y[3]**2 * np.cos(sol.y[2])) \
       - (m1 + m2 + m3)*g

R_Oy = ((m1/2)*l + (m2 + m3)*(R + l)) * (phi_ddot_values * np.cos(sol.y[0]) - sol.y[1]**2 * np.sin(sol.y[0])) \
       + m3*(R - r)*(psi_ddot_values * np.cos(sol.y[2]) - sol.y[3]**2 * np.sin(sol.y[2]))

# Анимация
def update(frame):
    plt.cla()
    phi = sol.y[0, frame]
    psi = sol.y[2, frame]
    
    # Координаты стержня
    rod_x = [0, l * np.sin(phi)]
    rod_y = [0, -l * np.cos(phi)]
    
    # Координаты центра цилиндра A
    center_A_x = (l + R) * np.sin(phi)
    center_A_y = -(l + R) * np.cos(phi)
    
    # Координаты центра цилиндра B
    center_B_x = center_A_x + (R-r) * np.sin(psi)
    center_B_y = center_A_y - (R-r) * np.cos(psi)
    
    ## Обновление угла для линии-индикатора вращения цилиндра B
    # Угол вращения цилиндра B по его окружности
    theta_B = (R - r) * psi / r
    
    # Добавление линии-индикатора вращения для цилиндра B
    indicator_x = center_B_x + r * np.cos(theta_B)
    indicator_y = center_B_y + r * np.sin(theta_B)
    
    # Отрисовка
    plt.plot(rod_x, rod_y, 'k-', linewidth=2)  # Стержень
    
    # Отрисовка цилиндра A
    circle_A = plt.Circle((center_A_x, center_A_y), R, fill=False, color='blue')
    plt.gca().add_patch(circle_A)
    
    # Отрисовка цилиндра B с индикатором вращения
    circle_B = plt.Circle((center_B_x, center_B_y), r, color='red', alpha=0.5)
    plt.gca().add_patch(circle_B)
    plt.plot([center_B_x, indicator_x], [center_B_y, indicator_y], 'k-')  # Индикатор вращения
    
    # Установка пределов графика и соотношения сторон
    plt.xlim(-2, 2)
    plt.ylim(-2.5, 1)
    plt.gca().set_aspect('equal')
    plt.grid(True)

# Создание фигуры
fig = plt.figure(figsize=(8, 8))

# Создание анимации
anim = FuncAnimation(fig, update, frames=len(t_eval), interval=50, repeat=True)

plt.show()

# Построение графиков
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(sol.t, sol.y[0], label='phi(t)')
plt.xlabel('t')
plt.ylabel('phi')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(sol.t, sol.y[2], label='psi(t)')
plt.xlabel('t')
plt.ylabel('psi')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(sol.t, R_Ox, label='R_Ox(t)')
plt.xlabel('t')
plt.ylabel('R_Ox')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(sol.t, R_Oy, label='R_Oy(t)')
plt.xlabel('t')
plt.ylabel('R_Oy')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
