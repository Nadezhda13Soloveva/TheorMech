import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Константы
R = 1.0  # Радиус цилиндра A
r = 0.4  # Радиус цилиндра B
l = 2.0  # Длина стержня

# Массив времени
t = np.linspace(0, 10, 200)

def update(frame):
    plt.cla()  # Очистить текущую ось
    
    # Определение уравнений движения
    phi = 0.5 * np.sin(t[frame])  # Угол стержня
    
    # Более агрессивное колебание для цилиндра B
    theta = 1.5 * np.sin(t[frame])  # Увеличенная амплитуда (1.5)
    psi = -(R/r) * theta  # Угол вращения цилиндра B
    
    # Координаты стержня
    rod_x = [0, l * np.sin(phi)]
    rod_y = [0, -l * np.cos(phi)]
    
    # Координаты центра цилиндра A
    center_A_x = (l + R) * np.sin(phi)
    center_A_y = -(l + R) * np.cos(phi)
    
    # Координаты центра цилиндра B - колеблется внизу
    center_B_x = center_A_x + (R-r) * np.sin(theta)
    center_B_y = center_A_y - (R-r) * np.cos(theta)
    
    # Добавление линии-индикатора вращения для цилиндра B
    indicator_x = center_B_x + r * np.cos(psi)
    indicator_y = center_B_y + r * np.sin(psi)
    
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
    plt.xlim(-3, 3)
    plt.ylim(-5, 2)
    plt.gca().set_aspect('equal')
    plt.grid(True)

# Создание фигуры
fig = plt.figure(figsize=(8, 8))

# Создание анимации
anim = FuncAnimation(fig, update, frames=len(t), interval=50, repeat=True)

plt.show() 