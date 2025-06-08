import random
import os


def generate_adjacency_matrix(n, weight_range=(1, 100), density=1.0, uniform=False, directed=False, filename=None):
    """
    Генерирует матрицу смежности графа с заданными параметрами.

    :param n: количество вершин
    :param weight_range: диапазон весов рёбер (min, max)
    :param density: плотность графа (0.0–1.0)
    :param uniform: True — одинаковые веса, False — случайные
    :param directed: True — ориентированный граф, False — неориентированный
    :param filename: имя файла (автоматически, если не указано)
    :return: матрица смежности
    """
    min_weight, max_weight = weight_range
    matrix = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if directed or i < j:
                if random.random() <= density:
                    weight = min_weight if uniform else random.randint(min_weight, max_weight)
                    matrix[i][j] = weight
                    if not directed:
                        matrix[j][i] = weight

    if filename is None:
        uniform_str = "uniform" if uniform else "random"
        density_pct = int(density * 100)
        direction_str = "directed" if directed else "undirected"
        filename = f"graph_{n}_{density_pct}pct_{uniform_str}_{direction_str}.txt"

    with open(filename, 'w') as f:
        f.write(f"{n}\n")
        for row in matrix:
            formatted = ['inf' if x == 0 else str(x) for x in row]
            f.write(' '.join(formatted) + '\n')

    print(f"Матрица сохранена в файл: {filename}")
    return matrix


if __name__ == "__main__":
    N = int(input("Введите количество вершин (N): "))
    min_w = int(input("Минимальный вес ребра: "))
    max_w = int(input("Максимальный вес ребра: "))
    density = float(input("Полнота графа (например, 1.0 для полного, 0.6 для 60%): "))
    uniform_input = input("Все веса одинаковые? (y/n): ").strip().lower()
    uniform = uniform_input == 'y'
    directed_input = input("Граф ориентированный? (y/n): ").strip().lower()
    directed = directed_input == 'y'

    generate_adjacency_matrix(
        N,
        weight_range=(min_w, max_w),
        density=density,
        uniform=uniform,
        directed=directed
    )
