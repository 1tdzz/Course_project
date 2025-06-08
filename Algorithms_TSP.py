from itertools import permutations
from functools import lru_cache
import time
import sys
import random
from typing import List, Optional, Tuple
from math import inf
import pandas as pd


def read_graph_from_file(filename):
    """Читает граф из файла в виде матрицы смежности"""
    try:
        with open(filename, 'r') as file:
            # Читаем число вершин
            n = int(file.readline().strip())
            # Инициализируем матрицу смежности
            matrix = []
            # Читаем n строк с весами рёбер
            for _ in range(n):
                row = list(map(float, file.readline().strip().split()))
                if len(row) != n:
                    raise ValueError("Неверный формат строки в файле")
                matrix.append(row)
            if len(matrix) != n:
                raise ValueError("Количество строк не соответствует числу вершин")
            return matrix
    except FileNotFoundError:
        print(f"Ошибка: Файл {filename} не найден")
        sys.exit(1)
    except ValueError as e:
        print(f"Ошибка: {e}")
        sys.exit(1)

def calculate_distance(route, distance_matrix):
    """Вычисляет стоимость маршрута, возвращает None, если маршрут недопустим"""
    total = 0
    n = len(route)
    for i in range(n - 1):
        if distance_matrix[route[i]][route[i + 1]] == float('inf'):
            return None
        total += distance_matrix[route[i]][route[i + 1]]
    # Проверяем возврат в начальную вершину
    if distance_matrix[route[-1]][route[0]] == float('inf'):
        return None
    total += distance_matrix[route[-1]][route[0]]
    return total


def brute_force_tsp(distance_matrix):
    """Алгоритм полного перебора для ЗК"""
    n = len(distance_matrix)
    if n <= 1:
        return [0], 0

    cities = list(range(n))
    min_distance = float('inf')
    best_route = None

    for perm in permutations(cities):
        current_distance = calculate_distance(perm, distance_matrix)
        if current_distance is not None and current_distance < min_distance:
            min_distance = current_distance
            best_route = perm

    if best_route is None:
        return None, None
    return list(best_route), min_distance


def tsp_dp(distance_matrix):
    """Алгоритм динамического программирования (Held-Karp) для ЗК"""
    n = len(distance_matrix)

    if n == 1:
        return [0], 0

    all_visited = (1 << n) - 1  # Все города посещены

    # Словарь для хранения предыдущих городов для восстановления пути
    parent = {}

    @lru_cache(None)
    def visit(city, visited):
        if visited == all_visited:
            return distance_matrix[city][0] if distance_matrix[city][0] != float('inf') else float('inf')

        min_cost = float('inf')
        best_next_city = None
        for next_city in range(n):
            if not (visited & (1 << next_city)):  # Если город ещё не посещён
                new_cost = distance_matrix[city][next_city]
                if new_cost != float('inf'):
                    new_cost += visit(next_city, visited | (1 << next_city))
                    if new_cost < min_cost:
                        min_cost = new_cost
                        best_next_city = next_city
        # Сохраняем лучший следующий город для текущего состояния
        if best_next_city is not None:
            parent[(city, visited)] = best_next_city
        return min_cost

    # Вычисляем минимальную стоимость
    min_cost = visit(0, 1 << 0)

    if min_cost == float('inf'):
        return None, None

    # Восстанавливаем маршрут
    path = [0]
    visited = 1 << 0
    current_city = 0
    while visited != all_visited:
        next_city = parent.get((current_city, visited))
        if next_city is None:
            return None, None
        path.append(next_city)
        visited |= 1 << next_city
        current_city = next_city
    path.append(0)  # Возвращаемся в начальный город

    return path, min_cost


def nearest_neighbor_tsp(distance_matrix):
    """Алгоритм ближайшего соседа для ЗК"""
    n = len(distance_matrix)
    if n <= 1:
        return [0], 0

    for i in range(n):
        distance_matrix[i][i] = 0

    visited = [False] * n
    route = [0]  # Начинаем с первого города
    visited[0] = True

    for _ in range(n - 1):
        last_city = route[-1]
        # Ищем непосещённые города с конечными весами
        candidates = [(i, distance_matrix[last_city][i]) for i in range(n) if not visited[i] and distance_matrix[last_city][i] != float('inf')]
        if not candidates:  # Если нет допустимых соседей
            return None, None
        nearest_city, _ = min(candidates, key=lambda x: x[1])
        route.append(nearest_city)
        visited[nearest_city] = True

    # Проверка на возможность вернуться в начальный город
    if distance_matrix[route[-1]][0] == float('inf'):
        return None, None

    route.append(0)  # Возвращаемся в начальный город
    total_distance = calculate_distance(route, distance_matrix)
    if total_distance is None:
        return None, None
    return route, total_distance


class AntPath:
    """Класс для хранения пути муравья и его длины"""
    def __init__(self):
        self.vertices: List[int] = []
        self.distance: float = 0.0

class Ant:
    """Класс, представляющий муравья в муравьином алгоритме"""
    def __init__(self, start_vertex: int):
        self.path = AntPath()
        self.visited: List[int] = []
        self.start_location = start_vertex
        self.current_location = start_vertex
        self.can_continue = True

    def get_random_choice(self) -> float:
        """Генерирует случайное число от 0 до 1 для выбора вершины"""
        return random.random()

    def get_neighbor_vertices(self, graph: List[List[float]]) -> List[int]:
        """Возвращает список непосещённых соседних вершин"""
        n = len(graph)
        neighbors = []
        for to in range(n):
            edge_exists = graph[self.current_location][to] != float('inf')
            vertex_unvisited = to not in self.visited
            if edge_exists and vertex_unvisited:
                neighbors.append(to)
        return neighbors

    def make_choice(self, graph: List[List[float]], pheromone: List[List[float]], alpha: float, beta: float):
        """Выбирает следующую вершину на основе феромонов и весов"""
        if not self.path.vertices:
            self.path.vertices.append(self.current_location)
            self.visited.append(self.current_location)

        neighbor_vertices = self.get_neighbor_vertices(graph)

        if not neighbor_vertices:
            self.can_continue = False
            if graph[self.current_location][self.start_location] != float('inf'):
                self.path.vertices.append(self.start_location)
                self.path.distance += graph[self.current_location][self.start_location]
            return

        choosing_probability = [0.0] * len(neighbor_vertices)
        wish = []
        probability = []
        summary_wish = 0.0

        # Вычисляем желания для перехода
        for v in neighbor_vertices:
            t = pheromone[self.current_location][v]
            w = graph[self.current_location][v]
            n = 1.0 / w if w != 0 else float('inf')
            wish_value = (t ** alpha) * (n ** beta)
            wish.append(wish_value)
            summary_wish += wish_value

        # Вычисляем вероятности
        for i in range(len(neighbor_vertices)):
            prob = wish[i] / summary_wish if summary_wish != 0 else 0.0
            probability.append(prob)
            choosing_probability[i] = prob if i == 0 else choosing_probability[i - 1] + prob

        # Выбираем следующую вершину
        choose = self.get_random_choice()
        next_vertex = neighbor_vertices[0]
        for i, prob in enumerate(choosing_probability):
            if choose <= prob:
                next_vertex = neighbor_vertices[i]
                break

        self.path.vertices.append(next_vertex)
        self.path.distance += graph[self.current_location][next_vertex]
        self.visited.append(next_vertex)
        self.current_location = next_vertex

class AntColonyOptimization:
    """Класс для реализации муравьиного алгоритма"""
    def __init__(self, graph: List[List[float]]):
        self.graph = graph
        self.k_alpha = 1.0
        self.k_beta = 2.0
        self.k_pheromone0 = 1.0
        self.k_evaporation = 0.2

        valid_edges = [
            w for row in graph for w in row
            if w != float('inf') and w > 0
        ]
        if not valid_edges:
            self.is_valid_graph = False
            return
        else:
            self.is_valid_graph = True

        # Вычисляем kQ как 0.015 * суммарный вес графа
        graph_weight = sum(w for row in graph for w in row if w != float('inf') and w != 0) / 2
        self.k_q = 0.015 * graph_weight if graph_weight != 0 else 1.0

        n = len(graph)
        # Инициализируем матрицу феромонов
        self.pheromone = [[self.k_pheromone0 if i != j else 0.0 for j in range(n)] for i in range(n)]
        self.ants: List[Ant] = []

    def create_ants(self):
        """Создаёт муравьёв, по одному на каждую вершину"""
        n = len(self.graph)
        self.ants = [Ant(i) for i in range(n)]

    def update_global_pheromone(self, local_pheromone_update: List[List[float]]):
        """Обновляет глобальную матрицу феромонов"""
        n = len(self.graph)
        for from_v in range(n):
            for to_v in range(n):
                self.pheromone[from_v][to_v] = (
                    (1 - self.k_evaporation) * self.pheromone[from_v][to_v] + local_pheromone_update[from_v][to_v]
                )
                if self.pheromone[from_v][to_v] < 0.01 and from_v != to_v:
                    self.pheromone[from_v][to_v] = 0.01

    def solve_tsp(self) -> Tuple[Optional[List[int]], Optional[float]]:
        """Решает ЗК с помощью муравьиного алгоритма"""
        if not self.is_valid_graph:
            return None, None
        n = len(self.graph)
        if n == 0:
            return None, None

        max_iterations_without_improvement = 100
        counter = 0
        best_path = AntPath()
        best_path.distance = float('inf')

        while counter < max_iterations_without_improvement:
            local_pheromone_update = [[0.0] * n for _ in range(n)]
            self.create_ants()

            for ant in self.ants:
                while ant.can_continue:
                    ant.make_choice(self.graph, self.pheromone, self.k_alpha, self.k_beta)

                ant_path = ant.path
                if len(ant_path.vertices) == n + 1:  # Полный гамильтонов цикл
                    if ant_path.distance < best_path.distance:
                        best_path = ant_path
                        counter = 0

                    for v in range(len(ant_path.vertices) - 1):
                        local_pheromone_update[ant_path.vertices[v]][ant_path.vertices[v + 1]] += (
                            self.k_q / ant_path.distance
                        )

            self.update_global_pheromone(local_pheromone_update)
            counter += 1

        if best_path.distance == float('inf'):
            return None, None
        return best_path.vertices, best_path.distance


def run_single(graph):
    """Выполняет один прогон всех алгоритмов и возвращает результаты"""
    results = {}

    # Полный перебор
    print("=== Полный перебор ===")
    start_time = time.time()
    best_path_bf, min_cost_bf = brute_force_tsp(graph)
    end_time = time.time()
    bf_time = end_time - start_time

    if best_path_bf is None or min_cost_bf is None:
        print("Решение не найдено: граф не содержит гамильтонова цикла")
        results['brute_force'] = {'time': bf_time, 'cost': None}
    else:
        print(f"Минимальная стоимость маршрута: {min_cost_bf}")
        print(f"Оптимальный маршрут: {' -> '.join(map(str, best_path_bf))}")
        results['brute_force'] = {'time': bf_time, 'cost': min_cost_bf}
    print(f"Время выполнения: {bf_time:.6f} секунд")

    # Динамическое программирование
    print("\n=== Динамическое программирование (Held-Karp) ===")
    start_time = time.time()
    best_path_dp, min_cost_dp = tsp_dp(graph)
    end_time = time.time()
    dp_time = end_time - start_time

    if best_path_dp is None or min_cost_dp is None:
        print("Решение не найдено: граф не содержит гамильтонова цикла")
        results['dynamic_programming'] = {'time': dp_time, 'cost': None}
    else:
        print(f"Минимальная стоимость маршрута: {min_cost_dp}")
        print(f"Оптимальный маршрут: {' -> '.join(map(str, best_path_dp))}")
        results['dynamic_programming'] = {'time': dp_time, 'cost': min_cost_dp}
    print(f"Время выполнения: {dp_time:.6f} секунд")

    # Ближайший сосед
    print("\n=== Ближайший сосед ===")
    start_time = time.time()
    best_path_nn, min_cost_nn = nearest_neighbor_tsp(graph)
    end_time = time.time()
    nn_time = end_time - start_time

    if best_path_nn is None or min_cost_nn is None:
        print("Решение не найдено: граф не содержит гамильтонова цикла")
        results['nearest_neighbor'] = {'time': nn_time, 'cost': None}
    else:
        print(f"Приближённая стоимость маршрута: {min_cost_nn}")
        print(f"Жадный маршрут: {' -> '.join(map(str, best_path_nn))}")
        results['nearest_neighbor'] = {'time': nn_time, 'cost': min_cost_nn}
    print(f"Время выполнения: {nn_time:.6f} секунд")

    # Муравьиный алгоритм
    print("\n=== Муравьиный алгоритм ===")
    start_time = time.time()
    aco = AntColonyOptimization(graph)
    best_path_aco, min_cost_aco = aco.solve_tsp()
    end_time = time.time()
    aco_time = end_time - start_time

    if best_path_aco is None or min_cost_aco is None:
        print("Решение не найдено: граф не содержит гамильтонова цикла")
        results['ant_colony'] = {'time': aco_time, 'cost': None}
    else:
        print(f"Приближённая стоимость маршрута: {min_cost_aco}")
        print(f"Маршрут: {' -> '.join(map(str, best_path_aco))}")
        results['ant_colony'] = {'time': aco_time, 'cost': min_cost_aco}
    print(f"Время выполнения: {aco_time:.6f} секунд")

    return results


def save_to_excel(all_results, filename="tsp_results.xlsx"):
    """Сохраняет результаты в Excel файл"""
    data = []

    for run_num, results in enumerate(all_results, 1):
        row = [run_num]

        # Добавляем данные для каждого алгоритма
        algorithms = ['brute_force', 'dynamic_programming', 'nearest_neighbor', 'ant_colony']
        for algo in algorithms:
            if algo in results:
                row.extend([results[algo]['time'], results[algo]['cost']])
            else:
                row.extend([None, None])

        data.append(row)

    # Создаем DataFrame
    columns = ['Номер запуска']
    algorithm_names = ['Полный перебор', 'Хелд-Карп', 'Ближ. сосед', 'Муравьиный']
    for name in algorithm_names:
        columns.extend([f'{name} - Время (сек)', f'{name} - Стоимость'])

    df = pd.DataFrame(data, columns=columns)

    # Добавляем строку со средними значениями
    if len(all_results) > 1:  # Только если больше одного прогона
        avg_row = ['СРЕДНЕЕ']
        algorithms = ['brute_force', 'dynamic_programming', 'nearest_neighbor', 'ant_colony']

        for algo in algorithms:
            # Вычисляем среднее время
            times = [result[algo]['time'] for result in all_results
                     if algo in result and result[algo]['time'] is not None]
            avg_time = sum(times) / len(times) if times else None

            # Вычисляем среднюю стоимость
            costs = [result[algo]['cost'] for result in all_results
                     if algo in result and result[algo]['cost'] is not None]
            avg_cost = sum(costs) / len(costs) if costs else None

            avg_row.extend([avg_time, avg_cost])

        # Добавляем пустую строку для разделения
        empty_row = [''] * len(columns)
        df.loc[len(df)] = empty_row

        # Добавляем строку со средними значениями
        df.loc[len(df)] = avg_row

    # Сохраняем в Excel
    df.to_excel(filename, index=False)
    print(f"\nРезультаты сохранены в файл: {filename}")


def main():
    filename = "graph_10_50pct_random_directed.txt"
    graph = read_graph_from_file(filename)

    print("Программа для тестирования алгоритмов решения задачи коммивояжёра")
    print("=" * 60)

    num_runs = 10
    output_filename =  "tsp_results.xlsx"

    all_results = []

    print(f"\nЗапуск {num_runs} прогонов...")
    print("=" * 60)

    for run in range(1, num_runs + 1):
        print(f"\n{'=' * 20} ПРОГОН {run} {'=' * 20}")
        try:
            results = run_single(graph)
            all_results.append(results)
            print(f"Прогон {run} завершён успешно")
        except Exception as e:
            print(f"Ошибка в прогоне {run}: {e}")
            continue

        print("-" * 60)

    # Сохраняем результаты в Excel
    if all_results:
        save_to_excel(all_results, output_filename)
        print(f"\nВсего выполнено прогонов: {len(all_results)}")
        print("Средние значения добавлены в таблицу Excel")
    else:
        print("Нет результатов для сохранения!")


if __name__ == "__main__":
    main()