import unittest
import tempfile
import os
from Algorithms_TSP import nearest_neighbor_tsp, read_graph_from_file, calculate_distance
from math import inf

class MyTestCase(unittest.TestCase):
    def setUp(self):
        """Создание временных файлов для тестов"""
        self.temp_files = []

    def tearDown(self):
        """Удаление временных файлов после тестов"""
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def create_temp_file(self, content):
        """Создает временный файл с заданным содержимым"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        temp_file.write(content)
        temp_file.close()
        self.temp_files.append(temp_file.name)
        return temp_file.name

    def test_asymmetric_graph(self):
        """Тест асимметричного графа"""
        distance_matrix = [
            [0, 1, 5],
            [2, 0, 3],
            [4, 6, 0]
        ]
        route, distance = nearest_neighbor_tsp(distance_matrix)
        self.assertIsNotNone(route)
        self.assertIsNotNone(distance)
        self.assertEqual(len(route), 4)
        self.assertEqual(route[0], 0)
        self.assertEqual(route[-1], 0)
        # Ближайший сосед: 0->1->2->0 = 1+3+4 = 8
        self.assertEqual(distance, 8)

    def test_large_weights(self):
        """Тест графа с большими весами"""
        distance_matrix = [
            [0, 1000, 2000],
            [1000, 0, 3000],
            [2000, 3000, 0]
        ]
        route, distance = nearest_neighbor_tsp(distance_matrix)
        self.assertIsNotNone(route)
        self.assertIsNotNone(distance)
        # Ближайший сосед: 0->1->2->0 = 1000+3000+2000 = 6000
        self.assertEqual(distance, 6000)

    def test_zero_weights(self):
        """Тест графа с нулевыми весами"""
        distance_matrix = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        route, distance = nearest_neighbor_tsp(distance_matrix)
        self.assertIsNotNone(route)
        self.assertIsNotNone(distance)
        self.assertEqual(distance, 0)

    def test_mixed_finite_infinite_weights(self):
        """Тест графа со смешанными конечными и бесконечными весами"""
        distance_matrix = [
            [0, 1, inf, 4],
            [1, 0, 2, inf],
            [inf, 2, 0, 3],
            [4, inf, 3, 0]
        ]
        route, distance = nearest_neighbor_tsp(distance_matrix)
        self.assertIsNotNone(route)
        self.assertIsNotNone(distance)
        # Ближайший сосед должен найти путь: 0->1->2->3->0 = 1+2+3+4 = 10
        self.assertEqual(distance, 10)

    def test_greedy_choice_correctness(self):
        """Тест корректности жадного выбора"""
        # Граф, где ближайший сосед делает очевидные жадные выборы
        distance_matrix = [
            [0, 1, 10, 100],
            [1, 0, 2, 10],
            [10, 2, 0, 3],
            [100, 10, 3, 0]
        ]
        route, distance = nearest_neighbor_tsp(distance_matrix)
        self.assertIsNotNone(route)
        self.assertIsNotNone(distance)
        # Ожидаемый путь: 0->1->2->3->0 = 1+2+3+100 = 106
        self.assertEqual(distance, 106)
        expected_route = [0, 1, 2, 3, 0]
        self.assertEqual(route, expected_route)

    def test_no_valid_neighbors(self):
        """Тест случая, когда нет допустимых соседей на каком-то шаге"""
        distance_matrix = [
            [0, 1, inf],
            [1, 0, inf],
            [inf, inf, 0]
        ]
        route, distance = nearest_neighbor_tsp(distance_matrix)
        self.assertIsNone(route)
        self.assertIsNone(distance)

    def test_cannot_return_to_start(self):
        """Тест случая, когда нельзя вернуться в начальный город"""
        distance_matrix = [
            [0, 1, 2],
            [1, 0, 3],
            [inf, 3, 0]  # Нельзя вернуться из города 2 в город 0
        ]
        route, distance = nearest_neighbor_tsp(distance_matrix)
        self.assertIsNone(route)
        self.assertIsNone(distance)

    def test_path_validity(self):
        """Тест проверки корректности найденного пути"""
        distance_matrix = [
            [0, 2, 9, 10],
            [1, 0, 6, 4],
            [15, 7, 0, 8],
            [6, 3, 12, 0]
        ]
        route, distance = nearest_neighbor_tsp(distance_matrix)
        self.assertIsNotNone(route)
        self.assertIsNotNone(distance)

        # Проверяем, что путь начинается и заканчивается в городе 0
        self.assertEqual(route[0], 0)
        self.assertEqual(route[-1], 0)

        # Проверяем, что все промежуточные города уникальны
        intermediate_cities = route[1:-1]
        self.assertEqual(len(intermediate_cities), len(set(intermediate_cities)))

        # Проверяем, что посещены все города
        all_cities = set(route[:-1])
        expected_cities = set(range(len(distance_matrix)))
        self.assertEqual(all_cities, expected_cities)

        # Проверяем корректность вычисления стоимости
        calculated_distance = calculate_distance(route[:-1], distance_matrix)
        self.assertEqual(calculated_distance, distance)

    def test_diagonal_initialization(self):
        """Тест проверки правильной инициализации диагонали"""
        # Создаем матрицу с ненулевой диагональю
        distance_matrix = [
            [5, 1, 2],
            [1, 10, 3],
            [2, 3, 15]
        ]
        route, distance = nearest_neighbor_tsp(distance_matrix)
        self.assertIsNotNone(route)
        self.assertIsNotNone(distance)
        # Алгоритм должен обнулить диагональ внутри себя
        # Ожидаемый маршрут: 0->1->2->0 = 1+3+2 = 6
        self.assertEqual(distance, 6)

    def test_algorithm_determinism(self):
        """Тест детерминированности алгоритма"""
        distance_matrix = [
            [0, 3, 4, 2, 7],
            [3, 0, 4, 6, 3],
            [4, 4, 0, 5, 8],
            [2, 6, 5, 0, 6],
            [7, 3, 8, 6, 0]
        ]

        # Запускаем алгоритм несколько раз
        results = []
        for _ in range(5):
            route, distance = nearest_neighbor_tsp(distance_matrix)
            results.append((route, distance))

        # Все результаты должны быть одинаковыми
        first_result = results[0]
        for result in results[1:]:
            self.assertEqual(result, first_result)


    def test_suboptimal_solution(self):
        """Тест случая, где ближайший сосед даёт субоптимальное решение"""
        # Граф, где жадный выбор приводит к плохому результату
        distance_matrix = [
            [0, 1, 100, 1],
            [1, 0, 1, 100],
            [100, 1, 0, 1],
            [1, 100, 1, 0]
        ]
        route, distance = nearest_neighbor_tsp(distance_matrix)
        self.assertIsNotNone(route)
        self.assertIsNotNone(distance)

        # Ближайший сосед может выбрать: 0->1->2->3->0 = 1+1+1+1 = 4 (оптимально)
        # Или 0->3->2->1->0 = 1+1+1+1 = 4 (тоже оптимально в данном случае)
        self.assertEqual(distance, 4)


    def test_integration_file_to_algorithm(self):
        """Интеграционный тест: чтение файла и решение задачи"""
        content = """4
    0 10 15 20
    10 0 35 25
    15 35 0 30
    20 25 30 0"""
        filename = self.create_temp_file(content)
        matrix = read_graph_from_file(filename)
        route, distance = nearest_neighbor_tsp(matrix)
        self.assertIsNotNone(route)
        self.assertIsNotNone(distance)
        self.assertEqual(len(route), 5)
        # Ближайший сосед: 0->1->3->2->0 = 10+25+30+15 = 80
        self.assertEqual(distance, 80)


    def test_edge_case_all_infinite_except_required(self):
        """Тест случая, где есть только необходимые рёбра"""
        distance_matrix = [
            [0, 1, inf, inf],
            [inf, 0, 2, inf],
            [inf, inf, 0, 3],
            [4, inf, inf, 0]
        ]
        route, distance = nearest_neighbor_tsp(distance_matrix)
        self.assertIsNotNone(route)
        self.assertIsNotNone(distance)
        # Единственный возможный путь: 0->1->2->3->0 = 1+2+3+4 = 10
        self.assertEqual(distance, 10)
        expected_route = [0, 1, 2, 3, 0]
        self.assertEqual(route, expected_route)


    def test_multiple_equal_nearest_neighbors(self):
        """Тест случая с несколькими равно близкими соседями"""
        distance_matrix = [
            [0, 5, 5, 5],
            [5, 0, 5, 5],
            [5, 5, 0, 5],
            [5, 5, 5, 0]
        ]
        route, distance = nearest_neighbor_tsp(distance_matrix)
        self.assertIsNotNone(route)
        self.assertIsNotNone(distance)
        self.assertEqual(distance, 20)  # Любой путь стоит одинаково
        self.assertEqual(len(route), 5)


    def test_simple_graph_4_cities(self):
        """Тест простого графа с 4 городами"""
        distance_matrix = [
            [0, 10, 15, 20],
            [10, 0, 35, 25],
            [15, 35, 0, 30],
            [20, 25, 30, 0]
        ]
        route, distance = nearest_neighbor_tsp(distance_matrix)
        self.assertIsNotNone(route)
        self.assertIsNotNone(distance)
        self.assertEqual(len(route), 5)  # Включает возврат в начальный город
        self.assertEqual(route[0], 0)  # Начинаем с города 0
        self.assertEqual(route[-1], 0)  # Заканчиваем в городе 0
        # Проверяем, что все города посещены
        visited_cities = set(route[:-1])
        self.assertEqual(visited_cities, {0, 1, 2, 3})
        # Алгоритм ближайшего соседа может не найти оптимальное решение
        self.assertGreaterEqual(distance, 80)  # Не хуже оптимального


    def test_single_city(self):
        """Тест графа с одним городом"""
        distance_matrix = [[0]]
        route, distance = nearest_neighbor_tsp(distance_matrix)
        self.assertEqual(route, [0])
        self.assertEqual(distance, 0)


    def test_empty_graph(self):
        """Тест пустого графа"""
        distance_matrix = []
        route, distance = nearest_neighbor_tsp(distance_matrix)
        self.assertEqual(route, [0])
        self.assertEqual(distance, 0)


    def test_disconnected_graph(self):
        """Тест несвязного графа"""
        distance_matrix = [
            [0, 10, inf, inf],
            [10, 0, inf, inf],
            [inf, inf, 0, 5],
            [inf, inf, 5, 0]
        ]
        route, distance = nearest_neighbor_tsp(distance_matrix)
        self.assertIsNone(route)
        self.assertIsNone(distance)


    def test_no_hamiltonian_cycle(self):
        """Тест графа без гамильтонова цикла"""
        distance_matrix = [
            [0, 1, inf, inf],
            [1, 0, 2, inf],
            [inf, 2, 0, 3],
            [inf, inf, 3, 0]
        ]
        route, distance = nearest_neighbor_tsp(distance_matrix)
        self.assertIsNone(route)
        self.assertIsNone(distance)


if __name__ == '__main__':
    unittest.main()
