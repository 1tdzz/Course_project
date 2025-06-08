import unittest
import tempfile
import os
from Algorithms_TSP import tsp_dp, read_graph_from_file, calculate_distance
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

    def test_simple_graph_4_cities(self):
        """Тест простого графа с 4 городами"""
        distance_matrix = [
            [0, 10, 15, 20],
            [10, 0, 35, 25],
            [15, 35, 0, 30],
            [20, 25, 30, 0]
        ]
        path, distance = tsp_dp(distance_matrix)
        self.assertIsNotNone(path)
        self.assertIsNotNone(distance)
        self.assertEqual(len(path), 5)  # Включает возврат в начальный город
        self.assertEqual(path[0], 0)  # Начинаем с города 0
        self.assertEqual(path[-1], 0)  # Заканчиваем в городе 0
        # Проверяем, что все города посещены
        visited_cities = set(path[:-1])  # Исключаем последний элемент (возврат)
        self.assertEqual(visited_cities, {0, 1, 2, 3})
        # Проверяем оптимальную стоимость
        self.assertEqual(distance, 80)

    def test_single_city(self):
        """Тест графа с одним городом"""
        distance_matrix = [[0]]
        path, distance = tsp_dp(distance_matrix)
        self.assertEqual(path, [0])
        self.assertEqual(distance, 0)

    def test_empty_graph(self):
        """Тест пустого графа"""
        distance_matrix = []
        path, distance = tsp_dp(distance_matrix)
        self.assertIsNone(path)
        self.assertIsNone(distance)

    def test_disconnected_graph(self):
        """Тест несвязного графа"""
        distance_matrix = [
            [0, 10, inf, inf],
            [10, 0, inf, inf],
            [inf, inf, 0, 5],
            [inf, inf, 5, 0]
        ]
        path, distance = tsp_dp(distance_matrix)
        self.assertIsNone(path)
        self.assertIsNone(distance)

    def test_no_hamiltonian_cycle(self):
        """Тест графа без гамильтонова цикла"""
        distance_matrix = [
            [0, 1, inf, inf],
            [1, 0, 2, inf],
            [inf, 2, 0, 3],
            [inf, inf, 3, 0]
        ]
        path, distance = tsp_dp(distance_matrix)
        self.assertIsNone(path)
        self.assertIsNone(distance)

    def test_asymmetric_graph(self):
        """Тест асимметричного графа"""
        distance_matrix = [
            [0, 1, 5],
            [2, 0, 3],
            [4, 6, 0]
        ]
        path, distance = tsp_dp(distance_matrix)
        self.assertIsNotNone(path)
        self.assertIsNotNone(distance)
        self.assertEqual(len(path), 4)
        self.assertEqual(path[0], 0)
        self.assertEqual(path[-1], 0)
        self.assertEqual(distance, 8)

    def test_large_weights(self):
        """Тест графа с большими весами"""
        distance_matrix = [
            [0, 1000, 2000],
            [1000, 0, 3000],
            [2000, 3000, 0]
        ]
        path, distance = tsp_dp(distance_matrix)
        self.assertIsNotNone(path)
        self.assertIsNotNone(distance)
        self.assertEqual(distance, 6000)

    def test_zero_weights(self):
        """Тест графа с нулевыми весами"""
        distance_matrix = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        path, distance = tsp_dp(distance_matrix)
        self.assertIsNotNone(path)
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
        path, distance = tsp_dp(distance_matrix)
        self.assertIsNotNone(path)
        self.assertIsNotNone(distance)
        self.assertEqual(distance, 10)

    def test_path_validity(self):
        """Тест проверки корректности найденного пути"""
        distance_matrix = [
            [0, 2, 9, 10],
            [1, 0, 6, 4],
            [15, 7, 0, 8],
            [6, 3, 12, 0]
        ]
        path, distance = tsp_dp(distance_matrix)
        self.assertIsNotNone(path)
        self.assertIsNotNone(distance)

        # Проверяем, что путь начинается и заканчивается в городе 0
        self.assertEqual(path[0], 0)
        self.assertEqual(path[-1], 0)

        # Проверяем, что все промежуточные города уникальны
        intermediate_cities = path[1:-1]
        self.assertEqual(len(intermediate_cities), len(set(intermediate_cities)))

        # Проверяем, что посещены все города
        all_cities = set(path[:-1])
        expected_cities = set(range(len(distance_matrix)))
        self.assertEqual(all_cities, expected_cities)

        # Проверяем корректность вычисления стоимости
        calculated_distance = calculate_distance(path[:-1], distance_matrix)
        self.assertEqual(calculated_distance, distance)

    def test_small_complete_graphs(self):
        """Тест малых полных графов разных размеров"""
        # Граф размера 3
        matrix_3 = [
            [0, 4, 1],
            [4, 0, 2],
            [1, 2, 0]
        ]
        path, distance = tsp_dp(matrix_3)
        self.assertEqual(distance, 7)  # 0->2->1->0 = 1+2+4 = 7

        # Граф размера 5
        matrix_5 = [
            [0, 1, 2, 3, 4],
            [1, 0, 5, 6, 7],
            [2, 5, 0, 8, 9],
            [3, 6, 8, 0, 10],
            [4, 7, 9, 10, 0]
        ]
        path, distance = tsp_dp(matrix_5)
        self.assertIsNotNone(path)
        self.assertIsNotNone(distance)
        self.assertEqual(len(path), 6)

    def test_partial_connectivity(self):
        """Тест графа с частичной связностью"""
        distance_matrix = [
            [0, 1, inf, 10],
            [1, 0, 5, inf],
            [inf, 5, 0, 2],
            [10, inf, 2, 0]
        ]
        path, distance = tsp_dp(distance_matrix)
        self.assertIsNotNone(path)
        self.assertIsNotNone(distance)
        # Единственный возможный путь: 0->1->2->3->0 = 1+5+2+10 = 18
        self.assertEqual(distance, 18)

    def test_integration_file_to_algorithm(self):
        """Интеграционный тест: чтение файла и решение задачи"""
        content = """4
    0 10 15 20
    10 0 35 25
    15 35 0 30
    20 25 30 0"""
        filename = self.create_temp_file(content)
        matrix = read_graph_from_file(filename)
        path, distance = tsp_dp(matrix)
        self.assertIsNotNone(path)
        self.assertIsNotNone(distance)
        self.assertEqual(len(path), 5)
        self.assertEqual(distance, 80)

if __name__ == '__main__':
    unittest.main()
