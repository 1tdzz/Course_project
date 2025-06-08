import unittest
import tempfile
import os
from Algorithms_TSP import brute_force_tsp, calculate_distance, read_graph_from_file
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
            [inf, 10, 15, 20],
            [10, 0, 35, 25],
            [15, 35, 0, 30],
            [20, 25, 30, 0]
        ]
        route, distance = brute_force_tsp(distance_matrix)
        self.assertIsNotNone(route)
        self.assertIsNotNone(distance)
        self.assertEqual(len(route), 4)
        # Проверяем, что маршрут содержит все города
        self.assertEqual(set(route), {0, 1, 2, 3})
        # Проверяем минимальную стоимость (оптимальный маршрут: 0->1->3->2->0 = 80)
        self.assertEqual(distance, 80)

    def test_single_city(self):
        """Тест графа с одним городом"""
        distance_matrix = [[0]]
        route, distance = brute_force_tsp(distance_matrix)
        self.assertEqual(route, [0])
        self.assertEqual(distance, 0)

    def test_empty_graph(self):
        """Тест пустого графа"""
        distance_matrix = []
        path, cost = brute_force_tsp(distance_matrix)
        self.assertEqual(path, [0])
        self.assertEqual(cost, 0)

    def test_graph_with_no_cycle(self):
        """Тест графа без циклов"""
        matrix = [
            [inf, inf],
            [inf, inf]
        ]
        path, cost = brute_force_tsp(matrix)
        self.assertIsNone(path)
        self.assertIsNone(cost)

    def test_two_cities_connected(self):
        """Тест графа с двумя связанными городами"""
        distance_matrix = [
            [0, 5],
            [5, 0]
        ]
        route, distance = brute_force_tsp(distance_matrix)
        self.assertIsNotNone(route)
        self.assertIsNotNone(distance)
        self.assertEqual(len(route), 2)
        self.assertEqual(distance, 10)  # 0->1->0 = 5+5 = 10

    def test_disconnected_graph(self):
        """Тест несвязного графа"""
        distance_matrix = [
            [0, 10, inf, inf],
            [10, 0, inf, inf],
            [inf, inf, 0, 5],
            [inf, inf, 5, 0]
        ]
        route, distance = brute_force_tsp(distance_matrix)
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
        route, distance = brute_force_tsp(distance_matrix)
        self.assertIsNone(route)
        self.assertIsNone(distance)

    def test_asymmetric_graph(self):
        """Тест асимметричного графа"""
        distance_matrix = [
            [0, 1, 5],
            [2, 0, 3],
            [4, 6, 0]
        ]
        route, distance = brute_force_tsp(distance_matrix)
        self.assertIsNotNone(route)
        self.assertIsNotNone(distance)
        self.assertEqual(len(route), 3)
        # Оптимальный маршрут: 0->1->2->0 = 1+3+4 = 8
        self.assertEqual(distance, 8)

    def test_large_weights(self):
        """Тест графа с большими весами"""
        distance_matrix = [
            [0, 1000, 2000],
            [1000, 0, 3000],
            [2000, 3000, 0]
        ]
        route, distance = brute_force_tsp(distance_matrix)
        self.assertIsNotNone(route)
        self.assertIsNotNone(distance)
        self.assertEqual(distance, 6000)

    def test_zero_weights(self):
        """Тест графа с нулевыми весами"""
        distance_matrix = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        route, distance = brute_force_tsp(distance_matrix)
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
        route, distance = brute_force_tsp(distance_matrix)
        self.assertIsNotNone(route)
        self.assertIsNotNone(distance)
        # Единственно возможный маршрут: 0->1->2->3->0 = 1+2+3+4 = 10
        self.assertEqual(distance, 10)

    def test_calculate_distance_function(self):
        """Тест функции calculate_distance"""
        distance_matrix = [
            [0, 10, 15],
            [10, 0, 20],
            [15, 20, 0]
        ]

        # Валидный маршрут
        route = [0, 1, 2]
        distance = calculate_distance(route, distance_matrix)
        self.assertEqual(distance, 45)  # 10 + 20 + 15 = 45

        # Маршрут с бесконечным весом
        distance_matrix[1][2] = inf
        distance = calculate_distance(route, distance_matrix)
        self.assertIsNone(distance)

    def test_read_graph_from_file_valid(self):
        """Тест чтения корректного файла"""
        content = """3
0 1 2
1 0 3
2 3 0"""
        filename = self.create_temp_file(content)
        matrix = read_graph_from_file(filename)
        expected = [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]]
        self.assertEqual(matrix, expected)

    def test_read_graph_from_file_with_inf(self):
        """Тест чтения файла с бесконечными весами"""
        content = """2
0 inf
inf 0"""
        filename = self.create_temp_file(content)
        matrix = read_graph_from_file(filename)
        expected = [[0.0, inf], [inf, 0.0]]
        self.assertEqual(matrix, expected)

    def test_read_graph_nonexistent_file(self):
        """Тест чтения несуществующего файла"""
        with self.assertRaises(SystemExit):
            read_graph_from_file("nonexistent_file.txt")

    def test_read_graph_invalid_format(self):
        """Тест чтения файла с неверным форматом"""
        # Неверное количество элементов в строке
        content = """2
0 1 2
1 0"""
        filename = self.create_temp_file(content)
        with self.assertRaises(SystemExit):
            read_graph_from_file(filename)

    def test_read_graph_wrong_number_of_rows(self):
        """Тест чтения файла с неверным количеством строк"""
        content = """3
0 1 2
1 0 3"""
        filename = self.create_temp_file(content)
        with self.assertRaises(SystemExit):
            read_graph_from_file(filename)

    def test_integration_file_to_algorithm(self):
        """Интеграционный тест: чтение файла и решение задачи"""
        content = """4
0 10 15 20
10 0 35 25
15 35 0 30
20 25 30 0"""
        filename = self.create_temp_file(content)
        matrix = read_graph_from_file(filename)
        route, distance = brute_force_tsp(matrix)
        self.assertIsNotNone(route)
        self.assertIsNotNone(distance)
        self.assertEqual(len(route), 4)
        self.assertEqual(distance, 80)
if __name__ == '__main__':
    unittest.main()
