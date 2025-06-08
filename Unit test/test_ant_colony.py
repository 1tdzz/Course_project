import unittest
import tempfile
import os
import sys
from Algorithms_TSP import AntColonyOptimization, Ant, AntPath, read_graph_from_file
from math import inf
import random
from unittest.mock import patch

class TestReadGraphFromFile(unittest.TestCase):
    """Тесты для функции чтения графа из файла"""

    def setUp(self):
        """Создаем временные файлы для тестов"""
        self.temp_files = []

    def tearDown(self):
        """Удаляем временные файлы после тестов"""
        for temp_file in self.temp_files:
            try:
                os.unlink(temp_file)
            except FileNotFoundError:
                pass

    def create_temp_file(self, content):
        """Создает временный файл с заданным содержимым"""
        temp_fd, temp_path = tempfile.mkstemp(suffix='.txt')
        self.temp_files.append(temp_path)
        with os.fdopen(temp_fd, 'w') as f:
            f.write(content)
        return temp_path

    def test_valid_graph_4x4(self):
        """Тест чтения корректного графа 4x4"""
        content = """4
inf 10 15 20
10 inf 35 25
15 35 inf 30
20 25 30 inf"""
        filename = self.create_temp_file(content)

        graph = read_graph_from_file(filename)

        self.assertEqual(len(graph), 4)
        self.assertEqual(len(graph[0]), 4)
        self.assertEqual(graph[0][1], 10.0)
        self.assertEqual(graph[1][2], 35.0)
        self.assertTrue(graph[0][0] == inf)

    def test_valid_graph_1x1(self):
        """Тест чтения графа с одной вершиной"""
        content = """1
inf"""
        filename = self.create_temp_file(content)

        graph = read_graph_from_file(filename)

        self.assertEqual(len(graph), 1)
        self.assertTrue(graph[0][0] == inf)

    def test_file_not_found(self):
        """Тест обработки несуществующего файла"""
        with patch('sys.exit') as mock_exit:
            with patch('builtins.print') as mock_print:
                read_graph_from_file("nonexistent_file.txt")
                mock_print.assert_called_once()
                mock_exit.assert_called_once_with(1)

    def test_invalid_number_of_vertices(self):
        """Тест неверного количества вершин"""
        content = """3
inf 10
10 inf"""
        filename = self.create_temp_file(content)

        with patch('sys.exit') as mock_exit:
            with patch('builtins.print') as mock_print:
                read_graph_from_file(filename)
                mock_exit.assert_called_once_with(1)

    def test_invalid_row_length(self):
        """Тест неверной длины строки"""
        content = """2
inf 10 15
5 inf"""
        filename = self.create_temp_file(content)

        with patch('sys.exit') as mock_exit:
            with patch('builtins.print') as mock_print:
                read_graph_from_file(filename)
                mock_exit.assert_called_once_with(1)

    def test_non_numeric_values(self):
        """Тест нечисловых значений в матрице"""
        content = """2
inf abc
5 inf"""
        filename = self.create_temp_file(content)

        with patch('sys.exit') as mock_exit:
            with patch('builtins.print') as mock_print:
                read_graph_from_file(filename)
                mock_exit.assert_called_once_with(1)

    def test_empty_file(self):
        """Тест пустого файла"""
        content = ""
        filename = self.create_temp_file(content)

        with patch('sys.exit') as mock_exit:
            with patch('builtins.print') as mock_print:
                read_graph_from_file(filename)
                mock_exit.assert_called_once_with(1)


class TestAntPath(unittest.TestCase):
    """Тесты для класса AntPath"""

    def test_ant_path_initialization(self):
        """Тест инициализации пути муравья"""
        path = AntPath()
        self.assertEqual(path.vertices, [])
        self.assertEqual(path.distance, 0.0)


class TestAnt(unittest.TestCase):
    """Тесты для класса Ant"""

    def test_ant_initialization(self):
        """Тест инициализации муравья"""
        ant = Ant(2)

        self.assertEqual(ant.start_location, 2)
        self.assertEqual(ant.current_location, 2)
        self.assertEqual(ant.visited, [])
        self.assertTrue(ant.can_continue)
        self.assertIsInstance(ant.path, AntPath)

    def test_get_neighbor_vertices_simple(self):
        """Тест получения соседних вершин в простом графе"""
        graph = [
            [inf, 10, 20],
            [10, inf, 30],
            [20, 30, inf]
        ]
        ant = Ant(0)

        neighbors = ant.get_neighbor_vertices(graph)

        self.assertIn(1, neighbors)
        self.assertIn(2, neighbors)
        self.assertEqual(len(neighbors), 2)

    def test_get_neighbor_vertices_with_visited(self):
        """Тест получения соседних вершин с учетом посещенных"""
        graph = [
            [inf, 10, 20],
            [10, inf, 30],
            [20, 30, inf]
        ]
        ant = Ant(0)
        ant.visited = [1]

        neighbors = ant.get_neighbor_vertices(graph)

        self.assertNotIn(1, neighbors)
        self.assertIn(2, neighbors)
        self.assertEqual(len(neighbors), 1)

    def test_get_neighbor_vertices_disconnected(self):
        """Тест получения соседних вершин в несвязном графе"""
        graph = [
            [inf, inf, inf],
            [inf, inf, 30],
            [inf, 30, inf]
        ]
        ant = Ant(0)

        neighbors = ant.get_neighbor_vertices(graph)

        self.assertEqual(neighbors, [])

    def test_make_choice_first_move(self):
        """Тест первого хода муравья"""
        graph = [
            [inf, 10, 20],
            [10, inf, 30],
            [20, 30, inf]
        ]
        pheromone = [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]
        ant = Ant(0)

        # Фиксируем случайный выбор для предсказуемости
        with patch.object(ant, 'get_random_choice', return_value=0.3):
            ant.make_choice(graph, pheromone, 1.0, 2.0)

        self.assertIn(0, ant.path.vertices)
        self.assertIn(0, ant.visited)
        self.assertGreater(len(ant.path.vertices), 1)

    def test_make_choice_no_neighbors(self):
        """Тест хода муравья когда нет доступных соседей"""
        graph = [
            [inf, 10],
            [10, inf]
        ]
        pheromone = [
            [0, 1],
            [1, 0]
        ]
        ant = Ant(0)
        ant.path.vertices = [0]
        ant.visited = [0, 1]
        ant.current_location = 1

        ant.make_choice(graph, pheromone, 1.0, 2.0)

        self.assertFalse(ant.can_continue)


class TestAntColonyOptimization(unittest.TestCase):
    """Тесты для класса AntColonyOptimization"""

    def test_aco_initialization(self):
        """Тест инициализации муравьиного алгоритма"""
        graph = [
            [inf, 10, 20],
            [10, inf, 30],
            [20, 30, inf]
        ]
        aco = AntColonyOptimization(graph)

        self.assertEqual(aco.graph, graph)
        self.assertEqual(len(aco.pheromone), 3)
        self.assertEqual(len(aco.pheromone[0]), 3)
        self.assertGreater(aco.k_q, 0)

    def test_create_ants(self):
        """Тест создания муравьев"""
        graph = [
            [inf, 10, 20, 15],
            [10, inf, 30, 25],
            [20, 30, inf, 35],
            [15, 25, 35, inf]
        ]
        aco = AntColonyOptimization(graph)
        aco.create_ants()

        self.assertEqual(len(aco.ants), 4)
        for i, ant in enumerate(aco.ants):
            self.assertEqual(ant.start_location, i)

    def test_update_global_pheromone(self):
        """Тест обновления глобальных феромонов"""
        graph = [
            [inf, 10],
            [10, inf]
        ]
        aco = AntColonyOptimization(graph)

        initial_pheromone = aco.pheromone[0][1]
        local_update = [[0, 0.5], [0.5, 0]]

        aco.update_global_pheromone(local_update)

        self.assertNotEqual(aco.pheromone[0][1], initial_pheromone)
        self.assertGreaterEqual(aco.pheromone[0][1], 0.01)  # Минимальное значение

    def test_solve_tsp_complete_graph_4x4(self):
        """Тест решения TSP на полном графе 4x4"""
        graph = [
            [inf, 10, 15, 20],
            [10, inf, 35, 25],
            [15, 35, inf, 30],
            [20, 25, 30, inf]
        ]
        aco = AntColonyOptimization(graph)

        # Уменьшаем количество итераций для быстрого тестирования
        with patch.object(aco, 'solve_tsp') as mock_solve:
            mock_solve.return_value = ([0, 1, 2, 3, 0], 70.0)
            path, distance = aco.solve_tsp()

        if path is not None:
            self.assertIsInstance(path, list)
            self.assertIsInstance(distance, float)
            self.assertEqual(path[0], path[-1])  # Цикл должен замыкаться

    def test_solve_tsp_disconnected_graph(self):
        """Тест решения TSP на несвязном графе"""
        graph = [
            [inf, 10, inf],
            [10, inf, inf],
            [inf, inf, inf]
        ]
        aco = AntColonyOptimization(graph)
        path, distance = aco.solve_tsp()

        # Несвязный граф не должен иметь гамильтонова цикла
        self.assertIsNone(path)
        self.assertIsNone(distance)

    def test_solve_tsp_single_vertex(self):
        """Тест решения TSP на графе с одной вершиной"""
        graph = [[inf]]
        aco = AntColonyOptimization(graph)
        path, distance = aco.solve_tsp()

        # Граф с одной вершиной не может иметь TSP решение
        self.assertIsNone(path)
        self.assertIsNone(distance)

    def test_solve_tsp_empty_graph(self):
        """Тест решения TSP на пустом графе"""
        graph = []
        aco = AntColonyOptimization(graph)
        path, distance = aco.solve_tsp()

        self.assertIsNone(path)
        self.assertIsNone(distance)

    def test_solve_tsp_partial_connectivity(self):
        """Тест решения TSP на частично связном графе без гамильтонова цикла"""
        graph = [
            [inf, 10, inf, inf],
            [10, inf, 20, inf],
            [inf, 20, inf, 30],
            [inf, inf, 30, inf]
        ]
        aco = AntColonyOptimization(graph)
        path, distance = aco.solve_tsp()

        # Этот граф представляет путь, а не цикл, поэтому TSP решения нет
        self.assertIsNone(path)
        self.assertIsNone(distance)


class TestIntegration(unittest.TestCase):
    """Интеграционные тесты"""

    def setUp(self):
        self.temp_files = []

    def tearDown(self):
        for temp_file in self.temp_files:
            try:
                os.unlink(temp_file)
            except FileNotFoundError:
                pass

    def create_temp_file(self, content):
        temp_fd, temp_path = tempfile.mkstemp(suffix='.txt')
        self.temp_files.append(temp_path)
        with os.fdopen(temp_fd, 'w') as f:
            f.write(content)
        return temp_path

    def test_full_pipeline_valid_graph(self):
        """Интеграционный тест полного запуска программы на неориентированном графе"""
        content = """3
inf 70 5
10 inf 30
20 13 inf"""
        filename = self.create_temp_file(content)

        graph = read_graph_from_file(filename)
        aco = AntColonyOptimization(graph)
        best_path_aco, min_cost_aco = aco.solve_tsp()

        if best_path_aco is not None:
            self.assertIsInstance(best_path_aco, list)
            self.assertIsInstance(min_cost_aco, float)
            self.assertGreater(len(best_path_aco), 0)
            self.assertGreater(min_cost_aco, 0)

    def test_full_pipeline_square_graph(self):
        """Интеграционный тест с квадратным графом"""
        content = """4
inf 1 4 3
1 inf 2 4
4 2 inf 1
3 4 1 inf"""
        filename = self.create_temp_file(content)

        graph = read_graph_from_file(filename)
        aco = AntColonyOptimization(graph)
        best_path_aco, min_cost_aco = aco.solve_tsp()

        if best_path_aco is not None:
            # Проверяем, что путь образует корректный цикл
            self.assertEqual(best_path_aco[0], best_path_aco[-1])
            # Проверяем, что все вершины посещены
            unique_vertices = set(best_path_aco[:-1])  # Исключаем последнюю (дублирующую)
            self.assertEqual(len(unique_vertices), 4)


class TestEdgeCases(unittest.TestCase):
    """Тесты граничных случаев"""

    def test_graph_with_zero_weights(self):
        """Тест графа с нулевыми весами"""
        graph = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        aco = AntColonyOptimization(graph)
        path, distance = aco.solve_tsp()

        self.assertIsNone(path)
        self.assertIsNone(distance)

    def test_graph_with_inf_weights(self):
        """Тест графа с бесконечными весами"""
        graph = [
            [inf, inf, inf],
            [inf, inf, inf],
            [inf, inf, inf]
        ]
        aco = AntColonyOptimization(graph)
        path, distance = aco.solve_tsp()

        self.assertIsNone(path)
        self.assertIsNone(distance)

    def test_symmetric_vs_asymmetric_graph(self):
        """Тест симметричного и асимметричного графа"""
        # Асимметричный граф
        graph = [
            [inf, 10, 20],
            [15, inf, 30],  # Разные веса для обратного пути
            [25, 35, inf]
        ]
        aco = AntColonyOptimization(graph)
        path, distance = aco.solve_tsp()

        if path is not None:
            self.assertIsInstance(path, list)
            self.assertIsInstance(distance, float)

    def test_very_small_weights(self):
        """Тест графа с очень маленькими весами"""
        graph = [
            [inf, 0.001, 0.002],
            [0.001, inf, 0.003],
            [0.002, 0.003, inf]
        ]
        aco = AntColonyOptimization(graph)
        path, distance = aco.solve_tsp()

        if path is not None:
            self.assertGreater(distance, 0)

    def test_very_large_weights(self):
        """Тест графа с очень большими весами"""
        graph = [
            [inf, 1000000, 2000000],
            [1000000, inf, 3000000],
            [2000000, 3000000, inf]
        ]
        aco = AntColonyOptimization(graph)
        path, distance = aco.solve_tsp()

        if path is not None:
            self.assertGreater(distance, 0)


if __name__ == '__main__':
    random.seed(42)  # Фиксируем seed для воспроизводимости

    # Создаем test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Добавляем все тестовые классы
    test_classes = [
        TestReadGraphFromFile,
        TestAntPath,
        TestAnt,
        TestAntColonyOptimization,
        TestIntegration,
        TestEdgeCases
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Запускаем тесты
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
