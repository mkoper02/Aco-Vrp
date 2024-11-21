import numpy as np
from customer import Customer

class AntColony:
    def __init__(self, customers: list[Customer], iterations: int, ants: int, alpha: int, beta: int, evaporation: float, Q: int, init_pheromone: float):
        self.customers = customers
        self.iterations = iterations
        self.ants = ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.Q = Q
        self.init_pheromone = init_pheromone

    def run(self) -> list[int]:
        self.distance_matrix = self.generate_distance_matrix(self.customers)
        self.heuristic_matrix = self.generate_heuristic_matrix()
        self.pheromone_matrix = np.ones((len(self.customers), len(self.customers))) * self.init_pheromone

        best_route = (None, float('inf'))

        for _ in range(self.iterations):
            routes = []

            for _ in range(self.ants):
                routes.append(self.run_ant())

            route = min(routes, key=lambda x: self.calculate_route_cost(x, self.distance_matrix))
            route_cost = self.calculate_route_cost(route, self.distance_matrix)

            self.update_pheromones(routes=routes)

            if route_cost < best_route[1]:
                best_route = (route, route_cost)

        return best_route[0]

    def run_ant(self) -> list[int]:
        visited = [0]

        for _ in range(len(self.customers) - 1):
            current_customer = visited[-1]
            next_customer = self.select_next_customer(current_customer, visited)
            visited.append(next_customer)

        visited.append(0)

        return visited
    
    def update_pheromones(self, routes: list[int]) -> None:
        self.pheromone_matrix *= self.evaporation

        for route in routes:
            cost = self.calculate_route_cost(route, self.distance_matrix)
            for i in range(len(route) - 1):
                self.pheromone_matrix[route[i], route[i + 1]] += self.Q / cost

    def generate_heuristic_matrix(self) -> np.ndarray:
        n_customers = len(self.customers)
        heuristic_matrix = np.zeros((n_customers, n_customers))

        for i in range(n_customers):
            for j in range(n_customers):
                if i == j:
                    heuristic_matrix[i, j] = 0
                else:
                    heuristic_matrix[i, j] = self.Q / self.distance_matrix[i, j]

        return heuristic_matrix
    
    @staticmethod
    def static_generate_heuristic_matrix(n_customers: int, Q: int, distance_matrix: np.ndarray) -> np.ndarray:
        heuristic_matrix = np.zeros((n_customers, n_customers))

        for i in range(n_customers):
            for j in range(n_customers):
                if i == j:
                    heuristic_matrix[i, j] = 0
                else:
                    heuristic_matrix[i, j] = Q / distance_matrix[i, j]

        return heuristic_matrix
    
    def calculate_probabilites(self, current_customer: int, visited: list[int]) -> np.ndarray:
        n_customers = len(self.customers)
        unvisited = [i for i in range(n_customers) if i not in visited]
        probabilities = np.zeros(n_customers)

        for i in unvisited:
            probabilities[i] = self.pheromone_matrix[current_customer, i] ** self.alpha * self.heuristic_matrix[current_customer, i] ** self.beta

        probabilities = probabilities / np.sum(probabilities)

        return probabilities
    
    def select_next_customer(self, current_customer: int, visited: list[int]) -> int:
        probabilities = self.calculate_probabilites(current_customer, visited)
        return np.random.choice(len(self.customers), p=probabilities)
    
    @staticmethod
    def static_select_next_customer(current_customer: int, visited: list[int], pheromone_matrix: np.ndarray, heuristic_matrix: np.ndarray, alpha: int, beta: int) -> int:
        n_customers = pheromone_matrix.shape[0]
        unvisited = [i for i in range(n_customers) if i not in visited]
        probabilities = np.zeros(n_customers)

        for i in unvisited:
            probabilities[i] = pheromone_matrix[current_customer, i] ** alpha * heuristic_matrix[current_customer, i] ** beta

        probabilities = probabilities / np.sum(probabilities)

        return np.random.choice(n_customers, p=probabilities)

    @staticmethod
    def generate_distance_matrix(customers: list[Customer]) -> np.ndarray:
        n_customers = len(customers)
        distance_matrix = np.zeros((n_customers, n_customers))

        for i in range(n_customers):
            for j in range(n_customers):
                distance_matrix[i, j] = AntColony.calculate_distance(np.array([customers[i].x, customers[i].y]), np.array([customers[j].x, customers[j].y]))

        return distance_matrix

    @staticmethod
    def calculate_distance(a: np.ndarray, b: np.ndarray) -> float:
        return np.linalg.norm(a - b)
    
    @staticmethod
    def calculate_route_cost(route: list[int], distance_matrix: np.ndarray) -> float:
        cost = 0

        for i in range(len(route) - 1):
            cost += distance_matrix[route[i], route[i + 1]]

        return cost