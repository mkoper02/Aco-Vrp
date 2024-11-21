import numpy as np
from ant_colony import AntColony
import threading
from multiprocessing import Process, Lock, Pool

class AntColonyThreaded(AntColony):
    def run(self) -> list[int]:
        self.distance_matrix = self.generate_distance_matrix(self.customers)
        self.heuristic_matrix = self.generate_heuristic_matrix()
        self.pheromone_matrix = np.ones((len(self.customers), len(self.customers))) * self.init_pheromone

        best_route = (None, float('inf'))

        for _ in range(self.iterations):
            routes = []
            lock = Lock()
            
            def ant(*args) -> None:
                route = self.run_ant()
                with lock:
                    routes.append(route)

            with Pool(self.ants) as pool:
                pool.map(ant, range(self.ants))

            route = min(routes, key=lambda x: self.calculate_route_cost(x, self.distance_matrix))
            route_cost = self.calculate_route_cost(route, self.distance_matrix)

            self.update_pheromones(routes=routes)

            if route_cost < best_route[1]:
                best_route = (route, route_cost)

        return best_route[0]
        
