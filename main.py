from ant_colony import AntColony
from ant_colony_thread import AntColonyThreaded
from customer import Customer, generateCustomers, plotSolution, MAX_XY_VALUE

def main() -> None:
    customers = [Customer(0, MAX_XY_VALUE // 2, MAX_XY_VALUE // 2)]
    customers += generateCustomers()

    ac = AntColonyThreaded(customers=customers, iterations=500, ants=12, alpha=1, beta=2, evaporation=0.3, Q=4, init_pheromone=1)
    result = ac.run()

    print(result)
    plotSolution(customers=customers, solution=result, cost = AntColony.calculate_route_cost(result, ac.distance_matrix))

if __name__ == "__main__":
    main()