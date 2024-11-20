import matplotlib.pyplot as plt
import random

from ant_colony import Customer, AntColony

NUMBER_OF_CUSTOMERS = 16

def generateCustomers() -> list[Customer]:
    customers = []
    n_customers = NUMBER_OF_CUSTOMERS

    for i in range(n_customers):
        customers.append(Customer(i + 1, round(random.uniform(0, 100), 2), round(random.uniform(0, 100), 2)))

    return customers

def plotSolution(customers: list[Customer], solution: list[int], cost: float = 0) -> None:
    plt.scatter(customers[0].x, customers[0].y, color='red', zorder=5)
    plt.scatter([c.x for c in customers[1:]], [c.y for c in customers[1:]], color='cyan', zorder=5)

    for customer in customers[1:]:
        plt.text(customer.x, customer.y + 1, f'ID: {customer.id}', fontsize=6, ha='center', zorder=6)

    for i in range(len(solution) - 1):
        plt.plot([customers[solution[i]].x, customers[solution[i + 1]].x], [customers[solution[i]].y, customers[solution[i + 1]].y], color='blue')

    plt.title(f'Cost: {cost:.2f}')
    plt.legend(['Depot', 'Customers'], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    plt.show()

def main() -> None:
    customers = [Customer(0, 50, 50)]
    customers += generateCustomers()

    ac = AntColony(customers=customers, iterations=300, ants=16, alpha=1, beta=1, evaporation=0.3, Q=4, init_pheromone=1)
    result = ac.run()

    print(result)
    plotSolution(customers=customers, solution=result, cost = AntColony.calculate_route_cost(result, ac.distance_matrix))

if __name__ == "__main__":
    main()