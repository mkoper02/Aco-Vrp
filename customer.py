import random
import matplotlib.pyplot as plt
from typing import NamedTuple

NUMBER_OF_CUSTOMERS = 25
MAX_XY_VALUE = 250

class Customer(NamedTuple):
    id: int
    x: float
    y: float

def generateCustomers() -> list[Customer]:
    customers = []
    n_customers = NUMBER_OF_CUSTOMERS

    for i in range(n_customers):
        customers.append(Customer(i + 1, round(random.uniform(0, MAX_XY_VALUE), 2), round(random.uniform(0, MAX_XY_VALUE), 2)))

    return customers

def plotSolution(customers: list[Customer], solution: list[int], cost: float = 0) -> None:
    plt.scatter(customers[0].x, customers[0].y, color='red', zorder=5)
    plt.scatter([c.x for c in customers[1:]], [c.y for c in customers[1:]], color='cyan', zorder=5)

    for customer in customers[1:]:
        plt.text(customer.x, customer.y + 1, f'ID: {customer.id}', fontsize=6, ha='center', zorder=6)

    for i in range(len(solution) - 1):
        plt.plot([customers[solution[i]].x, customers[solution[i + 1]].x], [customers[solution[i]].y, customers[solution[i + 1]].y], color='orange')

    plt.title(f'Cost: {cost:.2f}')
    plt.legend(['Depot', 'Customers'], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    plt.show()