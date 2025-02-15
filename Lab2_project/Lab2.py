import math
import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache
from scipy.special import factorial


class ProbabilityModel:
    def __init__(self, lambda_val):
        self.lambda_val = lambda_val

    @lru_cache(maxsize=None)
    def calculate_probability(self, i):
        if i == 1:
            return 1
        if i <= 0:
            return 0
        return (1 - 1 / i) ** (i - 1)

    @lru_cache(maxsize=None)
    def poisson_distribution(self, j):
        if j < 0:
            return 0
        if j == 0:
            return math.exp(-self.lambda_val)
        return (self.lambda_val ** j) * math.exp(-self.lambda_val) / factorial(j)

    @lru_cache(maxsize=None)
    def probability_to_zero(self, j):
        return self.poisson_distribution(j)

    @lru_cache(maxsize=None)
    def probability_from_i_to_0(self, i):
        if i < 1:
            return 0
        return self.calculate_probability(i) * self.poisson_distribution(0)

    @lru_cache(maxsize=None)
    def probability_to_stay(self, i):
        return (1 - self.calculate_probability(i)) * self.poisson_distribution(0) + self.calculate_probability(i) * self.poisson_distribution(1)

    @lru_cache(maxsize=None)
    def probability_from_i_to_j(self, i, j):
        if j < i:
            return 0
        return self.calculate_probability(i) * self.poisson_distribution(j - i + 1) + (1 - self.calculate_probability(i)) * self.poisson_distribution(j - i)


def process_data(lambda_val, slot_count):
    user_count = []
    state = []
    current_state = 0
    for _ in range(slot_count):
        poisson_val = np.random.poisson(lambda_val)
        if current_state > 0:
            transmission_prob = 1 / current_state
            binomial_val = np.random.binomial(current_state, transmission_prob)
        else:
            binomial_val = 0
        if binomial_val == 1:
            current_state -= 1
        elif binomial_val > 1:
            pass
        current_state += poisson_val
        user_count.append(current_state)
        state.append(1 if binomial_val == 1 else 0)
    avg_users = np.mean(user_count)
    avg_time = avg_users / lambda_val
    avg_state = np.mean(state)
    return avg_users, avg_time, avg_state


def transition_matrix(lambda_val, max_users):
    prob_model = ProbabilityModel(lambda_val)
    matrix = np.zeros((max_users + 1, max_users + 1))
    for i in range(max_users + 1):
        for j in range(max_users + 1):
            if i == 0:
                matrix[i, j] = prob_model.probability_to_zero(j)
            elif i == j:
                matrix[i, j] = prob_model.probability_to_stay(i)
            elif i < j:
                matrix[i, j] = prob_model.probability_from_i_to_j(i, j)
            elif i == j + 1:
                matrix[i, j] = prob_model.probability_from_i_to_0(i)
            else:
                matrix[i, j] = 0

    matrix_with_ones = np.vstack([matrix.T - np.eye(max_users + 1), np.ones(max_users + 1)])

    coeffs = np.zeros(max_users + 2)
    coeffs[-1] = 1

    steady_state = np.linalg.lstsq(matrix_with_ones, coeffs, rcond=None)[0]
    return steady_state


def average_users_across_lambdas(lambda_values, max_users):
    averages = []
    for lambda_val in lambda_values:
        steady_state = transition_matrix(lambda_val, max_users)
        avg_users = np.dot(steady_state, np.arange(max_users + 1))
        averages.append(avg_users)
    return averages


def plot_results(lambda_values, avg_results, max_users_values, simulation_results):
    plt.figure(figsize=(12, 7))

    colors = plt.cm.plasma(np.linspace(0, 1, len(max_users_values)))
    markers = ['o', 's', 'D', '^', 'v']

    for idx, max_users in enumerate(max_users_values):
        plt.plot(lambda_values, avg_results[idx],
                 label=f"Макс. пользователей = {max_users}",
                 marker=markers[idx % len(markers)],
                 linestyle='-',
                 color=colors[idx],
                 markersize=9,
                 linewidth=2)

    plt.plot(lambda_values, simulation_results,
             label="Имитационное моделирование для N = 100000",
             marker='o',
             linestyle='-',
             color='red',
             markersize=8,
             linewidth=3)

    plt.xlabel("Поток λ", fontsize=16, fontweight='bold', color='darkblue')
    plt.ylabel("Среднее количество пользователей", fontsize=16, fontweight='bold', color='darkblue')
    plt.yscale("log")
    plt.title("График зависимости среднего количества пользователей от интенсивности λ", fontsize=18, fontweight='bold', color='darkgreen')
    plt.axvline(x=np.exp(-1), color='orange', linestyle='--', label='λ = exp(-1)', linewidth=2)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle=':', linewidth=0.7, color='gray')
    plt.tight_layout()
    plt.savefig(f"lambda_plot.png")
    plt.show()


def main():
    lambda_values = np.arange(0.05, 0.71, 0.05)
    num_slots = 100000
    simulation_results = []
    for lambda_val in lambda_values:
        avg_users, avg_time, avg_state = process_data(lambda_val, num_slots)
        simulation_results.append(avg_users)

    max_users_values = [10, 50, 100, 1000]
    avg_results = []
    for max_users in max_users_values:
        averages = average_users_across_lambdas(lambda_values, max_users)
        avg_results.append(averages)

    print(f"{'Макс. пользователей':^25}{'λ (Интенсивность)':^20}{'Среднее количество пользователей':^30}")
    print("-" * 77)
    for i, max_users in enumerate(max_users_values):
        for lambda_val, avg in zip(lambda_values, avg_results[i]):
            print(f"{max_users:^25}{lambda_val:^20.2f}{avg:^30.4f}")
        print()

    plot_results(lambda_values, avg_results, max_users_values, simulation_results)


if __name__ == "__main__":
    main()
