import numpy as np
import matplotlib.pyplot as plt


class GraphicMaker:
    def __init__(self):
        self.exp_minus_1 = np.exp(-1)
        self.lambda_values: list[float] = [x / 100 for x in range(5, 55, 5)]

    def show_graphic(self, plot_y: list[float], x_label: str, y_label: str, title: str, png_name: str) -> None:
        plt.figure()
        plt.plot(lambda_values, plot_y, marker='o')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.axvline(x=self.exp_minus_1, color='r', linestyle='--')
        plt.grid(True)
        plt.savefig(f"Графики/{png_name}.png", format="png", dpi=300)
        plt.show()


# Создание графикмэйкера
GraphicMaker: GraphicMaker = GraphicMaker()
# Значения лямбда с шагом 0.05 сгенерированные в графикмэйкере
lambda_values: list[float] = GraphicMaker.lambda_values
S = 1000000  # Количество слотов для моделирования

# Результаты для графиков
average_N = []  # Среднее количество абонентов в системе
average_T = []  # Среднее время нахождения абонента в системе
G_values = []  # Пропускная способность канала

# Симуляция для каждого значения λ
for lam in lambda_values:
    N = 0  # Текущее количество абонентов в системе
    total_messages = 0  # Общее количество отправленных сообщений
    total_time = 0  # Общее время нахождения абонентов в системе

    for slot in range(S):
        # Определяем количество абонентов, у которых есть сообщения по распределению Пуассона
        new_messages = np.random.poisson(lam)

        N += new_messages  # Добавляем новых абонентов в систему

        if N > 0:
            # Вероятность передачи для каждого абонента
            p = 1 / N

            # Количество абонентов, пытающихся передать сообщения (биномиальное распределение)
            transmitting = np.random.binomial(N, p)

            if transmitting == 1:
                # Событие "успех"
                total_messages += 1

                total_time += N

                N -= 1  # Абонент покидает систему после успешной передачи

            elif transmitting > 1:
                # Событие "конфликт"
                total_time += N

            else:
                # Событие "пусто"
                total_time += N

        else:
            # В системе нет абонентов, просто переходим к следующему слоту
            pass

    # Среднее количество абонентов
    average_N.append(total_time / S)
    # Среднее время нахождения абонента в системе
    average_T.append(total_time / total_messages if total_messages > 0 else 0)
    # Пропускная способность канала
    G_values.append(total_messages / S)

# Построение отдельных графиков
# График 1: Среднее количество абонентов в системе от λ
GraphicMaker.show_graphic([x for x in average_N],
                          # Нормируем график, для этого делим каждое значение на графике на среднее арифметическое
                          "λ (интенсивность потока сообщений)",
                          "Среднее количество абонентов в системе N̂",
                          f"Cреднее количество абонентов от λ (точность {S})",
                          "Количество")

# График 2: Среднее время нахождения абонента в системе от λ
GraphicMaker.show_graphic([x / S for x in average_T],
                          # Нормируем график, для этого делим каждое значение на графике на количество слотов
                          "λ (интенсивность потока сообщений)",
                          "Среднее время нахождения абонента T̂",
                          f"Среднее время абонента в сети от λ (точность {S})",
                          "Время")

# График 3: Средняя пропускная способность канала G от λ
GraphicMaker.show_graphic(G_values,
                          "λ (интенсивность потока сообщений)",
                          "Пропускная способность канала G",
                          f"Пропускная способность канала от λ (точность {S})",
                          "Пропускная способность")
