import numpy as np
import matplotlib.pyplot as plt
from AVO import AVOSystem
from Regulators import PIDController, NeuroController
from Temperature import simulate_temperature
import sklearn.metrics
import time

def simulate(system: AVOSystem, controller, t, T_in, T_env):
    T_out = np.zeros_like(t)
    T_out[0] = system.state["T_gas"]
    U = np.zeros_like(t)
    third = len(t) // 3
    setpoints = [35.0] * third + [40.0] * third + [35.0] * (len(t) - 2*third)
    if isinstance(controller, NeuroController):
        controller.setpoint = setpoints
    
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]

        if isinstance(controller, NeuroController):
            U[i] = controller.compute(T_out[i-1], T_in[i], T_env[i], i)
        else:
            controller.setpoint = setpoints[i]
            U[i] = controller.compute(T_out[i-1], dt)

        T_out[i] = system.update(U[i], T_in[i], T_env[i], dt)
    
    return T_out, U, setpoints

# Создаем тренировочные данные с ПИД-регулятором
def generate_training_data():
    system = AVOSystem()
    pid = PIDController(30, 7.0, 0.000002, 0.00000052)
    T_env = simulate_temperature(days=20, hours_per_day=24, noise_level=0)
    t_train = np.linspace(0, len(T_env), len(T_env))
    T_in = 80 + 1*np.random.normal(0, 1, len(T_env))
    T_out_train, U_train, setpoints_train = simulate(system, pid, t_train, T_in, T_env)
    return T_out_train, U_train, T_in, T_env, setpoints_train

if __name__ == "__main__":
    system_pid = AVOSystem()
    system_nn = AVOSystem()
    pid = PIDController(30, 7.0, 0.000002, 0.00000052)

    T_env_test = simulate_temperature(days=20, hours_per_day=24)
    t_test = np.linspace(0, len(T_env_test), len(T_env_test))
    T_in_test = 80 + 2*np.random.normal(0, 1, len(t_test))
    
    third = len(t_test) // 3
    setpointsLow = [34.0] * third + [39.0] * third + [34.0] * (len(t_test) - 2*third)
    setpointsHigh = [36.0] * third + [41.0] * third + [36.0] * (len(t_test) - 2*third)

    T_out_pid, U_pid, setpoints = simulate(system_pid, pid, t_test, T_in_test, T_env_test)
    
    print("Запуск обучения...")

    # Создаем и обучаем нейрорегулятор
    T_out_train, U_train, T_in_train, T_env_train, setpoints_train = generate_training_data()
    neuro = NeuroController(setpoints_train)
    X_train, y_train = neuro.prepare_data(T_out_train, U_train, T_in_train, T_env_train)
    nn_train_start_time = time.time()
    train_history = neuro.train(X_train, y_train, epochs=100, batch_size=24)
    nn_train_end_time = time.time()
    nn_training_time = nn_train_end_time - nn_train_start_time
    print(f"Время обучения нейросети: {nn_training_time} секунд")

    #plt.figure(figsize=(10, 5))
    #plt.plot(train_history.history['loss'], label='Обучение')
    #plt.plot(train_history.history['val_loss'], label='Валидация')
    #plt.title('История обучения')
    #plt.ylabel('Ошибка (MSE)')
    #plt.xlabel('Эпоха')
    #plt.legend()
    #plt.grid(True)
    #plt.show()

    print("Запуск моделирования...")
    
    nn_start_time = time.time()
    T_out_nn, U_nn, _ = simulate(system_nn, neuro, t_test, T_in_test, T_env_test)
    nn_end_time = time.time()
    nn_simulation_time = nn_end_time - nn_start_time
    print(f"Время моделирования нейросети: {nn_simulation_time} секунд")

    mse_pid = sklearn.metrics.mean_squared_error(T_out_pid, setpoints)
    mse_nn = sklearn.metrics.mean_squared_error(T_out_nn, setpoints)

    print(f"MSE (PID Controller): {mse_pid}")
    print(f"MSE (Neuro Controller): {mse_nn}")
    # Создаем графики
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(t_test, T_out_pid, label='При ПИД-регулироании')
    plt.plot(t_test, T_out_nn, label='При нейросетевом регулировании')
    plt.plot(t_test, setpointsLow, 'k--', label='Уставка')
    plt.plot(t_test, setpointsHigh, 'k--')
    plt.ylabel('Температура (°C)\nгаза на выходе')
    plt.xlabel('Время (ч)')
    plt.legend()
    plt.grid(True)
    plt.title('Сравнение переходных процессов с изменяющейся уставкой')

    plt.subplot(2, 1, 2)
    plt.plot(t_test, T_in_test, label='T_in')
    plt.plot(t_test, T_env_test, label='T_env')
    plt.ylabel('Температура (°C)')
    plt.xlabel('Время (ч)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.plot(t_test, U_pid, label='Управляющее воздействие ПИД-регулятора', linewidth=1)
    plt.plot(t_test, U_nn, label='Управляющее воздействие нейрорегулятора', linewidth=1)
    plt.ylabel('U (В)')
    plt.xlabel('Время (ч)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()