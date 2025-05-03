import numpy as np

def simulate_temperature(days=365, hours_per_day=24, daily_amplitude=10, yearly_amplitude=25,
                        base_temp=5, noise_level=2, seed=None):
    """Генерирует имитацию температуры в течение года (почасово)."""
    if seed is not None:
        np.random.seed(seed)

    day_indices = np.arange(days)
    hour_indices = np.arange(hours_per_day)

    # Годовая температура (синусоида)
    year_progress = 2 * np.pi * day_indices / days
    yearly_temp = yearly_amplitude * np.sin(year_progress - np.pi/2)

    # Суточная температура (синусоида)
    day_progress = 2 * np.pi * hour_indices / hours_per_day
    daily_temp_pattern = daily_amplitude * np.sin(day_progress - np.pi/2)


    temperature = np.zeros((days, hours_per_day))
    for day in range(days):
        temperature[day, :] = (base_temp + yearly_temp[day] + daily_temp_pattern)

    # Добавляем случайный шум
    temperature += np.random.normal(0, noise_level, (days, hours_per_day))

    return temperature.flatten()