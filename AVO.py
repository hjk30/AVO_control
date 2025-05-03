import numpy as np

class HeatExchanger:
    def __init__(self, tau=83.0, K_heat=0.10, K_cool=0.0000055, K_env=0.0007):
        """
        Параметры:
            tau: float - постоянная времени (60 сек)
            K_heat: float - коэффициент нагрева от газа (T_in)
            K_cool: float - коэффициент охлаждения от вентиляции (Q)
            K_env: float - коэффициент влияния среды (T_env)
        """
        self.tau = tau
        self.K_heat = K_heat
        self.K_cool = K_cool
        self.K_env = K_env
        self.T_out = 35.0
    
    def update(self, Q, T_in, T_env, dt):
        # Нагрев от входящего газа (чем выше T_in, тем сильнее нагрев)
        heating = self.K_heat * (T_in - self.T_out)
        # Охлаждение вентиляцией (чем выше Q, тем сильнее охлаждение)
        cooling = self.K_cool * Q * (80 + T_env)
        # Влияние окружающей среды
        env_effect = self.K_env * (80 + T_env - self.T_out)

        dT_out = (heating - cooling + env_effect) / self.tau
        self.T_out += dT_out * dt
        return self.T_out
    
class AVOSystem:
    def __init__(self):
        # Параметры системы
        self.params = {
            'K_inv': 5.0,        # Коэф. инвертора [Гц/В]
            'T_motor': 1.0,      # Постоянная времени двигателя [с]
            'K_motor': 6.0,      # Коэф. двигателя [рад/с/Гц]
            'K_fan': 100.0,        # Коэф. вентилятора [м³/с/(рад/с)]
        }
        self.reset()
        self.heater = HeatExchanger()
    
    def reset(self):
        # Состояние системы
        self.state = {
            'omega': 0.0,        # Скорость двигателя [рад/с]
            'T_gas': 35.0,        # Текущая температура газа [°C]
        }
    
    def update(self, U, T_in, T_env, dt):
        """Обновление состояния системы за шаг dt"""
        for dtsec in np.arange(0, dt, 1/60):
            f = self.params['K_inv'] * U
            self.state['omega'] += (self.params['K_motor']*f - self.state['omega'])/self.params['T_motor'] * dtsec
            if self.state['omega'] < 0:
                self.state['omega'] = 0
            if self.state['omega'] > 150:
                self.state['omega'] = 150
            self.state['Q'] = self.params['K_fan'] * self.state['omega']
            self.state['T_gas'] = self.heater.update(self.state['Q'], T_in, T_env, dtsec)
        
        return self.state['T_gas']