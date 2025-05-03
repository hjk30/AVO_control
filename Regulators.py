import numpy as np
import keras
from sklearn.preprocessing import MinMaxScaler

class PIDController:
    def __init__(self, setpoint=40.0, Kp=5.50, Ki=0.10, Kd=0.50):
        self.setpoint = setpoint
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.reset()
    
    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
    
    def compute(self, T_measured, dt):
        error = T_measured - self.setpoint
        P = self.Kp * error
        self.integral += error * dt
        I = self.Ki * self.integral
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        D = self.Kd * derivative
        self.prev_error = error
        # Ограничение выхода (0-10 В)
        return np.clip(P + I + D, 2.0, 10.0)

class NeuroController:
    def __init__(self, setpoint, lookback=10):
        self.setpoint = setpoint
        self.lookback = lookback
        self.scaler = MinMaxScaler()
        self.neuro_model = self.build_model()
        self.history = []
    
    def build_model(self):
        neuro_model = keras.models.Sequential([
            keras.layers.Input(shape=(self.lookback, 4)),
            keras.layers.LSTM(32),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        neuro_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss='mse')
        return neuro_model
    
    def prepare_data(self, T_out, U, T_in, T_env):
        X, y = [], []
        for i in range(self.lookback, len(T_out)):
            features = []
            for j in range(i-self.lookback, i):
                error = self.setpoint[j] - T_out[j]
                features.append([T_out[j], error, T_in[j], T_env[j]])
            X.append(features)
            y.append(U[i])
        return np.array(X), np.array(y)
    
    def train(self, X_train, y_train, epochs=100, batch_size=24):
        # Нормализация данных
        X_train_reshaped = X_train.reshape(-1, 4)
        self.scaler.fit(X_train_reshaped)
        X_train_scaled = self.scaler.transform(X_train_reshaped).reshape(-1, self.lookback, 4)
        # Обучение модели
        history = self.neuro_model.fit(
            X_train_scaled, y_train/10.0,  # Нормализация выхода к [0,1]
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1,
        )
        return history
    
    def compute(self, T_out, T_in, T_env, ind):
        error = self.setpoint[ind] - T_out
        self.history.append([T_out, error, T_in, T_env])
        if len(self.history) > self.lookback:
            self.history = self.history[-self.lookback:]
        if len(self.history) < self.lookback:
            return 5.0
        X = np.array(self.history).reshape(1, self.lookback, 4)
        X_scaled = self.scaler.transform(X.reshape(-1, 4)).reshape(1, self.lookback, 4)
        U_norm = self.neuro_model.predict(X_scaled, verbose=0)[0][0] * 10.0
        # Ограничение выхода (0-10 В)
        return np.clip(U_norm, 2.0, 10.0)
