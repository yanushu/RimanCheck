import numpy as np
import smtplib
import ssl

from mpmath import zetazero
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def generate_data(n_samples):
    data = []
    labels = []
    for i in range(n_samples):
        is_on_critical_line = (i % 2 == 0)
        if is_on_critical_line:
            zero = zetazero(i // 2 + 1)
            labels.append(1)
        else:
            random_number = np.random.rand()
            random_imag = np.random.uniform(-30, 30)
            not_on_critical_line = complex(0.4 + 0.2 * random_number, random_imag)
            zero = not_on_critical_line
            labels.append(0)
        data.append([zero.real, zero.imag])
    return np.array(data), np.array(labels)

def create_model():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(2,)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


n_samples = 1000
data, labels = generate_data(n_samples)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = create_model()
model.fit(X_train_scaled, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=1)

evaluation = model.evaluate(X_test_scaled, y_test)
print(f"Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}")


def test_model(n_zeros):
    random_zeros = []
    for _ in range(n_zeros):
        random_number = np.random.rand()
        random_imag = np.random.uniform(-30, 30)
        random_zero = complex(0.4 + 0.2 * random_number, random_imag)
        random_zeros.append([random_zero.real, random_zero.imag])

    random_zeros_scaled = scaler.transform(random_zeros)
    predictions = model.predict(random_zeros_scaled)

    for zero, prediction in zip(random_zeros, predictions):
        if prediction[0] > 0.5:
            status = "в соответствии с гипотезой Римана"
        else:
            status = "НЕ в соответствии с гипотезой Римана"
        print(f"Для {zero}: {status}")


# Укажите количество случайных комплексных чисел, которые нейросеть будет проверять
n_zeros_to_check = 1000
test_model(n_zeros_to_check)
