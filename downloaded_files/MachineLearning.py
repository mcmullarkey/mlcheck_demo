import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer
from sklearn.model_selection import train_test_split

data = np.load('data.npy')
data = data[:, 1:57, :]
label = np.load('label.npy')

X, X_test, y, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

input_shape = (56, 2001)
model = Sequential()
model.add(InputLayer(input_shape=input_shape))
model.add(LSTM(units=64, dropout=0.5, return_sequences=True))
model.add(LSTM(units=32, dropout=0.5, return_sequences=True))
model.add(Dense(64, kernel_regularizer='l2'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_val, y_val))
results = model.evaluate(X_test, y_test, batch_size=128)
print("test loss, test acc:", results)
model.save('RNN')