import logging
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class LSTMModel:
    def __init__(self, sequence_length=50, n_features=9):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = self._build_model()
        self.logger = logging.getLogger('lstm_model')

    def _build_model(self):
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.2),
            LSTM(100, return_sequences=False),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=50, model_path='models/lstm_doge.h5'):
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size, epochs=epochs, verbose=1
        )
        self.model.save(model_path)
        return history

    def predict(self, X):
        return self.model.predict(X)