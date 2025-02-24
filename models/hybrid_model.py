import logging
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam

class HybridCryptoModel:
    def __init__(self, sequence_length=50, n_features=8):  # Changed from 9 to 8
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = self._build_model()
        self.logger = logging.getLogger('hybrid_model')

    def _build_model(self):
        inputs = Input(shape=(self.sequence_length, self.n_features))
        conv = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
        lstm = LSTM(128, return_sequences=False)(conv)
        dense = Dense(256, activation='relu')(lstm)
        dense = Dropout(0.3)(dense)
        price_output = Dense(1, activation='sigmoid', name='price_output')(dense)
        model = Model(inputs=inputs, outputs=price_output)
        model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_train, y_price_train, y_volatility_train, validation_data, batch_size=32, epochs=50, model_path='model.h5'):
        X_val, y_price_val, _ = validation_data
        history = self.model.fit(
            X_train, y_price_train,
            validation_data=(X_val, y_price_val),
            batch_size=batch_size, epochs=epochs, verbose=1
        )
        self.model.save(model_path)
        return history

    def predict(self, X):
        return self.model.predict(X), np.zeros_like(self.model.predict(X))  # Dummy volatility