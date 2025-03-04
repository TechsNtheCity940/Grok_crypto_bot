import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, 
    MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
import logging

class TransformerCryptoModel:
    """
    Transformer-based model for cryptocurrency price prediction.
    Uses attention mechanisms to better capture temporal dependencies
    and market patterns.
    """
    def __init__(self, sequence_length=50, n_features=9):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = self._build_model()
        self.logger = logging.getLogger('transformer_model')
    
    def to(self, device):
        # Dummy method to handle PyTorch's .to(device) calls
        # TensorFlow models don't use this method, but we add it for compatibility
        return self
    
    def _build_model(self):
        """
        Build a Transformer model with multi-head attention
        """
        # Input shape: [batch_size, sequence_length, n_features]
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        # Transformer Encoder Block 1
        attention_output1 = MultiHeadAttention(
            num_heads=4, key_dim=32
        )(inputs, inputs)
        attention_output1 = Dropout(0.1)(attention_output1)
        attention_output1 = LayerNormalization(epsilon=1e-6)(inputs + attention_output1)
        
        # Feed-forward network
        ffn_output1 = Dense(128, activation='relu')(attention_output1)
        ffn_output1 = Dense(self.n_features)(ffn_output1)
        ffn_output1 = Dropout(0.1)(ffn_output1)
        encoder_output1 = LayerNormalization(epsilon=1e-6)(attention_output1 + ffn_output1)
        
        # Transformer Encoder Block 2
        attention_output2 = MultiHeadAttention(
            num_heads=4, key_dim=32
        )(encoder_output1, encoder_output1)
        attention_output2 = Dropout(0.1)(attention_output2)
        attention_output2 = LayerNormalization(epsilon=1e-6)(encoder_output1 + attention_output2)
        
        # Feed-forward network
        ffn_output2 = Dense(128, activation='relu')(attention_output2)
        ffn_output2 = Dense(self.n_features)(ffn_output2)
        ffn_output2 = Dropout(0.1)(ffn_output2)
        encoder_output2 = LayerNormalization(epsilon=1e-6)(attention_output2 + ffn_output2)
        
        # Global pooling
        pooled = GlobalAveragePooling1D()(encoder_output2)
        
        # Final prediction layers
        x = Dense(64, activation='relu')(pooled)
        x = Dropout(0.2)(x)
        price_output = Dense(1, activation='sigmoid', name='price_output')(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=price_output)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=50, model_path='models/transformer.h5', save_weights_only=True):
        """
        Train the transformer model
        """
        # Learning rate scheduler
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001
        )
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Model checkpoint
        checkpoint_path = model_path
        if save_weights_only and not model_path.endswith('.weights.h5'):
            checkpoint_path = model_path.replace('.h5', '.weights.h5')
            
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True
        )
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[lr_scheduler, early_stopping, model_checkpoint],
            verbose=1
        )
        
        # Load the best weights
        self.model.load_weights(checkpoint_path)
        
        return history
    
    def predict(self, X):
        """
        Make predictions with the transformer model
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data
        """
        return self.model.evaluate(X_test, y_test)
    
    def save(self, model_path, save_weights_only=True):
        """
        Save the model weights
        """
        # Adjust path for weights-only files if needed
        if save_weights_only and not model_path.endswith('.weights.h5'):
            model_path = model_path.replace('.h5', '.weights.h5')
            
        self.model.save_weights(model_path)
        
    def load(self, model_path):
        """
        Load the model weights
        """
        # Check if we need to adjust the path for weights-only files
        if not os.path.exists(model_path) and not model_path.endswith('.weights.h5'):
            weights_path = model_path.replace('.h5', '.weights.h5')
            if os.path.exists(weights_path):
                model_path = weights_path
        
        self.model.load_weights(model_path)
