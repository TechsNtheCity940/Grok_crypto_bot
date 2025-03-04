import os
import logging
from models.hybrid_model import HybridCryptoModel
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerCryptoModel
from models.backtest import backtest_model
from utils.log_setup import logger
from config_manager import config

class ModelFactory:
    """
    Factory class for creating and managing different types of models.
    This provides a unified interface for model creation, training, and evaluation.
    """
    
    @staticmethod
    def create_model(model_type, sequence_length=None, n_features=None):
        """
        Create a model of the specified type
        
        Args:
            model_type: Type of model to create ('hybrid', 'lstm', 'transformer')
            sequence_length: Length of input sequences
            n_features: Number of features in input data
        
        Returns:
            Instantiated model object
        """
        # Use config values if not provided
        sequence_length = sequence_length or config.get('sequence_length', 50)
        n_features = n_features or config.get('n_features', 9)
        
        if model_type == 'hybrid':
            return HybridCryptoModel(sequence_length=sequence_length, n_features=n_features)
        elif model_type == 'lstm':
            return LSTMModel(sequence_length=sequence_length, n_features=n_features)
        elif model_type == 'transformer':
            return TransformerCryptoModel(sequence_length=sequence_length, n_features=n_features)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def train_model(model_type, symbol, df, model_dir='models/trained_models'):
        """
        Train a model of the specified type
        
        Args:
            model_type: Type of model to train
            symbol: Trading pair symbol (e.g., 'DOGE/USD')
            df: DataFrame with processed data
            model_dir: Directory to save trained model
        
        Returns:
            Tuple of (trained_model, accuracy)
        """
        # Create model path
        os.makedirs(model_dir, exist_ok=True)
        model_path = f'{model_dir}/{model_type}_{symbol.replace("/", "_")}.h5'
        
        # Create model
        model = ModelFactory.create_model(model_type)
        
        # Prepare training data
        X = []
        y_price = []
        feature_columns = ['momentum', 'rsi', 'macd', 'atr', 'sentiment', 
                          'arbitrage_spread', 'whale_activity', 'bb_upper', 'defi_apr']
        
        sequence_length = model.sequence_length
        
        for i in range(len(df) - sequence_length):
            X.append(df[feature_columns].iloc[i:i+sequence_length].values)
            price_change = (df['close'].iloc[i+sequence_length] - df['close'].iloc[i+sequence_length-1]) / df['close'].iloc[i+sequence_length-1]
            y_price.append(1 if price_change > 0.02 else 0)
        
        import numpy as np
        X = np.array(X)
        y_price = np.array(y_price)
        
        # Split data
        split_idx = int(len(X) * config.get('train_test_split', 0.8))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y_price[:split_idx], y_price[split_idx:]
        
        # Train model based on type
        if model_type == 'hybrid':
            if os.path.exists(model_path):
                model.model.load_weights(model_path)
                logger.info(f"Loaded existing hybrid model for {symbol} from {model_path}")
            else:
                history = model.train(X_train, y_train, np.zeros_like(y_train), 
                                     (X_val, y_val, np.zeros_like(y_val)), 
                                     model_path=model_path)
                logger.info(f"Trained new hybrid model for {symbol}")
        
        elif model_type == 'lstm':
            if os.path.exists(model_path):
                model.model.load_weights(model_path)
                logger.info(f"Loaded existing LSTM model for {symbol} from {model_path}")
            else:
                history = model.train(X_train, y_train, X_val, y_val, model_path=model_path)
                logger.info(f"Trained new LSTM model for {symbol}")
        
        elif model_type == 'transformer':
            if os.path.exists(model_path):
                model.load(model_path)
                logger.info(f"Loaded existing transformer model for {symbol} from {model_path}")
            else:
                history = model.train(X_train, y_train, X_val, y_val, model_path=model_path)
                logger.info(f"Trained new transformer model for {symbol}")
        
        # Backtest model
        accuracy = backtest_model(model, symbol, df)
        logger.info(f"{model_type.capitalize()} backtest accuracy for {symbol}: {accuracy:.2f}")
        
        return model, accuracy
    
    @staticmethod
    def load_model(model_type, symbol, model_dir='models/trained_models'):
        """
        Load a trained model
        
        Args:
            model_type: Type of model to load
            symbol: Trading pair symbol
            model_dir: Directory with trained models
        
        Returns:
            Loaded model object
        """
        model_path = f'{model_dir}/{model_type}_{symbol.replace("/", "_")}.h5'
        
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            return None
        
        model = ModelFactory.create_model(model_type)
        
        try:
            if model_type == 'transformer':
                model.load(model_path)
            else:
                model.model.load_weights(model_path)
            logger.info(f"Loaded {model_type} model for {symbol} from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    @staticmethod
    def ensemble_predict(models, X, weights=None):
        """
        Make predictions using an ensemble of models
        
        Args:
            models: Dict of {model_type: model_object}
            X: Input data
            weights: Dict of {model_type: weight} or None for equal weights
        
        Returns:
            Ensemble prediction
        """
        if not models:
            return 0.5  # Default neutral prediction
        
        # Use equal weights if not provided
        if weights is None:
            weights = {model_type: 1.0 / len(models) for model_type in models}
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Make predictions with each model
        predictions = {}
        for model_type, model in models.items():
            if model_type in weights:
                try:
                    if model_type == 'hybrid':
                        pred = model.predict(X)[0]  # First element of tuple
                    else:
                        pred = model.predict(X)
                    predictions[model_type] = pred
                except Exception as e:
                    logger.error(f"Error predicting with {model_type} model: {e}")
        
        if not predictions:
            return 0.5  # Default neutral prediction
        
        # Combine predictions with weights
        ensemble_pred = 0
        total_used_weight = 0
        
        for model_type, pred in predictions.items():
            if model_type in weights:
                ensemble_pred += pred.flatten() * weights[model_type]
                total_used_weight += weights[model_type]
        
        # Normalize by actually used weights
        if total_used_weight > 0:
            ensemble_pred = ensemble_pred / total_used_weight
        
        return ensemble_pred
