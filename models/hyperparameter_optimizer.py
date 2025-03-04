import os
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
import optuna
from optuna.samplers import TPESampler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# Set up logging
logger = logging.getLogger('hyperparameter_optimizer')

class HyperparameterOptimizer:
    """
    Hyperparameter optimization for trading models and strategies
    using Optuna for efficient parameter search.
    """
    def __init__(self, model_factory=None, strategy_selector=None, results_dir='models/optimization_results'):
        self.model_factory = model_factory
        self.strategy_selector = strategy_selector
        self.results_dir = results_dir
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Set up Optuna study storage
        self.storage = None
        
        # Default optimization parameters
        self.n_trials = 50
        self.timeout = 3600  # 1 hour
        self.n_jobs = 1
    
    def optimize_lstm_model(self, X, y, symbol, n_trials=None, timeout=None):
        """
        Optimize hyperparameters for LSTM model
        
        Args:
            X: Input features
            y: Target values
            symbol: Trading pair symbol
            n_trials: Number of trials (default: self.n_trials)
            timeout: Timeout in seconds (default: self.timeout)
        
        Returns:
            dict: Best hyperparameters
        """
        n_trials = n_trials or self.n_trials
        timeout = timeout or self.timeout
        
        # Define the objective function
        def objective(trial):
            # Define hyperparameters to optimize
            params = {
                'lstm_units': trial.suggest_int('lstm_units', 32, 256),
                'dense_units': trial.suggest_int('dense_units', 16, 128),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                'epochs': 50  # Fixed number of epochs with early stopping
            }
            
            # Create time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Track scores across folds
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Build model with current hyperparameters
                model = self._build_lstm_model(
                    input_shape=(X.shape[1], X.shape[2]),
                    lstm_units=params['lstm_units'],
                    dense_units=params['dense_units'],
                    dropout_rate=params['dropout_rate'],
                    learning_rate=params['learning_rate']
                )
                
                # Train model
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
                
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                # Evaluate model
                val_loss = min(history.history['val_loss'])
                scores.append(val_loss)
            
            # Return mean validation loss
            return np.mean(scores)
        
        # Create Optuna study
        study_name = f"lstm_optimization_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            study_name=study_name,
            direction='minimize',
            sampler=TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=self.n_jobs)
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        # Log results
        logger.info(f"Best LSTM parameters for {symbol}: {best_params} with loss {best_value:.4f}")
        
        # Save results
        self._save_optimization_results(symbol, 'lstm', best_params, best_value, study)
        
        return best_params
    
    def optimize_transformer_model(self, X, y, symbol, n_trials=None, timeout=None):
        """
        Optimize hyperparameters for Transformer model
        
        Args:
            X: Input features
            y: Target values
            symbol: Trading pair symbol
            n_trials: Number of trials (default: self.n_trials)
            timeout: Timeout in seconds (default: self.timeout)
        
        Returns:
            dict: Best hyperparameters
        """
        n_trials = n_trials or self.n_trials
        timeout = timeout or self.timeout
        
        # Define the objective function
        def objective(trial):
            # Define hyperparameters to optimize
            params = {
                'num_heads': trial.suggest_int('num_heads', 2, 8),
                'key_dim': trial.suggest_int('key_dim', 16, 64),
                'ff_dim': trial.suggest_int('ff_dim', 64, 256),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                'epochs': 50  # Fixed number of epochs with early stopping
            }
            
            # Create time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Track scores across folds
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Build model with current hyperparameters
                model = self._build_transformer_model(
                    input_shape=(X.shape[1], X.shape[2]),
                    num_heads=params['num_heads'],
                    key_dim=params['key_dim'],
                    ff_dim=params['ff_dim'],
                    dropout_rate=params['dropout_rate'],
                    learning_rate=params['learning_rate']
                )
                
                # Train model
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
                
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                # Evaluate model
                val_loss = min(history.history['val_loss'])
                scores.append(val_loss)
            
            # Return mean validation loss
            return np.mean(scores)
        
        # Create Optuna study
        study_name = f"transformer_optimization_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            study_name=study_name,
            direction='minimize',
            sampler=TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=self.n_jobs)
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        # Log results
        logger.info(f"Best Transformer parameters for {symbol}: {best_params} with loss {best_value:.4f}")
        
        # Save results
        self._save_optimization_results(symbol, 'transformer', best_params, best_value, study)
        
        return best_params
    
    def optimize_grid_trading_strategy(self, df, symbol, n_trials=None, timeout=None):
        """
        Optimize hyperparameters for Grid Trading strategy
        
        Args:
            df: DataFrame with price data
            symbol: Trading pair symbol
            n_trials: Number of trials (default: self.n_trials)
            timeout: Timeout in seconds (default: self.timeout)
        
        Returns:
            dict: Best hyperparameters
        """
        n_trials = n_trials or self.n_trials
        timeout = timeout or self.timeout
        
        # Define the objective function
        def objective(trial):
            # Define hyperparameters to optimize
            params = {
                'num_grids': trial.suggest_int('num_grids', 5, 20),
                'grid_spread': trial.suggest_float('grid_spread', 0.01, 0.1),
                'max_position': trial.suggest_float('max_position', 0.1, 1.0),
                'min_profit': trial.suggest_float('min_profit', 0.001, 0.05)
            }
            
            # Simulate grid trading with these parameters
            from strategies.grid_trading import GridTrader
            
            # Create grid trader with current parameters
            grid_config = {'grid_trading': params}
            grid_trader = GridTrader(grid_config)
            
            # Initialize simulation
            initial_price = df['close'].iloc[0]
            grid_trader.setup_grids(initial_price, price_range=params['grid_spread'] * 10)
            
            # Simulate trading
            balance_usd = 1000.0
            balance_asset = 0.0
            trades = []
            
            for i in range(1, len(df)):
                price = df['close'].iloc[i]
                
                # Get grid orders
                grid_orders = grid_trader.get_grid_orders(price, balance_usd + balance_asset * price)
                
                # Execute orders
                for order in grid_orders:
                    if order['type'] == 'buy' and balance_usd >= order['amount'] * price:
                        balance_usd -= order['amount'] * price
                        balance_asset += order['amount']
                        trades.append({
                            'type': 'buy',
                            'price': price,
                            'amount': order['amount']
                        })
                    elif order['type'] == 'sell' and balance_asset >= order['amount']:
                        balance_usd += order['amount'] * price
                        balance_asset -= order['amount']
                        trades.append({
                            'type': 'sell',
                            'price': price,
                            'amount': order['amount']
                        })
                
                # Update grids
                grid_trader.update_grids_by_price(price)
            
            # Calculate final portfolio value
            final_value = balance_usd + balance_asset * df['close'].iloc[-1]
            initial_value = 1000.0
            
            # Calculate return
            total_return = (final_value - initial_value) / initial_value
            
            # Calculate Sharpe ratio (assuming risk-free rate of 0)
            if len(trades) > 1:
                trade_returns = []
                for i in range(1, len(trades)):
                    if trades[i]['type'] != trades[i-1]['type']:
                        if trades[i]['type'] == 'sell':
                            # Buy then sell
                            trade_return = (trades[i]['price'] - trades[i-1]['price']) / trades[i-1]['price']
                        else:
                            # Sell then buy
                            trade_return = (trades[i-1]['price'] - trades[i]['price']) / trades[i]['price']
                        trade_returns.append(trade_return)
                
                if trade_returns:
                    avg_return = np.mean(trade_returns)
                    std_return = np.std(trade_returns)
                    sharpe_ratio = avg_return / std_return if std_return > 0 else 0
                else:
                    sharpe_ratio = 0
            else:
                sharpe_ratio = 0
            
            # Objective: maximize Sharpe ratio
            return -sharpe_ratio  # Negative because Optuna minimizes
        
        # Create Optuna study
        study_name = f"grid_trading_optimization_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            study_name=study_name,
            direction='minimize',
            sampler=TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=self.n_jobs)
        
        # Get best parameters
        best_params = study.best_params
        best_value = -study.best_value  # Convert back to positive Sharpe ratio
        
        # Log results
        logger.info(f"Best Grid Trading parameters for {symbol}: {best_params} with Sharpe ratio {best_value:.4f}")
        
        # Save results
        self._save_optimization_results(symbol, 'grid_trading', best_params, best_value, study)
        
        return best_params
    
    def optimize_mean_reversion_strategy(self, df, symbol, n_trials=None, timeout=None):
        """
        Optimize hyperparameters for Mean Reversion strategy
        
        Args:
            df: DataFrame with price data
            symbol: Trading pair symbol
            n_trials: Number of trials (default: self.n_trials)
            timeout: Timeout in seconds (default: self.timeout)
        
        Returns:
            dict: Best hyperparameters
        """
        n_trials = n_trials or self.n_trials
        timeout = timeout or self.timeout
        
        # Define the objective function
        def objective(trial):
            # Define hyperparameters to optimize
            params = {
                'rsi_period': trial.suggest_int('rsi_period', 7, 21),
                'rsi_overbought': trial.suggest_int('rsi_overbought', 65, 85),
                'rsi_oversold': trial.suggest_int('rsi_oversold', 15, 35),
                'bollinger_period': trial.suggest_int('bollinger_period', 10, 30),
                'bollinger_std': trial.suggest_float('bollinger_std', 1.5, 3.0),
                'position_size': trial.suggest_float('position_size', 0.1, 0.5)
            }
            
            # Simulate mean reversion trading with these parameters
            from strategies.mean_reversion import MeanReversionStrategy
            
            # Create strategy with current parameters
            strategy = MeanReversionStrategy(params)
            
            # Calculate indicators
            df_indicators = strategy.calculate_indicators(df)
            
            # Simulate trading
            balance_usd = 1000.0
            balance_asset = 0.0
            trades = []
            
            for i in range(strategy.bollinger_period, len(df_indicators)):
                # Get signal
                signal = strategy.get_signal(df_indicators.iloc[:i+1])
                price = df_indicators.iloc[i]['close']
                
                # Execute trade
                if signal == 1 and balance_usd > 0:  # Buy
                    amount = strategy.calculate_position_size(signal, price, balance_usd, balance_asset)
                    if amount * price <= balance_usd:
                        balance_usd -= amount * price
                        balance_asset += amount
                        trades.append({
                            'action': 'buy',
                            'price': price,
                            'amount': amount,
                            'timestamp': df_indicators.iloc[i]['timestamp'] if 'timestamp' in df_indicators.columns else i
                        })
                elif signal == 2 and balance_asset > 0:  # Sell
                    amount = strategy.calculate_position_size(signal, price, balance_usd, balance_asset)
                    if amount <= balance_asset:
                        balance_usd += amount * price
                        balance_asset -= amount
                        trades.append({
                            'action': 'sell',
                            'price': price,
                            'amount': amount,
                            'timestamp': df_indicators.iloc[i]['timestamp'] if 'timestamp' in df_indicators.columns else i
                        })
            
            # Calculate final portfolio value
            final_value = balance_usd + balance_asset * df_indicators.iloc[-1]['close']
            initial_value = 1000.0
            
            # Calculate return
            total_return = (final_value - initial_value) / initial_value
            
            # Calculate Sharpe ratio
            if len(trades) > 1:
                trade_returns = []
                for i in range(1, len(trades)):
                    if trades[i]['action'] != trades[i-1]['action']:
                        if trades[i]['action'] == 'sell':
                            # Buy then sell
                            trade_return = (trades[i]['price'] - trades[i-1]['price']) / trades[i-1]['price']
                        else:
                            # Sell then buy
                            trade_return = (trades[i-1]['price'] - trades[i]['price']) / trades[i]['price']
                        trade_returns.append(trade_return)
                
                if trade_returns:
                    avg_return = np.mean(trade_returns)
                    std_return = np.std(trade_returns)
                    sharpe_ratio = avg_return / std_return if std_return > 0 else 0
                else:
                    sharpe_ratio = 0
            else:
                sharpe_ratio = 0
            
            # Objective: maximize Sharpe ratio
            return -sharpe_ratio  # Negative because Optuna minimizes
        
        # Create Optuna study
        study_name = f"mean_reversion_optimization_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            study_name=study_name,
            direction='minimize',
            sampler=TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=self.n_jobs)
        
        # Get best parameters
        best_params = study.best_params
        best_value = -study.best_value  # Convert back to positive Sharpe ratio
        
        # Log results
        logger.info(f"Best Mean Reversion parameters for {symbol}: {best_params} with Sharpe ratio {best_value:.4f}")
        
        # Save results
        self._save_optimization_results(symbol, 'mean_reversion', best_params, best_value, study)
        
        return best_params
    
    def optimize_breakout_strategy(self, df, symbol, n_trials=None, timeout=None):
        """
        Optimize hyperparameters for Breakout strategy
        
        Args:
            df: DataFrame with price data
            symbol: Trading pair symbol
            n_trials: Number of trials (default: self.n_trials)
            timeout: Timeout in seconds (default: self.timeout)
        
        Returns:
            dict: Best hyperparameters
        """
        n_trials = n_trials or self.n_trials
        timeout = timeout or self.timeout
        
        # Define the objective function
        def objective(trial):
            # Define hyperparameters to optimize
            params = {
                'lookback_period': trial.suggest_int('lookback_period', 10, 30),
                'volume_factor': trial.suggest_float('volume_factor', 1.0, 2.5),
                'atr_period': trial.suggest_int('atr_period', 7, 21),
                'atr_multiplier': trial.suggest_float('atr_multiplier', 0.5, 2.0),
                'min_touches': trial.suggest_int('min_touches', 1, 3),
                'max_risk_per_trade': trial.suggest_float('max_risk_per_trade', 0.01, 0.05)
            }
            
            # Simulate breakout trading with these parameters
            from strategies.breakout import BreakoutStrategy
            
            # Create strategy with current parameters
            strategy = BreakoutStrategy(params)
            
            # Calculate indicators
            df_indicators = strategy.calculate_indicators(df)
            
            # Simulate trading
            balance_usd = 1000.0
            balance_asset = 0.0
            trades = []
            
            for i in range(strategy.lookback_period, len(df_indicators)):
                # Get signal
                signal, stop_loss, take_profit = strategy.get_signal(df_indicators.iloc[:i+1])
                price = df_indicators.iloc[i]['close']
                
                # Execute trade
                if signal == 1 and balance_usd > 0:  # Buy
                    amount = strategy.calculate_position_size(signal, price, stop_loss, balance_usd, balance_asset)
                    if amount * price <= balance_usd:
                        balance_usd -= amount * price
                        balance_asset += amount
                        trades.append({
                            'action': 'buy',
                            'price': price,
                            'amount': amount,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'timestamp': df_indicators.iloc[i]['timestamp'] if 'timestamp' in df_indicators.columns else i
                        })
                elif signal == 2 and balance_asset > 0:  # Sell
                    amount = strategy.calculate_position_size(signal, price, stop_loss, balance_usd, balance_asset)
                    if amount <= balance_asset:
                        balance_usd += amount * price
                        balance_asset -= amount
                        trades.append({
                            'action': 'sell',
                            'price': price,
                            'amount': amount,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'timestamp': df_indicators.iloc[i]['timestamp'] if 'timestamp' in df_indicators.columns else i
                        })
                
                # Check for stop loss and take profit
                if trades and trades[-1]['action'] == 'buy':
                    if price <= trades[-1]['stop_loss']:
                        # Stop loss hit
                        balance_usd += balance_asset * price
                        balance_asset = 0
                        trades.append({
                            'action': 'stop_loss',
                            'price': price,
                            'amount': trades[-1]['amount'],
                            'timestamp': df_indicators.iloc[i]['timestamp'] if 'timestamp' in df_indicators.columns else i
                        })
                    elif price >= trades[-1]['take_profit']:
                        # Take profit hit
                        balance_usd += balance_asset * price
                        balance_asset = 0
                        trades.append({
                            'action': 'take_profit',
                            'price': price,
                            'amount': trades[-1]['amount'],
                            'timestamp': df_indicators.iloc[i]['timestamp'] if 'timestamp' in df_indicators.columns else i
                        })
                elif trades and trades[-1]['action'] == 'sell':
                    if price >= trades[-1]['stop_loss']:
                        # Stop loss hit
                        balance_asset += balance_usd / price
                        balance_usd = 0
                        trades.append({
                            'action': 'stop_loss',
                            'price': price,
                            'amount': trades[-1]['amount'],
                            'timestamp': df_indicators.iloc[i]['timestamp'] if 'timestamp' in df_indicators.columns else i
                        })
                    elif price <= trades[-1]['take_profit']:
                        # Take profit hit
                        balance_asset += balance_usd / price
                        balance_usd = 0
                        trades.append({
                            'action': 'take_profit',
                            'price': price,
                            'amount': trades[-1]['amount'],
                            'timestamp': df_indicators.iloc[i]['timestamp'] if 'timestamp' in df_indicators.columns else i
                        })
            
            # Calculate final portfolio value
            final_value = balance_usd + balance_asset * df_indicators.iloc[-1]['close']
            initial_value = 1000.0
            
            # Calculate return
            total_return = (final_value - initial_value) / initial_value
            
            # Calculate Sharpe ratio
            if len(trades) > 1:
                trade_returns = []
                for i in range(1, len(trades)):
                    if trades[i]['action'] in ['sell', 'take_profit'] and trades[i-1]['action'] == 'buy':
                        # Buy then sell
                        trade_return = (trades[i]['price'] - trades[i-1]['price']) / trades[i-1]['price']
                        trade_returns.append(trade_return)
                    elif trades[i]['action'] in ['buy', 'take_profit'] and trades[i-1]['action'] == 'sell':
                        # Sell then buy
                        trade_return = (trades[i-1]['price'] - trades[i]['price']) / trades[i]['price']
                        trade_returns.append(trade_return)
                
                if trade_returns:
                    avg_return = np.mean(trade_returns)
                    std_return = np.std(trade_returns)
                    sharpe_ratio = avg_return / std_return if std_return > 0 else 0
                else:
                    sharpe_ratio = 0
            else:
                sharpe_ratio = 0
            
            # Objective: maximize Sharpe ratio
            return -sharpe_ratio  # Negative because Optuna minimizes
        
        # Create Optuna study
        study_name = f"breakout_optimization_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            study_name=study_name,
            direction='minimize',
            sampler=TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=self.n_jobs)
        
        # Get best parameters
        best_params = study.best_params
        best_value = -study.best_value  # Convert back to positive Sharpe ratio
        
        # Log results
        logger.info(f"Best Breakout parameters for {symbol}: {best_params} with Sharpe ratio {best_value:.4f}")
        
        # Save results
        self._save_optimization_results(symbol, 'breakout', best_params, best_value, study)
        
        return best_params
    
    def _build_lstm_model(self, input_shape, lstm_units, dense_units, dropout_rate, learning_rate):
        """Build LSTM model with given hyperparameters"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.LSTM(lstm_units // 2),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(dense_units, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_transformer_model(self, input_shape, num_heads, key_dim, ff_dim, dropout_rate, learning_rate):
        """Build Transformer model with given hyperparameters"""
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # Transformer Encoder Block 1
        attention_output1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim
        )(inputs, inputs)
        attention_output1 = tf.keras.layers.Dropout(dropout_rate)(attention_output1)
        attention_output1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output1)
        
        # Feed-forward network
        ffn_output1 = tf.keras.layers.Dense(ff_dim, activation='relu')(attention_output1)
        ffn_output1 = tf.keras.layers.Dense(input_shape[-1])(ffn_output1)
        ffn_output1 = tf.keras.layers.Dropout(dropout_rate)(ffn_output1)
        encoder_output1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_output1 + ffn_output1)
        
        # Global pooling
        pooled = tf.keras.layers.GlobalAveragePooling1D()(encoder_output1)
        
        # Final prediction layers
        x = tf.keras.layers.Dense(ff_dim // 2, activation='relu')(pooled)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _save_optimization_results(self, symbol, model_type, best_params, best_value, study):
        """Save optimization results to file"""
        # Create results directory for this symbol if it doesn't exist
        symbol_dir = os.path.join(self.results_dir, symbol.replace('/', '_'))
        os.makedirs(symbol_dir, exist_ok=True)
        
        # Save best parameters
        results = {
            'symbol': symbol,
            'model_type': model_type,
            'best_params': best_params,
            'best_value': best_value,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to JSON file
        results_file = os.path.join(symbol_dir, f"{model_type}_optimization_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save study trials
        trials_df = study.trials_dataframe()
        trials_file = os.path.join(symbol_dir, f"{model_type}_optimization_trials.csv")
        trials_df.to_csv(trials_file, index=False)
        
        logger.info(f"Saved optimization results to {results_file}")
    
    def load_best_parameters(self, symbol, model_type):
        """
        Load best parameters from previous optimization
        
        Args:
            symbol: Trading pair symbol
            model_type: Model type ('lstm', 'transformer', 'grid_trading', etc.)
        
        Returns:
            dict: Best parameters or None if not found
        """
        symbol_dir = os.path.join(self.results_dir, symbol.replace('/', '_'))
        results_file = os.path.join(symbol_dir, f"{model_type}_optimization_results.json")
        
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                logger.info(f"Loaded optimization results for {symbol} {model_type} from {results_file}")
                return results['best_params']
            except Exception as e:
                logger.error(f"Error loading optimization results: {e}")
                return None
        else:
            logger.warning(f"No optimization results found for {symbol} {model_type}")
            return None
