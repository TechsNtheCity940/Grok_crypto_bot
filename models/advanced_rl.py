import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
import gym
from gym import spaces
import logging
import random
from collections import deque, namedtuple
import time
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# Set up logging
logger = logging.getLogger('advanced_rl')

class TradingEnv(gym.Env):
    """
    Custom trading environment that follows gym interface.
    This is a more advanced version that supports various RL algorithms.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, symbol, initial_balance=10000, transaction_fee=0.001, window_size=20, reward_scaling=1.0):
        super(TradingEnv, self).__init__()
        
        # Save inputs
        self.df = df.copy()
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.window_size = window_size
        self.reward_scaling = reward_scaling
        
        # Trading state
        self.balance = initial_balance
        self.position = 0
        self.current_step = window_size
        self.done = False
        self.portfolio_value_history = [initial_balance]
        
        # Calculate features
        self._calculate_features()
        
        # Define action and observation space
        # Actions: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: OHLCV data + technical indicators + account info
        self.feature_dim = self.features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.window_size, self.feature_dim + 2),  # +2 for balance and position
            dtype=np.float32
        )
    
    def _calculate_features(self):
        """Calculate technical indicators and other features"""
        df = self.df.copy()
        
        # Ensure we have OHLCV data
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                if col == 'volume':
                    df[col] = 0
                else:
                    df[col] = df['close']
        
        # Normalize price data
        price_cols = ['open', 'high', 'low', 'close']
        df[price_cols] = df[price_cols].div(df['close'].rolling(window=self.window_size).mean(), axis=0)
        
        # Normalize volume
        if 'volume' in df.columns and df['volume'].sum() > 0:
            df['volume'] = df['volume'] / df['volume'].rolling(window=self.window_size).mean()
        
        # Add basic indicators
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        df['macd'], df['macd_signal'] = self._calculate_macd(df['close'])
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
        df['atr'] = self._calculate_atr(df[['high', 'low', 'close']], 14)
        
        # Add momentum indicators
        df['mom'] = df['close'].pct_change(periods=10)
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()
        
        # Add trend indicators
        df['ma_fast'] = df['close'].rolling(window=10).mean() / df['close']
        df['ma_slow'] = df['close'].rolling(window=30).mean() / df['close']
        
        # Add sentiment if available
        if 'sentiment' in df.columns:
            df['sentiment'] = (df['sentiment'] - df['sentiment'].mean()) / df['sentiment'].std()
        else:
            df['sentiment'] = 0
        
        # Add on-chain metrics if available
        onchain_cols = [col for col in df.columns if 'onchain_' in col]
        for col in onchain_cols:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
        
        # Fill NaN values
        df = df.fillna(0)
        
        # Select features
        feature_columns = price_cols + ['volume', 'rsi', 'macd', 'macd_signal', 
                                       'bb_upper', 'bb_lower', 'atr', 'mom', 
                                       'volatility', 'ma_fast', 'ma_slow', 'sentiment'] + onchain_cols
        
        self.features = df[feature_columns].values
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        deltas = np.diff(prices)
        seed = deltas[:window+1]
        up = seed[seed >= 0].sum() / window
        down = -seed[seed < 0].sum() / window
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:window] = 100. - 100. / (1. + rs)
        
        for i in range(window, len(prices)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (window - 1) + upval) / window
            down = (down * (window - 1) + downval) / window
            rs = up / down if down != 0 else 0
            rsi[i] = 100. - 100. / (1. + rs)
        
        return rsi
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD, MACD Signal and MACD Histogram"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        
        return macd, macd_signal
    
    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        return upper_band, rolling_mean, lower_band
    
    def _calculate_atr(self, ohlc, window=14):
        """Calculate Average True Range"""
        high = ohlc['high']
        low = ohlc['low']
        close = ohlc['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(window=window).mean()
        
        return atr
    
    def reset(self):
        """Reset the environment"""
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = self.window_size
        self.done = False
        self.portfolio_value_history = [self.initial_balance]
        
        return self._get_observation()
    
    def step(self, action):
        """Take a step in the environment"""
        # Get current price
        current_price = self.df['close'].iloc[self.current_step]
        
        # Initialize reward
        reward = 0
        
        # Execute action
        if action == 1:  # Buy
            if self.balance > 0:
                # Calculate transaction cost
                transaction_cost = current_price * self.transaction_fee
                # Calculate max amount to buy
                max_buy_amount = self.balance / (current_price + transaction_cost)
                # Update position and balance
                self.position += max_buy_amount
                self.balance = 0
        
        elif action == 2:  # Sell
            if self.position > 0:
                # Calculate transaction cost
                transaction_cost = self.position * current_price * self.transaction_fee
                # Update balance and position
                self.balance += self.position * current_price - transaction_cost
                self.position = 0
        
        # Move to next step
        self.current_step += 1
        
        # Calculate portfolio value
        portfolio_value = self.balance + self.position * self.df['close'].iloc[self.current_step]
        self.portfolio_value_history.append(portfolio_value)
        
        # Calculate reward (change in portfolio value)
        reward = (portfolio_value - self.portfolio_value_history[-2]) / self.portfolio_value_history[-2]
        reward = reward * self.reward_scaling
        
        # Check if done
        if self.current_step >= len(self.df) - 1:
            self.done = True
        
        # Get observation
        observation = self._get_observation()
        
        # Additional info
        info = {
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'position': self.position,
            'current_price': current_price
        }
        
        return observation, reward, self.done, info
    
    def _get_observation(self):
        """Get the current observation"""
        # Get window of features
        features_window = self.features[self.current_step - self.window_size:self.current_step]
        
        # Add balance and position information
        balance_normalized = self.balance / self.initial_balance
        position_normalized = self.position * self.df['close'].iloc[self.current_step] / self.initial_balance
        
        # Create observation with shape (window_size, feature_dim + 2)
        observation = np.zeros((self.window_size, self.feature_dim + 2))
        
        # Fill features
        observation[:, :self.feature_dim] = features_window
        
        # Fill balance and position (same for all time steps in the window)
        observation[:, self.feature_dim] = balance_normalized
        observation[:, self.feature_dim + 1] = position_normalized
        
        return observation
    
    def render(self, mode='human'):
        """Render the environment"""
        if mode != 'human':
            raise NotImplementedError(f"Render mode {mode} is not supported")
        
        # Get current price and portfolio value
        current_price = self.df['close'].iloc[self.current_step]
        portfolio_value = self.balance + self.position * current_price
        
        print(f"Step: {self.current_step}")
        print(f"Price: {current_price:.2f}")
        print(f"Balance: {self.balance:.2f}")
        print(f"Position: {self.position:.6f}")
        print(f"Portfolio Value: {portfolio_value:.2f}")
        print(f"Return: {(portfolio_value / self.initial_balance - 1) * 100:.2f}%")
        print("-" * 50)
    
    def close(self):
        """Close the environment"""
        pass


class SACAgent:
    """
    Soft Actor-Critic (SAC) agent for continuous action spaces.
    SAC is an off-policy actor-critic deep RL algorithm based on the maximum entropy
    reinforcement learning framework.
    """
    def __init__(self, env, model_path='models/trained_models/sac'):
        self.env = env
        self.model_path = model_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Create model
        self.model = SAC(
            'MlpPolicy',
            env,
            learning_rate=3e-4,
            buffer_size=100000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            ent_coef='auto',
            target_update_interval=1,
            train_freq=1,
            gradient_steps=1,
            verbose=1
        )
    
    def train(self, total_timesteps=100000, eval_freq=10000):
        """Train the agent"""
        # Create evaluation callback
        eval_env = DummyVecEnv([lambda: self.env])
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=self.model_path,
            log_path=self.model_path,
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )
        
        # Train the agent
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback
        )
        
        # Save the final model
        self.model.save(f"{self.model_path}/final_model")
    
    def load(self, path=None):
        """Load a trained model"""
        if path is None:
            path = f"{self.model_path}/best_model"
        
        self.model = SAC.load(path, env=self.env)
    
    def predict(self, observation, deterministic=True):
        """Predict action based on observation"""
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return action


class TD3Agent:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) agent.
    TD3 is an algorithm that addresses function approximation error in actor-critic methods.
    It is an improvement over DDPG and addresses issues such as overestimation bias.
    """
    def __init__(self, env, model_path='models/trained_models/td3'):
        self.env = env
        self.model_path = model_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Create model
        self.model = TD3(
            'MlpPolicy',
            env,
            learning_rate=3e-4,
            buffer_size=100000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            verbose=1
        )
    
    def train(self, total_timesteps=100000, eval_freq=10000):
        """Train the agent"""
        # Create evaluation callback
        eval_env = DummyVecEnv([lambda: self.env])
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=self.model_path,
            log_path=self.model_path,
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )
        
        # Train the agent
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback
        )
        
        # Save the final model
        self.model.save(f"{self.model_path}/final_model")
    
    def load(self, path=None):
        """Load a trained model"""
        if path is None:
            path = f"{self.model_path}/best_model"
        
        self.model = TD3.load(path, env=self.env)
    
    def predict(self, observation, deterministic=True):
        """Predict action based on observation"""
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return action


class DQNAgent:
    """
    Deep Q-Network (DQN) agent with several improvements:
    - Double DQN
    - Dueling DQN
    - Prioritized Experience Replay
    """
    def __init__(self, state_size, action_size, model_path='models/trained_models/dqn'):
        self.state_size = state_size
        self.action_size = action_size
        self.model_path = model_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Hyperparameters
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.update_target_freq = 1000
        self.batch_size = 64
        
        # Create models
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Experience replay buffer
        self.memory = PrioritizedReplayBuffer(100000, 0.6)
        
        # Training step counter
        self.train_step = 0
    
    def _build_model(self):
        """Build a dueling DQN model"""
        # Input layer
        input_layer = Input(shape=(self.state_size,))
        
        # Shared layers
        x = Dense(256, activation='relu')(input_layer)
        x = BatchNormalization()(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        
        # Value stream
        value_stream = Dense(64, activation='relu')(x)
        value = Dense(1)(value_stream)
        
        # Advantage stream
        advantage_stream = Dense(64, activation='relu')(x)
        advantage = Dense(self.action_size)(advantage_stream)
        
        # Combine value and advantage streams
        q_values = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
        
        # Create model
        model = Model(inputs=input_layer, outputs=q_values)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        
        return model
    
    def update_target_model(self):
        """Update target model weights with current model weights"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Add experience to memory"""
        # Initial priority is set to max priority
        max_priority = self.memory.max_priority if self.memory.buffer else 1.0
        self.memory.add(state, action, reward, next_state, done, max_priority)
    
    def act(self, state, training=True):
        """Choose action based on epsilon-greedy policy"""
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = np.reshape(state, [1, self.state_size])
        q_values = self.model.predict(state, verbose=0)[0]
        return np.argmax(q_values)
    
    def replay(self):
        """Train the model with experiences from memory"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        experiences, indices, weights = self.memory.sample(self.batch_size)
        
        states = np.zeros((self.batch_size, self.state_size))
        actions = np.zeros(self.batch_size, dtype=np.int32)
        rewards = np.zeros(self.batch_size)
        next_states = np.zeros((self.batch_size, self.state_size))
        dones = np.zeros(self.batch_size, dtype=np.bool_)
        
        for i, experience in enumerate(experiences):
            states[i] = experience.state
            actions[i] = experience.action
            rewards[i] = experience.reward
            next_states[i] = experience.next_state
            dones[i] = experience.done
        
        # Double DQN: Use current model to select actions and target model to evaluate them
        q_values = self.model.predict(next_states, verbose=0)
        best_actions = np.argmax(q_values, axis=1)
        
        target_q_values = self.target_model.predict(next_states, verbose=0)
        target_q = np.array([target_q_values[i, best_actions[i]] for i in range(self.batch_size)])
        
        # Calculate target values
        targets = rewards + self.gamma * target_q * (1 - dones)
        
        # Get current Q values
        current_q = self.model.predict(states, verbose=0)
        
        # Calculate TD errors for prioritized replay
        td_errors = np.abs(targets - np.array([current_q[i, actions[i]] for i in range(self.batch_size)]))
        
        # Update priorities in memory
        for i in range(self.batch_size):
            self.memory.update_priority(indices[i], td_errors[i])
        
        # Update target values for selected actions
        for i in range(self.batch_size):
            current_q[i, actions[i]] = targets[i]
        
        # Train the model
        self.model.fit(states, current_q, batch_size=self.batch_size, epochs=1, verbose=0, sample_weight=weights)
        
        # Update target model periodically
        self.train_step += 1
        if self.train_step % self.update_target_freq == 0:
            self.update_target_model()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, path=None):
        """Save the model"""
        if path is None:
            path = f"{self.model_path}/model"
        
        self.model.save(path)
    
    def load(self, path=None):
        """Load the model"""
        if path is None:
            path = f"{self.model_path}/model"
        
        self.model = tf.keras.models.load_model(path)
        self.target_model = tf.keras.models.load_model(path)


class A3CAgent:
    """
    Asynchronous Advantage Actor-Critic (A3C) agent.
    A3C is a policy gradient method that maintains a policy and a value function,
    and uses multiple parallel workers to update a global network.
    """
    def __init__(self, env, model_path='models/trained_models/a3c'):
        self.env = env
        self.model_path = model_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Create model
        self.model = A2C(
            'MlpPolicy',
            env,
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_rms_prop=True,
            normalize_advantage=True,
            verbose=1
        )
    
    def train(self, total_timesteps=100000, eval_freq=10000):
        """Train the agent"""
        # Create evaluation callback
        eval_env = DummyVecEnv([lambda: self.env])
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=self.model_path,
            log_path=self.model_path,
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )
        
        # Train the agent
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback
        )
        
        # Save the final model
        self.model.save(f"{self.model_path}/final_model")
    
    def load(self, path=None):
        """Load a trained model"""
        if path is None:
            path = f"{self.model_path}/best_model"
        
        self.model = A2C.load(path, env=self.env)
    
    def predict(self, observation, deterministic=True):
        """Predict action based on observation"""
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return action


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent.
    PPO is a policy gradient method that uses a clipped surrogate objective
    to ensure stable updates.
    """
    def __init__(self, env, model_path='models/trained_models/ppo'):
        self.env = env
        self.model_path = model_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Create model
        self.model = PPO(
            'MlpPolicy',
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=None,
            verbose=1
        )
    
    def train(self, total_timesteps=100000, eval_freq=10000):
        """Train the agent"""
        # Create evaluation callback
        eval_env = DummyVecEnv([lambda: self.env])
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=self.model_path,
            log_path=self.model_path,
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )
        
        # Train the agent
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback
        )
        
        # Save the final model
        self.model.save(f"{self.model_path}/final_model")
    
    def load(self, path=None):
        """Load a trained model"""
        if path is None:
            path = f"{self.model_path}/best_model"
        
        self.model = PPO.load(path, env=self.env)
    
    def predict(self, observation, deterministic=True):
        """Predict action based on observation"""
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return action


class RainbowDQNAgent:
    """
    Rainbow DQN agent that combines multiple improvements to DQN:
    - Double Q-learning
    - Dueling networks
    - Prioritized experience replay
    - Multi-step learning
    - Distributional RL
    - Noisy networks
    """
    def __init__(self, state_size, action_size, model_path='models/trained_models/rainbow'):
        self.state_size = state_size
        self.action_size = action_size
        self.model_path = model_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Hyperparameters
        self.gamma = 0.99
        self.learning_rate = 0.0001
        self.batch_size = 32
        self.update_target_freq = 1000
        self.n_step = 3  # Multi-step learning
        self.num_atoms = 51  # Number of atoms for distributional RL
        self.v_min = -10.0  # Minimum value for distributional RL
        self.v_max = 10.0  # Maximum value for distributional RL
        
        # Create PyTorch device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create models
        self.model = RainbowNetwork(state_size, action_size, self.num_atoms, self.v_min, self.v_max).to(self.device)
        self.target_model = RainbowNetwork(state_size, action_size, self.num_atoms, self.v_min, self.v_max).to(self.device)
        self.update_target_model()
        
        # Create optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Experience replay buffer
        self.memory = NStepPrioritizedReplayBuffer(100000, 0.6, self.n_step, self.gamma)
        
        # Training step counter
        self.train_step = 0
        
        # Support for distributional RL
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(self.device)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
    
    def update_target_model(self):
        """Update target model weights with current model weights"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Add experience to memory"""
        # Initial priority is set to max priority
        max_priority = self.memory.max_priority if self.memory.buffer else 1.0
        self.memory.add(state, action, reward, next_state, done, max_priority)
    
    def act(self, state, training=True):
        """Choose action based on current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get distributional value
            dist = self.model(state)
            # Calculate expected value
            expected_value = (dist * self.support).sum(2)
            # Choose action with highest expected value
            action = expected_value.argmax(1).item()
        
        return action
    
    def replay(self):
        """Train the model with experiences from memory"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        experiences, indices, weights = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.FloatTensor([e.done for e in experiences]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Get current state distributions
        current_dist = self.model(states)
        log_p = current_dist[range(self.batch_size), actions]
        
        with torch.no_grad():
            # Get next state distributions
            next_dist = self.target_model(next_states)
            
            # Calculate target distribution
            target_dist = torch.zeros((self.batch_size, self.num_atoms), device=self.device)
            
            for i in range(self.batch_size):
                if dones[i]:
                    # If terminal state, only consider immediate reward
                    tz = min(self.v_max, max(self.v_min, rewards[i]))
                    bj = int((tz - self.v_min) / self.delta_z)
                    target_dist[i, bj] = 1.0
                else:
                    # Calculate projected distribution
                    for j in range(self.num_atoms):
                        tz = min(self.v_max, max(self.v_min, rewards[i] + self.gamma * self.support[j]))
                        bj = int((tz - self.v_min) / self.delta_z)
                        target_dist[i, bj] += next_dist[i, actions[i].item(), j]
        
        # Calculate KL divergence loss
        loss = -(target_dist * torch.log(log_p + 1e-8)).sum(1)
        
        # Apply importance sampling weights
        weighted_loss = (loss * weights).mean()
        
        # Update model
        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()
        
        # Update priorities in memory
        priorities = loss.detach().cpu().numpy()
        for i in range(self.batch_size):
            self.memory.update_priority(indices[i], priorities[i])
        
        # Update target model periodically
        self.train_step += 1
        if self.train_step % self.update_target_freq == 0:
            self.update_target_model()
    
    def save(self, path=None):
        """Save the model"""
        if path is None:
            path = f"{self.model_path}/model.pt"
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_step': self.train_step
        }, path)
    
    def load(self, path=None):
        """Load the model"""
        if path is None:
            path = f"{self.model_path}/model.pt"
        
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_step = checkpoint['train_step']


# Supporting classes for the agents

class RainbowNetwork(nn.Module):
    """
    Neural network for Rainbow DQN with noisy layers and dueling architecture.
    """
    def __init__(self, state_size, action_size, num_atoms, v_min, v_max):
        super(RainbowNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # Feature layers
        self.features = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Value stream (dueling architecture)
        self.value_stream = nn.Sequential(
            NoisyLinear(128, 128),
            nn.ReLU(),
            NoisyLinear(128, num_atoms)
        )
        
        # Advantage stream (dueling architecture)
        self.advantage_stream = nn.Sequential(
            NoisyLinear(128, 128),
            nn.ReLU(),
            NoisyLinear(128, action_size * num_atoms)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Get features
        features = self.features(x)
        
        # Get value and advantage
        value = self.value_stream(features).view(batch_size, 1, self.num_atoms)
        advantage = self.advantage_stream(features).view(batch_size, self.action_size, self.num_atoms)
        
        # Combine value and advantage (dueling architecture)
        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Apply softmax to get probability distribution
        q_dist = F.softmax(q_dist, dim=2)
        
        return q_dist


class NoisyLinear(nn.Module):
    """
    Noisy linear layer for exploration in Rainbow DQN.
    """
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        
        # Initialize parameters
        self.reset_parameters()
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        self.reset_noise()
    
    def reset_parameters(self):
        """Reset learnable parameters"""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        """Reset noise parameters"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        """Scale noise"""
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
    
    def forward(self, x):
        """Forward pass with noise"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)


# Experience replay buffers

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer for DQN.
    """
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = 0.4  # Importance sampling exponent
        self.beta_increment = 0.001  # Beta increment per sampling
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def add(self, state, action, reward, next_state, done, priority=None):
        """Add experience to buffer"""
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        # Set priority
        if priority is None:
            priority = self.max_priority
        
        if self.position < len(self.priorities):
            self.priorities[self.position] = priority
        else:
            self.priorities = np.append(self.priorities, priority)
        
        # Update position
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample batch of experiences based on priorities"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Create named tuples for experiences
        Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])
        experiences = [Experience(*exp) for exp in experiences]
        
        return experiences, indices, weights
    
    def update_priority(self, index, priority):
        """Update priority of experience"""
        priority = max(1e-8, priority)  # Ensure priority is positive
        self.max_priority = max(self.max_priority, priority)
        self.priorities[index] = priority
    
    def __len__(self):
        """Return current size of buffer"""
        return len(self.buffer)


class NStepPrioritizedReplayBuffer(PrioritizedReplayBuffer):
    """
    N-step Prioritized Experience Replay buffer for Rainbow DQN.
    """
    def __init__(self, capacity, alpha, n_step, gamma):
        super(NStepPrioritizedReplayBuffer, self).__init__(capacity, alpha)
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)
    
    def add(self, state, action, reward, next_state, done, priority=None):
        """Add experience to n-step buffer"""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) < self.n_step:
            return
        
        # Calculate n-step reward and get n-step next state
        reward, next_state, done = self._get_n_step_info()
        state, action, _, _, _ = self.n_step_buffer[0]
        
        # Add n-step transition to buffer
        super().add(state, action, reward, next_state, done, priority)
    
    def _get_n_step_info(self):
        """Calculate n-step reward and get n-step next state"""
        reward, next_state, done = 0, None, False
        
        for idx, (_, _, r, next_s, d) in enumerate(self.n_step_buffer):
            reward += r * (self.gamma ** idx)
            if d:
                done = True
                break
        
        next_state = next_s
        
        return reward, next_state, done
