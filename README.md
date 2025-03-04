# Enhanced AI Crypto Trading Bot

An advanced cryptocurrency trading bot that uses machine learning, reinforcement learning, and multiple trading strategies to make data-driven trading decisions.

## Features

- **Multi-Model Ensemble**: Combines predictions from LSTM, CNN-LSTM hybrid, and Transformer models
- **Multiple Trading Strategies**:
  - Grid Trading
  - Mean Reversion
  - Breakout Trading
  - Reinforcement Learning (PPO)
- **Advanced Risk Management**:
  - Kelly Criterion for position sizing
  - Value at Risk (VaR) calculations
  - Correlation-based portfolio optimization
  - Dynamic stop-loss and take-profit levels
- **Enhanced Data Analysis**:
  - Multi-exchange data integration
  - On-chain metrics (simulated)
  - Advanced technical indicators
  - Market regime detection
- **Advanced Sentiment Analysis**:
  - Reddit sentiment analysis
  - Twitter sentiment simulation
  - Crypto news sentiment simulation
  - Sentiment momentum tracking
- **Performance Monitoring**:
  - Comprehensive metrics (Sharpe, Sortino, win rate, etc.)
  - Trade history tracking
  - Portfolio value visualization
  - Performance reporting

## Architecture

The bot is built with a modular architecture that allows for easy extension and customization:

```
├── config_manager.py      # Configuration management
├── config.json            # Configuration settings
├── main.py                # Original main script
├── enhanced_main.py       # Enhanced main script
├── data/                  # Data storage
├── execution/             # Trade execution
├── models/                # ML models
├── monitoring/            # Performance tracking
├── risk_management/       # Risk management
├── sentiment/             # Sentiment analysis
├── strategies/            # Trading strategies
└── utils/                 # Utility functions
```

## Getting Started

### Prerequisites

- Python 3.9+
- TA-Lib (Technical Analysis Library)
- Exchange API keys (Kraken, etc.)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/crypto-trading-bot.git
cd crypto-trading-bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install TA-Lib (platform-specific):
```bash
# For Windows:
pip install ta_lib-0.6.0-cp312-cp312-win_amd64.whl

# For Linux:
# Follow instructions at https://github.com/mrjbq7/ta-lib
```

4. Configure your API keys in `.env`:
```
KRAKEN_API_KEY=your_api_key
KRAKEN_API_SECRET=your_api_secret
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
MEM0_API_KEY=your_mem0_api_key
```

### Usage

1. Run the enhanced trading bot:
```bash
python enhanced_main.py
```

2. Run in trade-only mode (no retraining):
```bash
python enhanced_main.py --trade-only
```

3. Specify a custom configuration file:
```bash
python enhanced_main.py --config custom_config.json
```

## Configuration

The bot can be configured through the `config.json` file. Key settings include:

- `trading_pairs`: List of trading pairs to trade
- `model_types`: List of model types to use
- Feature flags for enabling/disabling enhancements:
  - `use_enhanced_data`: Enable multi-exchange data
  - `advanced_sentiment`: Enable advanced sentiment analysis
  - `advanced_risk`: Enable advanced risk management
  - `use_transformer_model`: Enable transformer model
  - `use_strategy_selector`: Enable adaptive strategy selection
  - `use_performance_tracking`: Enable performance monitoring

## Extending the Bot

The modular architecture makes it easy to extend the bot with new features:

1. **Add a new model**: Create a new model class in the `models/` directory
2. **Add a new strategy**: Create a new strategy class in the `strategies/` directory
3. **Add a new data source**: Extend the `MultiExchangeDataFetcher` class in `utils/enhanced_data_utils.py`

## Disclaimer

This software is for educational purposes only. Use at your own risk. Cryptocurrency trading involves significant risk and you can lose money. Past performance is not indicative of future results.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
