# AI Crypto Trading Bot with Web Dashboard

This is an advanced cryptocurrency trading bot that uses machine learning models (LSTM, Hybrid, and Reinforcement Learning) to make trading decisions on the Kraken exchange. The bot includes a web dashboard for monitoring and controlling your trading activities.

## Features

- **Multiple Trading Models**: LSTM, Hybrid Neural Networks, and Reinforcement Learning
- **Real-time Trading**: Connect to Kraken API for real-time trading
- **Risk Management**: Advanced risk management to protect your capital
- **Web Dashboard**: Monitor your portfolio, trades, and bot performance
- **Multiple Strategies**: Grid Trading, Mean Reversion, and Breakout strategies
- **Sentiment Analysis**: Incorporates market sentiment in trading decisions
- **Performance Tracking**: Track and analyze your trading performance

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your API keys in the `.env` file:
   ```
   KRAKEN_API_KEY="your_api_key"
   KRAKEN_API_SECRET="your_api_secret"
   ```
4. Configure your trading pairs and settings in `config.json`

## Running the Bot

### Using the Web Dashboard (Recommended)

1. Start the web dashboard:
   ```
   python -m dashboard.app
   ```
2. Open your browser and navigate to `http://localhost:5000`
3. Configure your settings in the dashboard
4. Click "Start Bot" to begin trading with real money

### Using the Command Line

1. Run the bot directly:
   ```
   python main.py
   ```
2. To run without retraining models (faster startup):
   ```
   python main.py --trade-only
   ```

## Trading with Real Money

To trade with real money on Kraken:

1. Make sure you have valid API keys with trading permissions set in your `.env` file
2. Ensure you have sufficient funds in your Kraken account
3. Start the bot using either the web dashboard or command line
4. The bot will automatically execute trades based on model predictions

**Important Notes for Real Trading:**
- Start with small amounts until you're comfortable with the bot's performance
- Monitor the bot regularly through the dashboard
- Check the logs for any errors or issues
- The bot trades on the pairs specified in your configuration

## Dashboard Pages

- **Home**: Overview of portfolio value, active trading pairs, and recent trades
- **Portfolio**: Detailed portfolio performance analytics
- **Trades**: Comprehensive trade history with filtering
- **Models**: Management interface for trained models and strategies
- **Settings**: Complete configuration control panel
- **Logs**: Real-time log monitoring

## Customization

- Modify trading pairs in `config.json` or through the dashboard
- Adjust risk parameters in the Settings page
- Enable/disable specific models or strategies
- Configure notification settings for trade alerts

## Troubleshooting

If the bot isn't trading:
1. Check your API keys have trading permissions
2. Verify you have sufficient funds in your account
3. Ensure the minimum trade size is set appropriately
4. Check the logs for any errors
5. Make sure the models are properly trained

## Disclaimer

Trading cryptocurrencies involves significant risk. This bot is provided as-is with no guarantees of profitability. Always start with small amounts and never trade more than you can afford to lose.
