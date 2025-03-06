import os
import sys
import json
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for
import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import threading
import time
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

from config_manager import config
from monitoring.performance_tracker import PerformanceTracker
from utils.log_setup import logger
from dashboard.chat_ai import chat_ai

# Initialize Flask app
app = Flask(__name__)

# Global variables
bot_status = {
    'running': False,
    'last_update': None,
    'active_pairs': [],
    'portfolio_value': 0.0,
    'balance_usd': 0.0,
    'balance_assets': {},
    'active_strategies': {},
    'recent_trades': []
}

# Performance tracker
performance_tracker = None

# Bot control thread
bot_thread = None

# Initialize performance tracker
def init_performance_tracker():
    global performance_tracker
    if performance_tracker is None:
        performance_tracker = PerformanceTracker()
        # Load existing data if available
        performance_tracker.load_from_csv()

# Routes
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html', bot_status=bot_status)

@app.route('/portfolio')
def portfolio():
    """Portfolio performance page"""
    return render_template('portfolio.html', bot_status=bot_status)

@app.route('/trades')
def trades():
    """Trade history page"""
    return render_template('trades.html', bot_status=bot_status)

@app.route('/settings')
def settings():
    """Settings page"""
    # Load current configuration
    current_config = config.get_all()
    return render_template('settings.html', config=current_config, bot_status=bot_status)

@app.route('/models')
def models():
    """Models and strategies page"""
    return render_template('models.html', bot_status=bot_status)

@app.route('/logs')
def logs():
    """Log viewer page"""
    return render_template('logs.html', bot_status=bot_status)

@app.route('/chat')
def chat():
    """AI Chat page"""
    return render_template('chat.html', bot_status=bot_status)

# API endpoints
@app.route('/api/status')
def api_status():
    """Get current bot status"""
    return jsonify(bot_status)

@app.route('/api/start', methods=['POST'])
def api_start():
    """Start the trading bot"""
    global bot_status, bot_thread
    
    if not bot_status['running']:
        # Start bot in a separate thread
        bot_thread = threading.Thread(target=run_bot)
        bot_thread.daemon = True
        bot_thread.start()
        
        bot_status['running'] = True
        bot_status['last_update'] = datetime.now().isoformat()
        
        return jsonify({'success': True, 'message': 'Bot started successfully'})
    else:
        return jsonify({'success': False, 'message': 'Bot is already running'})

@app.route('/api/stop', methods=['POST'])
def api_stop():
    """Stop the trading bot"""
    global bot_status
    
    if bot_status['running']:
        # Set flag to stop the bot
        bot_status['running'] = False
        bot_status['last_update'] = datetime.now().isoformat()
        
        return jsonify({'success': True, 'message': 'Bot stopped successfully'})
    else:
        return jsonify({'success': False, 'message': 'Bot is not running'})

@app.route('/api/portfolio/history')
def api_portfolio_history():
    """Get portfolio value history"""
    global performance_tracker
    
    # Initialize if needed
    init_performance_tracker()
    
    # Get timeframe from query parameters
    timeframe = request.args.get('timeframe', 'week')
    
    # Generate portfolio history
    history = performance_tracker.get_portfolio_history(timeframe)
    
    # Convert to list of dictionaries for JSON
    result = []
    for timestamp, value in history.items():
        result.append({
            'timestamp': timestamp.isoformat(),
            'value': value
        })
    
    return jsonify(result)

@app.route('/api/portfolio/composition')
def api_portfolio_composition():
    """Get current portfolio composition"""
    return jsonify(bot_status['balance_assets'])

@app.route('/api/trades/recent')
def api_trades_recent():
    """Get recent trades"""
    global performance_tracker
    
    # Initialize if needed
    init_performance_tracker()
    
    # Get limit from query parameters
    limit = int(request.args.get('limit', 20))
    
    # Get recent trades
    trades = performance_tracker.get_recent_trades(limit)
    
    return jsonify(trades)

@app.route('/api/trades/stats')
def api_trades_stats():
    """Get trade statistics"""
    global performance_tracker
    
    # Initialize if needed
    init_performance_tracker()
    
    # Get timeframe from query parameters
    timeframe = request.args.get('timeframe', 'all')
    
    # Generate performance report
    report = performance_tracker.generate_performance_report(timeframe)
    
    return jsonify(report)

@app.route('/api/settings', methods=['GET'])
def api_settings_get():
    """Get current settings"""
    return jsonify(config.get_all())

@app.route('/api/settings', methods=['POST'])
def api_settings_update():
    """Update settings"""
    new_settings = request.json
    
    # Update configuration
    for key, value in new_settings.items():
        config.set(key, value)
    
    # Save configuration
    config.save_to_file()
    
    return jsonify({'success': True, 'message': 'Settings updated successfully'})

@app.route('/api/models/list')
def api_models_list():
    """List available models"""
    models_dir = config.get('model_dir', 'models/trained_models')
    
    # Get list of model files
    model_files = []
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith('.h5') or file.endswith('.zip'):
                model_files.append(file)
    
    # Group by symbol and type
    models = {}
    for file in model_files:
        parts = file.split('_')
        if len(parts) >= 2:
            model_type = parts[0]
            symbol = '_'.join(parts[1:]).replace('.h5', '').replace('.zip', '')
            
            if symbol not in models:
                models[symbol] = []
            
            models[symbol].append({
                'type': model_type,
                'file': file,
                'last_modified': datetime.fromtimestamp(os.path.getmtime(os.path.join(models_dir, file))).isoformat()
            })
    
    return jsonify(models)

@app.route('/api/strategies/list')
def api_strategies_list():
    """List available strategies"""
    strategies = [
        {
            'id': 'grid_trading',
            'name': 'Grid Trading',
            'description': 'Places buy and sell orders at regular price intervals'
        },
        {
            'id': 'mean_reversion',
            'name': 'Mean Reversion',
            'description': 'Trades based on the assumption that prices will revert to their mean'
        },
        {
            'id': 'breakout',
            'name': 'Breakout',
            'description': 'Identifies key support/resistance levels and trades on breakouts'
        }
    ]
    
    return jsonify(strategies)

@app.route('/api/logs/recent')
def api_logs_recent():
    """Get recent log entries"""
    # Get limit from query parameters
    limit = int(request.args.get('limit', 100))
    since_id = int(request.args.get('since', 0))
    
    # Get log file path
    log_file = config.get('log_file', 'crypto_trading_bot.log')
    
    # Read log file
    logs = []
    log_id = since_id
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
            
            # Start from the end if we want recent logs
            start_idx = max(0, len(lines) - limit) if since_id == 0 else 0
            
            for i, line in enumerate(lines[start_idx:], start=start_idx):
                if i <= since_id:
                    continue
                    
                log_id = i
                line = line.strip()
                
                # Parse log entry
                try:
                    # Example format: 2023-03-04 12:34:56,789 - INFO - main - Message
                    parts = line.split(' - ')
                    timestamp = parts[0]
                    level = parts[1].strip() if len(parts) > 1 else 'INFO'
                    component = parts[2].strip() if len(parts) > 2 else 'system'
                    message = ' - '.join(parts[3:]) if len(parts) > 3 else line
                    
                    logs.append({
                        'id': log_id,
                        'timestamp': timestamp,
                        'level': level,
                        'component': component,
                        'message': message,
                        'raw': line
                    })
                    
                    # Add to chat AI log history
                    chat_ai.add_log(line)
                except Exception as e:
                    # Fallback for unparseable logs
                    logs.append({
                        'id': log_id,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3],
                        'level': 'UNKNOWN',
                        'component': 'system',
                        'message': line,
                        'raw': line
                    })
    
    return jsonify({'logs': logs, 'last_id': log_id})

@app.route('/api/chart/price')
def api_chart_price():
    """Get price chart data"""
    # Get symbol and timeframe from query parameters
    symbol = request.args.get('symbol', 'BTC/USD')
    timeframe = request.args.get('timeframe', '1h')
    
    # Load historical data
    data_dir = 'data/historical'
    file_path = os.path.join(data_dir, f"{symbol.replace('/', '_')}_{timeframe}.csv")
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create candlestick chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, subplot_titles=('Price', 'Volume'),
                           row_heights=[0.7, 0.3])
        
        # Add candlestick trace
        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ), row=1, col=1)
        
        # Add volume trace
        fig.add_trace(go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume'
        ), row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Price Chart',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False
        )
        
        # Convert to JSON
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return jsonify({'chart': chart_json})
    else:
        return jsonify({'error': f'No data found for {symbol} {timeframe}'})

@app.route('/api/chart/performance')
def api_chart_performance():
    """Get performance chart data"""
    global performance_tracker
    
    # Initialize if needed
    init_performance_tracker()
    
    # Get timeframe from query parameters
    timeframe = request.args.get('timeframe', 'month')
    
    # Generate portfolio history
    history = performance_tracker.get_portfolio_history(timeframe)
    
    # Create line chart
    fig = go.Figure()
    
    # Add portfolio value trace
    timestamps = list(history.keys())
    values = list(history.values())
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=values,
        mode='lines',
        name='Portfolio Value'
    ))
    
    # Update layout
    fig.update_layout(
        title='Portfolio Performance',
        xaxis_title='Date',
        yaxis_title='Value (USD)'
    )
    
    # Convert to JSON
    chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return jsonify({'chart': chart_json})

# Chat API endpoints
@app.route('/api/chat/message')
def api_chat_message():
    """Get a message from the AI chat"""
    force = request.args.get('force', 'false').lower() == 'true'
    message = chat_ai.get_summary(bot_status, force)
    return jsonify({'message': message})

@app.route('/api/chat/personality', methods=['POST'])
def api_chat_personality():
    """Change the AI chat personality"""
    personality = request.form.get('personality', 'analytical')
    success = chat_ai.change_personality(personality)
    return jsonify({'success': success})

@app.route('/api/chat/query', methods=['POST'])
def api_chat_query():
    """Process a user query to the AI chat"""
    query = request.form.get('query', '')
    
    # Simple response generation based on query
    response = None
    
    if re.search(r'(hi|hello|hey)', query.lower()):
        response = "Hello! How can I help you with your trading bot today?"
    elif re.search(r'(status|how.+bot|what.+doing)', query.lower()):
        response = chat_ai.generate_message(bot_status)
    elif re.search(r'(balance|portfolio|account)', query.lower()):
        portfolio_value = bot_status.get('portfolio_value', 0)
        balance_usd = bot_status.get('balance_usd', 0)
        assets = bot_status.get('balance_assets', {})
        
        asset_str = ", ".join([f"{amount} {symbol}" for symbol, amount in assets.items() if amount > 0])
        
        response = f"Your current portfolio value is ${portfolio_value:.2f}. You have ${balance_usd:.2f} in USD and {asset_str}."
    elif re.search(r'(trade|buy|sell)', query.lower()):
        recent_trades = bot_status.get('recent_trades', [])
        
        if recent_trades:
            trade = recent_trades[0]
            response = f"The most recent trade was a {trade.get('type')} of {trade.get('amount')} {trade.get('pair').split('/')[0]} at ${trade.get('price')}."
        else:
            response = "There haven't been any trades recently."
    elif re.search(r'(predict|forecast|trend)', query.lower()):
        response = "Based on current market analysis, our models are showing mixed signals. Some assets appear bullish while others show bearish trends. I recommend monitoring closely before making any major decisions."
    else:
        response = "I'm still learning how to respond to that type of question. In the meantime, I can tell you about your portfolio status, recent trades, or market predictions."
    
    return jsonify({'response': response})

# Import necessary components from main.py and config.py
from execution.trade_executor import TradeExecutor
from risk_management.risk_manager import RiskManager
from strategies.grid_trading import GridTrader
from utils.data_utils import fetch_real_time_data, process_data, fetch_historical_data, augment_data
from models.hybrid_model import HybridCryptoModel
from models.lstm_model import LSTMModel
from models.backtest import backtest_model
from stable_baselines3 import PPO
import gym
from gym import spaces
from config import TRADING_PAIRS, ACTIVE_EXCHANGE

# Real trading function
def run_bot():
    """Run the actual trading bot with real money"""
    global bot_status, performance_tracker
    
    logger.info("Starting real trading bot with Kraken API")
    
    # Load trading pairs from config
    trading_pairs = config.get('trading_pairs', ['BTC/USD', 'ETH/USD', 'XRP/USD'])
    bot_status['active_pairs'] = trading_pairs
    
    # Initialize performance tracker
    init_performance_tracker()
    
    # Initialize trading components
    executor = TradeExecutor()
    risk_manager = RiskManager(max_loss=config.get('max_loss', 0.5))
    
    # Initialize models and data
    dataframes = {}
    hybrid_models = {}
    lstm_models = {}
    ppo_models = {}
    grid_traders = {}
    
    model_dir = 'models/trained_models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    
    # Initial setup for each trading pair
    for symbol in trading_pairs:
        try:
            # Fetch and process historical data
            df = fetch_historical_data(symbol)
            df_augmented = augment_data(df)
            df_processed = process_data(df_augmented, symbol)
            dataframes[symbol] = df_processed
            logger.info(f"Initial historical data fetched and processed for {symbol}: {len(df_processed)} rows")
            
            # Load or train models
            hybrid_path = f'{model_dir}/hybrid_{symbol.replace("/", "_")}.h5'
            lstm_path = f'{model_dir}/lstm_{symbol.replace("/", "_")}.h5'
            ppo_path = f'{model_dir}/ppo_{symbol.replace("/", "_")}'
            
            # Try to load existing models
            try:
                hybrid_model = HybridCryptoModel(sequence_length=50, n_features=9)
                if os.path.exists(hybrid_path):
                    hybrid_model.model.load_weights(hybrid_path)
                    logger.info(f"Loaded existing hybrid model for {symbol}")
                else:
                    logger.warning(f"No existing hybrid model found for {symbol}. Using untrained model.")
                
                lstm_model = LSTMModel(sequence_length=50, n_features=9)
                if os.path.exists(lstm_path):
                    lstm_model.model.load_weights(lstm_path)
                    logger.info(f"Loaded existing LSTM model for {symbol}")
                else:
                    logger.warning(f"No existing LSTM model found for {symbol}. Using untrained model.")
                
                hybrid_models[symbol] = hybrid_model
                lstm_models[symbol] = lstm_model
                
                # Initialize grid trader
                grid_config = {'grid_trading': {'num_grids': 10, 'grid_spread': 0.05, 'max_position': 1.0, 'min_profit': 0.2}}
                grid_traders[symbol] = GridTrader(grid_config)
                
                # Get current balances and prices
                balance_usd, balance_asset = executor.get_balance(symbol)
                current_price = executor.fetch_current_price(symbol)
                
                # Update bot status with real balances
                bot_status['balance_usd'] = balance_usd
                if symbol.split('/')[0] not in bot_status['balance_assets']:
                    bot_status['balance_assets'][symbol.split('/')[0]] = balance_asset
                
                # Calculate portfolio value
                portfolio_value = balance_usd
                for asset, amount in bot_status['balance_assets'].items():
                    asset_price = executor.fetch_current_price(f"{asset}/USD")
                    portfolio_value += amount * asset_price
                
                bot_status['portfolio_value'] = portfolio_value
                bot_status['last_update'] = datetime.now().isoformat()
                
                # Update performance tracker
                performance_tracker.update_portfolio_value(
                    portfolio_value,
                    bot_status['balance_assets']
                )
                
                logger.info(f"Initialized {symbol} with balance USD: {balance_usd}, balance {symbol.split('/')[0]}: {balance_asset}")
            except Exception as e:
                logger.error(f"Error initializing models for {symbol}: {e}")
        except Exception as e:
            logger.error(f"Error setting up {symbol}: {e}")
    
    # Main trading loop
    while bot_status['running']:
        try:
            # Update portfolio value and balances
            total_usd = 0
            total_assets = {}
            
            for symbol in trading_pairs:
                try:
                    balance_usd, balance_asset = executor.get_balance(symbol)
                    current_price = executor.fetch_current_price(symbol)
                    
                    total_usd += balance_usd
                    asset = symbol.split('/')[0]
                    if asset not in total_assets:
                        total_assets[asset] = 0
                    total_assets[asset] += balance_asset * current_price
                    
                    # Update bot status
                    bot_status['balance_usd'] = total_usd
                    bot_status['balance_assets'] = {asset: amount / current_price for asset, amount in total_assets.items()}
                    
                    # Get latest data
                    new_data = fetch_real_time_data(symbol)
                    df = pd.concat([dataframes[symbol], new_data]).tail(100)
                    df = process_data(df, symbol)
                    dataframes[symbol] = df
                    
                    # Make prediction
                    X = df[['momentum', 'rsi', 'macd', 'atr', 'sentiment', 'arbitrage_spread', 'whale_activity', 'bb_upper', 'defi_apr']].iloc[-50:].values
                    if len(X) < 50:
                        X = np.pad(X, ((50 - len(X), 0), (0, 0)), mode='edge')
                    
                    hybrid_pred = hybrid_models[symbol].predict(np.expand_dims(X, axis=0))[0][0].item()
                    lstm_pred = lstm_models[symbol].predict(np.expand_dims(X, axis=0))[0][0].item()
                    ensemble_pred = np.mean([hybrid_pred, lstm_pred])
                    
                    # Determine action
                    action = 1 if ensemble_pred > 0.5 else 2  # 1=buy, 2=sell
                    
                    # Log prediction
                    logger.info(f"Prediction for {symbol}: hybrid={hybrid_pred:.4f}, lstm={lstm_pred:.4f}, ensemble={ensemble_pred:.4f}, action={action}")
                    
                    # Execute trade if conditions are met
                    min_trade_size = executor.min_trade_sizes.get(symbol, 10.0)
                    
                    if action == 1 and balance_usd > min_trade_size:  # Buy
                        amount = min_trade_size / current_price
                        if risk_manager.is_safe(action, symbol, balance_usd, balance_asset, current_price):
                            logger.info(f"Executing BUY for {symbol}: {amount} at {current_price}")
                            order, retry = executor.execute(action, symbol, amount)
                            
                            if order:
                                # Log successful trade
                                trade = {
                                    'timestamp': datetime.now().isoformat(),
                                    'pair': symbol,
                                    'type': 'buy',
                                    'price': current_price,
                                    'amount': amount,
                                    'value': current_price * amount,
                                    'strategy': 'ensemble'
                                }
                                
                                # Add to recent trades
                                bot_status['recent_trades'].insert(0, trade)
                                
                                # Keep only the most recent 20 trades
                                if len(bot_status['recent_trades']) > 20:
                                    bot_status['recent_trades'] = bot_status['recent_trades'][:20]
                                
                                # Log trade in performance tracker
                                performance_tracker.log_trade(
                                    symbol, 
                                    1,  # buy
                                    amount, 
                                    current_price, 
                                    strategy='ensemble'
                                )
                                
                                # Force AI to generate a message about this trade
                                chat_ai.get_summary(bot_status, True)
                                
                                logger.info(f"Successfully executed BUY for {symbol}: {order}")
                            else:
                                logger.warning(f"Failed to execute BUY for {symbol}")
                        else:
                            logger.info(f"Risk manager prevented BUY for {symbol}")
                    
                    elif action == 2 and balance_asset > 0:  # Sell
                        amount = min(balance_asset, min_trade_size / current_price)
                        if amount > 0 and risk_manager.is_safe(action, symbol, balance_usd, balance_asset, current_price):
                            logger.info(f"Executing SELL for {symbol}: {amount} at {current_price}")
                            order, retry = executor.execute(action, symbol, amount)
                            
                            if order:
                                # Log successful trade
                                trade = {
                                    'timestamp': datetime.now().isoformat(),
                                    'pair': symbol,
                                    'type': 'sell',
                                    'price': current_price,
                                    'amount': amount,
                                    'value': current_price * amount,
                                    'strategy': 'ensemble'
                                }
                                
                                # Add to recent trades
                                bot_status['recent_trades'].insert(0, trade)
                                
                                # Keep only the most recent 20 trades
                                if len(bot_status['recent_trades']) > 20:
                                    bot_status['recent_trades'] = bot_status['recent_trades'][:20]
                                
                                # Log trade in performance tracker
                                performance_tracker.log_trade(
                                    symbol, 
                                    2,  # sell
                                    amount, 
                                    current_price, 
                                    strategy='ensemble'
                                )
                                
                                # Force AI to generate a message about this trade
                                chat_ai.get_summary(bot_status, True)
                                
                                logger.info(f"Successfully executed SELL for {symbol}: {order}")
                            else:
                                logger.warning(f"Failed to execute SELL for {symbol}")
                        else:
                            logger.info(f"Risk manager prevented SELL for {symbol} or amount too small")
                    
                    # Update active strategies
                    if 'active_strategies' not in bot_status:
                        bot_status['active_strategies'] = {}
                    bot_status['active_strategies'][symbol] = 'ensemble'
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
            
            # Calculate total portfolio value
            portfolio_value = total_usd + sum(total_assets.values())
            bot_status['portfolio_value'] = portfolio_value
            bot_status['last_update'] = datetime.now().isoformat()
            
            # Update performance tracker with new portfolio value
            performance_tracker.update_portfolio_value(
                portfolio_value,
                {asset: amount for asset, amount in bot_status['balance_assets'].items()}
            )
            
            logger.info(f"Updated portfolio value: ${portfolio_value:.2f}")
            
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
        
        # Sleep for a bit before next iteration
        time.sleep(60)  # Check every minute

# Run the app
if __name__ == '__main__':
    # Initialize performance tracker
    init_performance_tracker()
    
    # Add sidebar link to base.html
    try:
        with open('dashboard/templates/base.html', 'r') as f:
            content = f.read()
        
        if '<a class="nav-link {% if request.path == \'/chat\' %}active{% endif %}" href="{{ url_for(\'chat\') }}">' not in content:
            # Add chat link to sidebar
            content = content.replace(
                '<li class="nav-item">\n                        <a class="nav-link {% if request.path == \'/logs\' %}active{% endif %}" href="{{ url_for(\'logs\') }}">\n                            <i class="fas fa-list"></i> Logs\n                        </a>\n                    </li>',
                '<li class="nav-item">\n                        <a class="nav-link {% if request.path == \'/logs\' %}active{% endif %}" href="{{ url_for(\'logs\') }}">\n                            <i class="fas fa-list"></i> Logs\n                        </a>\n                    </li>\n                    <li class="nav-item">\n                        <a class="nav-link {% if request.path == \'/chat\' %}active{% endif %}" href="{{ url_for(\'chat\') }}">\n                            <i class="fas fa-robot"></i> AI Chat\n                        </a>\n                    </li>'
            )
            
            with open('dashboard/templates/base.html', 'w') as f:
                f.write(content)
    except Exception as e:
        print(f"Error updating base.html: {e}")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
