import os
import sys
import json
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
    config.save()
    
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
    
    # Get log file path
    log_file = config.get('log_file', 'crypto_trading_bot.log')
    
    # Read log file
    logs = []
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines[-limit:]:
                logs.append(line.strip())
    
    return jsonify(logs)

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

# Bot simulation function (for testing)
def run_bot():
    """Simulate running the trading bot"""
    global bot_status
    
    # Load trading pairs from config
    trading_pairs = config.get('trading_pairs', ['BTC/USD', 'ETH/USD', 'XRP/USD'])
    bot_status['active_pairs'] = trading_pairs
    
    # Initialize performance tracker
    init_performance_tracker()
    
    # Simulate trading
    while bot_status['running']:
        # Update portfolio value
        portfolio_value = bot_status.get('portfolio_value', 10000.0)
        
        # Simulate price changes
        price_change = np.random.normal(0, 0.01)  # Random price change
        portfolio_value *= (1 + price_change)
        
        # Update bot status
        bot_status['portfolio_value'] = portfolio_value
        bot_status['last_update'] = datetime.now().isoformat()
        
        # Simulate balances
        bot_status['balance_usd'] = portfolio_value * 0.3  # 30% in USD
        
        # Simulate asset balances
        asset_balances = {}
        for pair in trading_pairs:
            asset = pair.split('/')[0]
            asset_balances[asset] = portfolio_value * 0.7 / len(trading_pairs)  # Evenly distribute remaining 70%
        
        bot_status['balance_assets'] = asset_balances
        
        # Simulate active strategies
        active_strategies = {}
        for pair in trading_pairs:
            # Randomly select a strategy
            strategies = ['grid_trading', 'mean_reversion', 'breakout']
            active_strategies[pair] = np.random.choice(strategies)
        
        bot_status['active_strategies'] = active_strategies
        
        # Simulate trades
        if np.random.random() < 0.2:  # 20% chance of a trade
            trade_type = np.random.choice(['buy', 'sell'])
            pair = np.random.choice(trading_pairs)
            price = np.random.uniform(100, 50000)
            amount = np.random.uniform(0.001, 0.1)
            
            trade = {
                'timestamp': datetime.now().isoformat(),
                'pair': pair,
                'type': trade_type,
                'price': price,
                'amount': amount,
                'value': price * amount,
                'strategy': active_strategies[pair]
            }
            
            # Add to recent trades
            bot_status['recent_trades'].insert(0, trade)
            
            # Keep only the most recent 20 trades
            if len(bot_status['recent_trades']) > 20:
                bot_status['recent_trades'] = bot_status['recent_trades'][:20]
            
            # Log trade in performance tracker
            performance_tracker.log_trade(
                pair, 
                1 if trade_type == 'buy' else 2, 
                amount, 
                price, 
                strategy=active_strategies[pair]
            )
            
            # Update portfolio value in performance tracker
            performance_tracker.update_portfolio_value(
                portfolio_value,
                {asset: value for asset, value in asset_balances.items()}
            )
        
        # Sleep for a bit
        time.sleep(5)

# Run the app
if __name__ == '__main__':
    # Initialize performance tracker
    init_performance_tracker()
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
