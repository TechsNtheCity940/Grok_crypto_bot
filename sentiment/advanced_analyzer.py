import os
import re
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import praw
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

from sentiment_analyzer import SentimentAnalyzer
from config_manager import config
from utils.log_setup import logger

class AdvancedSentimentAnalyzer(SentimentAnalyzer):
    """
    Enhanced sentiment analyzer that extends the base SentimentAnalyzer
    with additional data sources and more sophisticated analysis.
    """
    def __init__(self):
        # Initialize the base class
        super().__init__()
        
        # Load more advanced sentiment model
        try:
            model_name = "finiteautomata/bertweet-base-sentiment-analysis"
            self.advanced_sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                truncation=True,
                max_length=512
            )
            logger.info("Loaded advanced sentiment model")
        except Exception as e:
            logger.error(f"Error loading advanced sentiment model: {e}")
            # Fall back to base model
            self.advanced_sentiment_pipeline = self.sentiment_pipeline
        
        # Initialize NER pipeline for entity extraction
        try:
            self.ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple"
            )
            logger.info("Loaded NER model")
        except Exception as e:
            logger.error(f"Error loading NER model: {e}")
            self.ner_pipeline = None
        
        # Cache for sentiment results to avoid redundant API calls
        self.sentiment_cache = {}
        self.cache_expiry = 3600  # 1 hour in seconds
        
        # Initialize memory for tracking sentiment trends
        self.sentiment_history = {}
    
    def analyze_social_media(self, coin_symbol, timeframe='1h'):
        """
        Enhanced social media analysis that combines multiple sources
        """
        # Check cache first
        cache_key = f"{coin_symbol}_{timeframe}"
        if cache_key in self.sentiment_cache:
            cache_time, cache_data = self.sentiment_cache[cache_key]
            if time.time() - cache_time < self.cache_expiry:
                logger.info(f"Using cached sentiment for {coin_symbol}")
                return cache_data
        
        # Analyze Reddit (from base class)
        reddit_sentiment = super().analyze_social_media(coin_symbol, timeframe)
        
        # Analyze Twitter (simulated)
        twitter_sentiment = self._analyze_twitter(coin_symbol, timeframe)
        
        # Analyze crypto news
        news_sentiment = self._analyze_crypto_news(coin_symbol, timeframe)
        
        # Combine sentiment from different sources with weighting
        combined_score = (
            reddit_sentiment['sentiment_score'] * 0.3 +
            twitter_sentiment['sentiment_score'] * 0.4 +
            news_sentiment['sentiment_score'] * 0.3
        )
        
        # Combine trending topics
        all_topics = (
            reddit_sentiment.get('trending_topics', []) +
            twitter_sentiment.get('trending_topics', []) +
            news_sentiment.get('trending_topics', [])
        )
        
        # Remove duplicates and limit to top 10
        unique_topics = list(set(all_topics))[:10]
        
        # Extract entities related to the coin
        entities = self._extract_entities(all_topics)
        
        # Track sentiment history for trend analysis
        self._update_sentiment_history(coin_symbol, combined_score)
        
        # Calculate sentiment momentum (change over time)
        sentiment_momentum = self._calculate_sentiment_momentum(coin_symbol)
        
        # Prepare result
        result = {
            'sentiment_score': combined_score,
            'trending_topics': unique_topics,
            'entities': entities,
            'sentiment_sources': {
                'reddit': reddit_sentiment['sentiment_score'],
                'twitter': twitter_sentiment['sentiment_score'],
                'news': news_sentiment['sentiment_score']
            },
            'sentiment_momentum': sentiment_momentum
        }
        
        # Cache the result
        self.sentiment_cache[cache_key] = (time.time(), result)
        
        # Store in memory
        memory_data = {
            'coin': coin_symbol,
            'sentiment': combined_score,
            'topics': unique_topics,
            'momentum': sentiment_momentum
        }
        self.memory_client.add([memory_data], user_id=f"{coin_symbol}_trader")
        
        return result
    
    def _analyze_twitter(self, coin_symbol, timeframe='1h'):
        """
        Analyze Twitter sentiment for a cryptocurrency
        Note: This is a simulation as we don't have actual Twitter API access
        """
        # Simulate Twitter data
        tweet_count = np.random.randint(50, 200)
        
        # Generate simulated tweets
        simulated_tweets = [
            f"Just bought some {coin_symbol}! To the moon! 🚀",
            f"{coin_symbol} looking bullish today",
            f"Not sure about {coin_symbol}, might sell soon",
            f"Holding my {coin_symbol} for the long term",
            f"{coin_symbol} is the future of finance",
            f"Bearish on {coin_symbol} after the latest news",
            f"Just sold all my {coin_symbol}, too volatile",
            f"Accumulating more {coin_symbol} during this dip"
        ]
        
        # Randomly select tweets based on the current time to simulate changing sentiment
        hour = datetime.now().hour
        # More positive in morning hours (8-12), more negative in evening (18-22)
        positive_bias = 0.2 if 8 <= hour <= 12 else -0.2 if 18 <= hour <= 22 else 0
        
        selected_tweets = np.random.choice(
            simulated_tweets, 
            size=min(10, len(simulated_tweets)),
            replace=False
        )
        
        # Analyze sentiment
        try:
            sentiments = self.advanced_sentiment_pipeline(selected_tweets.tolist())
            sentiment_scores = []
            
            for s in sentiments:
                if s['label'] == 'POS' or s['label'] == 'POSITIVE':
                    sentiment_scores.append(0.75 + (s['score'] * 0.25))
                elif s['label'] == 'NEG' or s['label'] == 'NEGATIVE':
                    sentiment_scores.append(0.25 - (s['score'] * 0.25))
                else:
                    sentiment_scores.append(0.5)
            
            # Apply time-based bias
            sentiment_score = np.mean(sentiment_scores) + positive_bias
            # Ensure score is between 0 and 1
            sentiment_score = max(0, min(1, sentiment_score))
            
            # Extract trending hashtags (simulated)
            trending_topics = [
                f"#{coin_symbol}Army",
                f"#{coin_symbol}ToTheMoon",
                "#Crypto",
                "#Trading",
                f"#{coin_symbol}Dip"
            ]
            
            return {
                'sentiment_score': sentiment_score,
                'trending_topics': trending_topics,
                'tweet_count': tweet_count
            }
        except Exception as e:
            logger.error(f"Error analyzing Twitter sentiment: {e}")
            return {'sentiment_score': 0.5, 'trending_topics': [], 'tweet_count': 0}
    
    def _analyze_crypto_news(self, coin_symbol, timeframe='1h'):
        """
        Analyze cryptocurrency news sentiment
        Note: This is a simulation as we don't have actual news API access
        """
        # Simulate news data
        news_count = np.random.randint(5, 20)
        
        # Generate simulated news headlines
        simulated_headlines = [
            f"{coin_symbol} Price Surges 10% After Major Partnership Announcement",
            f"Analysts Predict Bright Future for {coin_symbol}",
            f"{coin_symbol} Faces Regulatory Scrutiny in Key Markets",
            f"Major Exchange Lists {coin_symbol}, Price Jumps",
            f"Investors Cautious About {coin_symbol} After Recent Volatility",
            f"New {coin_symbol} Update Promises Enhanced Security Features",
            f"{coin_symbol} Community Grows as Adoption Increases",
            f"Market Correction Hits {coin_symbol}, Down 5%"
        ]
        
        # Randomly select headlines
        selected_headlines = np.random.choice(
            simulated_headlines, 
            size=min(5, len(simulated_headlines)),
            replace=False
        )
        
        # Analyze sentiment
        try:
            sentiments = self.advanced_sentiment_pipeline(selected_headlines.tolist())
            sentiment_scores = []
            
            for s in sentiments:
                if s['label'] == 'POS' or s['label'] == 'POSITIVE':
                    sentiment_scores.append(0.75 + (s['score'] * 0.25))
                elif s['label'] == 'NEG' or s['label'] == 'NEGATIVE':
                    sentiment_scores.append(0.25 - (s['score'] * 0.25))
                else:
                    sentiment_scores.append(0.5)
            
            sentiment_score = np.mean(sentiment_scores)
            
            # Extract trending topics from headlines
            trending_topics = []
            for headline in selected_headlines:
                words = headline.split()
                for word in words:
                    if word.startswith(coin_symbol) or word.endswith(coin_symbol):
                        trending_topics.append(word)
            
            return {
                'sentiment_score': sentiment_score,
                'trending_topics': trending_topics[:5],
                'news_count': news_count
            }
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {e}")
            return {'sentiment_score': 0.5, 'trending_topics': [], 'news_count': 0}
    
    def _extract_entities(self, texts):
        """
        Extract named entities from text using NER
        """
        if not self.ner_pipeline or not texts:
            return []
        
        try:
            # Combine texts into a single string for processing
            combined_text = " ".join(texts)
            
            # Extract entities
            entities = self.ner_pipeline(combined_text)
            
            # Filter and format entities
            formatted_entities = []
            for entity in entities:
                if entity['entity_group'] in ['ORG', 'PER', 'LOC']:
                    formatted_entities.append({
                        'text': entity['word'],
                        'type': entity['entity_group'],
                        'score': entity['score']
                    })
            
            # Remove duplicates and sort by score
            unique_entities = {}
            for entity in formatted_entities:
                key = f"{entity['text']}_{entity['type']}"
                if key not in unique_entities or entity['score'] > unique_entities[key]['score']:
                    unique_entities[key] = entity
            
            return list(unique_entities.values())
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def _update_sentiment_history(self, coin_symbol, sentiment_score):
        """
        Update sentiment history for trend analysis
        """
        if coin_symbol not in self.sentiment_history:
            self.sentiment_history[coin_symbol] = []
        
        # Add current sentiment with timestamp
        self.sentiment_history[coin_symbol].append({
            'timestamp': datetime.now(),
            'score': sentiment_score
        })
        
        # Keep only last 24 hours of data
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.sentiment_history[coin_symbol] = [
            item for item in self.sentiment_history[coin_symbol]
            if item['timestamp'] >= cutoff_time
        ]
    
    def _calculate_sentiment_momentum(self, coin_symbol):
        """
        Calculate sentiment momentum (change over time)
        """
        if coin_symbol not in self.sentiment_history or len(self.sentiment_history[coin_symbol]) < 2:
            return 0
        
        history = self.sentiment_history[coin_symbol]
        
        # Sort by timestamp
        history.sort(key=lambda x: x['timestamp'])
        
        # Calculate weighted average of recent sentiment changes
        changes = []
        weights = []
        
        for i in range(1, len(history)):
            time_diff = (history[i]['timestamp'] - history[i-1]['timestamp']).total_seconds() / 3600  # hours
            if time_diff > 0:
                score_change = history[i]['score'] - history[i-1]['score']
                changes.append(score_change)
                # More recent changes get higher weights
                recency_weight = 1 / (1 + (datetime.now() - history[i]['timestamp']).total_seconds() / 3600)
                weights.append(recency_weight)
        
        if not changes:
            return 0
        
        # Normalize weights
        weights = np.array(weights) / sum(weights) if sum(weights) > 0 else np.ones(len(weights)) / len(weights)
        
        # Calculate weighted average
        momentum = np.sum(np.array(changes) * weights)
        
        return momentum
    
    def analyze_market_sentiment(self, symbols):
        """
        Analyze overall market sentiment across multiple coins
        """
        results = {}
        overall_sentiment = 0
        
        for symbol in symbols:
            coin = symbol.split('/')[0]
            result = self.analyze_social_media(coin, '1h')
            results[coin] = result
            overall_sentiment += result['sentiment_score']
        
        # Calculate average sentiment
        market_sentiment = overall_sentiment / len(symbols) if symbols else 0.5
        
        # Determine market regime
        if market_sentiment > 0.7:
            regime = "bullish"
        elif market_sentiment < 0.3:
            regime = "bearish"
        else:
            regime = "neutral"
        
        return {
            'market_sentiment': market_sentiment,
            'market_regime': regime,
            'coin_sentiments': results
        }
