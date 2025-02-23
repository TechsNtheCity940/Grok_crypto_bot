import numpy as np
from transformers import pipeline
import logging
import os
from dotenv import load_dotenv
import tweepy
from mem0 import MemoryClient

# Load environment variables
load_dotenv()

class SentimentAnalyzer:
    def __init__(self):
        self.sentiment_pipeline = pipeline("sentiment-analysis")
        self.logger = logging.getLogger('SentimentAnalyzer')
        # Initialize Mem0 MemoryClient
        self.client = MemoryClient(api_key=os.getenv("MEM0_API_KEY"))
        # Initialize Twitter API with Tweepy
        auth = tweepy.OAuth1UserHandler(
            os.getenv("TWITTER_CONSUMER_KEY"),
            os.getenv("TWITTER_CONSUMER_SECRET"),
            os.getenv("TWITTER_ACCESS_TOKEN"),
            os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
        )
        self.api = tweepy.API(auth, wait_on_rate_limit=True)
        self.logger.info("Initialized SentimentAnalyzer with Mem0 and Twitter API")

    def analyze_news(self, text):
        """Analyze sentiment from news articles or tweets"""
        result = self.sentiment_pipeline(text)
        score = result[0]['score'] if result[0]['label'] == 'POSITIVE' else -result[0]['score']
        return score

    def get_market_sentiment(self, sources):
        """Aggregate sentiment from multiple sources"""
        sentiments = [self.analyze_news(source) for source in sources]
        return np.mean(sentiments) if sentiments else 0

    def analyze_social_media(self, symbol, timeframe):
        """Analyze social media sentiment for a given crypto symbol and timeframe"""
        try:
            # Fetch real tweets from Twitter
            query = f"{symbol} crypto -filter:retweets"
            tweets = self.api.search_tweets(q=query, count=10, lang="en", result_type="recent")
            tweet_texts = [tweet.text for tweet in tweets]
            
            if not tweet_texts:
                self.logger.warning(f"No tweets found for {symbol}; using dummy data")
                dummy_texts = [f"{symbol} is trending positively!", f"Some concerns about {symbol}."]
                sentiment_score = self.get_market_sentiment(dummy_texts)
                volume = 2
            else:
                # Store tweets in Mem0
                for text in tweet_texts:
                    self.client.add(text, metadata={"symbol": symbol, "timeframe": timeframe})
                
                # Analyze sentiment
                sentiments = [self.analyze_news(text) for text in tweet_texts]
                sentiment_score = np.mean(sentiments) if sentiments else 0
                volume = len(tweet_texts)
                self.logger.debug(f"Processed {volume} tweets for {symbol}: sentiment_score={sentiment_score}")
            
            # Store aggregated sentiment in Mem0
            self.client.add(
                f"Sentiment analysis for {symbol}: score {sentiment_score}",
                metadata={"symbol": symbol, "timeframe": timeframe}
            )
            
            return {
                'sentiment_score': sentiment_score,
                'volume': volume,
                'trending_topics': [],  # Optional: add topic extraction later
                'source_breakdown': {'twitter': sentiment_score}
            }
        except Exception as e:
            self.logger.error(f"Error analyzing social media for {symbol}: {str(e)}")
            # Fallback to neutral dummy data on failure
            return {
                'sentiment_score': 0,
                'volume': 0,
                'trending_topics': [],
                'source_breakdown': {}
            }