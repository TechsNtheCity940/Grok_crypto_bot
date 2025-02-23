import numpy as np
from transformers import pipeline
import logging
from mem0 import MemoryClient
auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth)

class SentimentAnalyzer:
    def __init__(self):
        self.sentiment_pipeline = pipeline("sentiment-analysis")
        self.logger = logging.getLogger('SentimentAnalyzer')
        # Initialize Mem0 MemoryClient with your API key
        self.client = MemoryClient(api_key="quxAvEnPbem0-PCgGv0O8tVpOsKFnCaB4owemKlTBeC")
        self.logger.info("Initialized SentimentAnalyzer with Mem0 MemoryClient")

    def analyze_news(self, text):
        """Analyze sentiment from news articles"""
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
            # Search for relevant social media data in Mem0
            query = f"sentiment about {symbol} cryptocurrency on social media in the last {timeframe}"
            social_data = self.client.search(query, limit=10)
            
            if not social_data:
                self.logger.warning(f"No social media data found for {symbol} in Mem0; using fallback sentiment")
                dummy_texts = [f"{symbol} is trending positively!", f"Some concerns about {symbol}."]
                return {
                    'sentiment_score': self.get_market_sentiment(dummy_texts),
                    'volume': 2,
                    'trending_topics': [],
                    'source_breakdown': {}
                }
            
            # Process retrieved social media data
            sentiments = []
            for item in social_data:
                text = item.get('memory', '')  # Assuming 'memory' holds the text content
                sentiment = self.analyze_news(text)
                sentiments.append(sentiment)
                self.logger.debug(f"Processed sentiment for {symbol}: {sentiment} from '{text}'")
            
            sentiment_score = np.mean(sentiments) if sentiments else 0
            volume = len(social_data)
            
            # Store the processed sentiment back into Mem0 for future use
            self.client.add(f"Sentiment analysis for {symbol}: score {sentiment_score}", metadata={"symbol": symbol, "timeframe": timeframe})
            
            return {
                'sentiment_score': sentiment_score,
                'volume': volume,
                'trending_topics': [],  # Add topic extraction if desired
                'source_breakdown': {}  # Expand with source-specific data if available
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