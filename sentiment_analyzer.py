import numpy as np
from transformers import pipeline
from functools import lru_cache
import os


class SentimentAnalyzer:
    def __init__(self):
        self.sentiment_pipeline = pipeline("sentiment-analysis")

    def analyze_news(self, text):
        """Analyze sentiment from news articles"""
        result = self.sentiment_pipeline(text)
        self.client.log(f"Sentiment Analysis: {result}")
        return result[0]['score'] if result[0]['label'] == 'POSITIVE' else -result[0]['score']

    def get_market_sentiment(self, sources):
        """Aggregate sentiment from multiple sources"""
        sentiments = []
        for source in sources:
            sentiment = self.analyze_news(source)
            sentiments.append(sentiment)
        return np.mean(sentiments)
    @lru_cache(maxsize=128)
    def analyze_social_media(self, symbol, timeframe):
        """Analyze social media sentiment for a given crypto symbol and timeframe.
        
        Args:
            symbol (str): Cryptocurrency symbol (e.g. 'BTC', 'ETH')
            timeframe (str): Time period to analyze (e.g. '1h', '4h', '1d')
            
        Returns:
            dict: Sentiment analysis results containing:
                - sentiment_score: Overall sentiment (-1 to 1)
                - volume: Social media mention volume
                - trending_topics: Key trending topics
                - source_breakdown: Sentiment by source
        """
        try:
            # Query social media data from mem0 client
            query = f"crypto {symbol} sentiment analysis"
            social_data = client.search(query, limit=1000)
            
            # Initialize sentiment metrics
            sentiment_score = 0
            mention_volume = len(social_data)
            topics = {}
            source_sentiments = {
                'twitter': 0,
                'reddit': 0,
                'news': 0
            }
            
            # Process each social media item
            for item in social_data:
                # Calculate sentiment score (-1 to 1)
                text_sentiment = self._analyze_text_sentiment(item.text)
                sentiment_score += text_sentiment
                
                # Track source sentiment
                source = self._determine_source(item.source)
                if source in source_sentiments:
                    source_sentiments[source] += text_sentiment
                
                # Extract trending topics
                extracted_topics = self._extract_topics(item.text)
                for topic in extracted_topics:
                    topics[topic] = topics.get(topic, 0) + 1
            
            # Normalize scores
            if mention_volume > 0:
                sentiment_score /= mention_volume
                for source in source_sentiments:
                    source_sentiments[source] /= mention_volume
            
            # Get top trending topics
            trending_topics = sorted(topics.items(), 
                                  key=lambda x: x[1], 
                                  reverse=True)[:10]
            
            return {
                'sentiment_score': sentiment_score,
                'volume': mention_volume,
                'trending_topics': trending_topics,
                'source_breakdown': source_sentiments
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing social media: {str(e)}")
            return {
                'sentiment_score': 0,
                'volume': 0,
                'trending_topics': [],
                'source_breakdown': {}
            }
    
    def _analyze_text_sentiment(self, text):
        """Analyze sentiment of text using NLP.
        Returns score from -1 (negative) to 1 (positive)."""
        # Use NLTK's VADER sentiment analyzer
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            scores = analyzer.polarity_scores(text)
            return scores['compound']
        except:
            # Fallback to basic keyword analysis if NLTK fails
            positive_words = set(['bullish', 'buy', 'long', 'good', 'gr                                                                                                                                                                                                                                                                                                                                                                       w  eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeqeweeat', 'moon'])
            negative_words = set(['bearish', 'sell', 'short', 'bad', 'crash', 'dump'])
            
            text = text.lower()
            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)
            
            total = pos_count + neg_count
            if total == 0:
                return 0
            return (pos_count - neg_count) / total
    
    def _determine_source(self, source_url):
        """Determine social media source from URL."""
        if 'twitter.com' in source_url:
            return 'twitter'
        elif 'reddit.com' in source_url:
            return 'reddit'
        else:
            return 'news'
    
    def _extract_topics(self, text):
        """Extract key topics from text using basic keyword extraction."""
        # Remove common words and tokenize
        common_words = set(['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have'])
        words = text.lower().split()
        topics = [word for word in words 
                 if word not in common_words and len(word) > 3]
        
        # Return unique topics
        return list(set(topics))
