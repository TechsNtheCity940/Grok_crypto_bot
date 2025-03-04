import os
from transformers import pipeline
import praw
from config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, MEM0_API_KEY

# Try to import the real MemoryClient, fall back to mock if it fails
try:
    from mem0 import MemoryClient
    print("Using real MemoryClient from mem0 package")
except ImportError:
    print("Failed to import mem0 package, using mock implementation")
    from mock_mem0 import MemoryClient

class SentimentAnalyzer:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent='GrokCryptoBot/0.1'
        )
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment",
            tokenizer="cardiffnlp/twitter-roberta-base-sentiment",
            truncation=True,
            max_length=512
        )
        self.memory_client = MemoryClient(api_key=MEM0_API_KEY)

    def analyze_social_media(self, coin_symbol, timeframe='1h'):
        try:
            subreddit = self.reddit.subreddit('CryptoCurrency')
            posts = subreddit.search(f"{coin_symbol}", sort='new', time_filter='hour', limit=50)
            comments = []
            for post in posts:
                post.comments.replace_more(limit=0)
                comments.extend([comment.body for comment in post.comments.list()][:10])
            
            if not comments:
                return {'sentiment_score': 0.5, 'trending_topics': []}
            
            sentiments = self.sentiment_pipeline(comments)
            sentiment_scores = [1 if s['label'] == 'POSITIVE' else -1 if s['label'] == 'NEGATIVE' else 0 for s in sentiments]
            sentiment_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.5
            sentiment_score = (sentiment_score + 1) / 2  # Normalize to 0-1
            
            trending_topics = list(set([comment.split()[0] for comment in comments if len(comment.split()) > 0]))
            memory_data = {'coin': coin_symbol, 'sentiment': sentiment_score, 'topics': trending_topics}
            self.memory_client.add([memory_data], user_id=f"{coin_symbol}_trader")
            
            return {'sentiment_score': sentiment_score, 'trending_topics': trending_topics[:5]}
        except Exception as e:
            print(f"Error analyzing social media for {coin_symbol}: {e}")
            return {'sentiment_score': 0.5, 'trending_topics': []}
