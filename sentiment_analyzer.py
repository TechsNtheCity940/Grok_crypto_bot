import numpy as np
from transformers import pipeline
import logging
import os
from dotenv import load_dotenv
import praw
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from mem0 import MemoryClient

# Load environment variables
load_dotenv()

# Download NLTK data (run once if needed)
nltk.download('punkt')
nltk.download('stopwords')

# Reddit API credentials (replace with your own from reddit.com/prefs/apps)
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = "CryptoBot/1.0 by SwanIndividual9253"

class SentimentAnalyzer:
    def __init__(self):
        self.sentiment_pipeline = pipeline("sentiment-analysis")
        self.logger = logging.getLogger('SentimentAnalyzer')
        self.client = MemoryClient(api_key=os.getenv("MEM0_API_KEY"))
        # Initialize Reddit API with PRAW
        self.reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        self.logger.info("Initialized SentimentAnalyzer with Mem0 and Reddit API")

    def analyze_news(self, text):
        result = self.sentiment_pipeline(text)
        score = result[0]['score'] if result[0]['label'] == 'POSITIVE' else -result[0]['score']
        return score

    def get_market_sentiment(self, sources):
        sentiments = [self.analyze_news(source) for source in sources]
        return np.mean(sentiments) if sentiments else 0

    def analyze_social_media(self, symbol, timeframe):
        try:
            # Search Mem0 first for cached sentiment
            query = f"sentiment about {symbol} cryptocurrency on social media in the last {timeframe}"
            social_data = self.client.search(query, limit=10)
            
            if not social_data:
                self.logger.info(f"No cached data in Mem0 for {symbol}; fetching from Reddit")
                # Fetch posts from r/CryptoCurrency
                subreddit = self.reddit.subreddit("CryptoCurrency")
                posts = subreddit.search(f"{symbol}", limit=10, sort="new")
                texts = [post.title + " " + post.selftext for post in posts if post.selftext]
                
                if not texts:
                    self.logger.warning(f"No Reddit posts found for {symbol}; using dummy data")
                    dummy_texts = [
                        f"{symbol} is surging today due to bullish trends!",
                        f"Some traders are worried about {symbol} volatility."
                    ]
                    texts = dummy_texts
            else:
                texts = [item.get('memory', '') for item in social_data]
            
            # Process texts for sentiment and topics
            sentiments = []
            all_words = []
            stop_words = set(stopwords.words('english'))
            
            for text in texts:
                sentiment = self.analyze_news(text)
                sentiments.append(sentiment)
                # Tokenize and extract topics
                words = word_tokenize(text.lower())
                filtered_words = [word for word in words if word.isalnum() and word not in stop_words and len(word) > 3]
                all_words.extend(filtered_words)
                self.logger.debug(f"Processed sentiment for {symbol}: {sentiment} from '{text[:50]}...'")
            
            sentiment_score = np.mean(sentiments) if sentiments else 0
            volume = len(texts)
            
            # Extract trending topics
            word_freq = {}
            for word in all_words:
                word_freq[word] = word_freq.get(word, 0) + 1
            trending_topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Store in Mem0
            self.client.add(
                f"Sentiment analysis for {symbol}: score {sentiment_score}",
                metadata={"symbol": symbol, "timeframe": timeframe, "topics": [t[0] for t in trending_topics]}
            )
            
            return {
                'sentiment_score': sentiment_score,
                'volume': volume,
                'trending_topics': trending_topics,
                'source_breakdown': {'reddit': sentiment_score}
            }
        except Exception as e:
            self.logger.error(f"Error analyzing social media for {symbol}: {str(e)}")
            return {
                'sentiment_score': 0,
                'volume': 0,
                'trending_topics': [],
                'source_breakdown': {}
            }