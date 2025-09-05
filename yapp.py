import re
import json
from typing import Dict, List, Tuple, Optional

class SentimentResult:
    """Simple result class for sentiment analysis."""
    def __init__(self, text: str, sentiment: str, confidence: float, 
                 positive_score: float, negative_score: float, 
                 word_count: int, sentiment_words: List[Tuple[str, float]]):
        self.text = text
        self.sentiment = sentiment
        self.confidence = confidence
        self.positive_score = positive_score
        self.negative_score = negative_score
        self.word_count = word_count
        self.sentiment_words = sentiment_words

class SentimentAnalyzer:
    """Interactive sentiment analysis tool with user input support."""
    
    def __init__(self):
        """Initialize the sentiment analyzer."""
        self._initialize_lexicons()
        self.stats = {'total': 0, 'positive': 0, 'negative': 0, 'neutral': 0}
        self.history = []
    
    def _initialize_lexicons(self):
        """Initialize word lexicons."""
        self.positive_words = {
            'amazing', 'awesome', 'excellent', 'fantastic', 'great', 'wonderful', 
            'perfect', 'outstanding', 'brilliant', 'superb', 'love', 'like', 
            'enjoy', 'appreciate', 'adore', 'pleased', 'satisfied', 'happy', 
            'delighted', 'thrilled', 'good', 'best', 'better', 'nice', 'fine', 
            'beautiful', 'gorgeous', 'stunning', 'impressive', 'remarkable',
            'helpful', 'useful', 'valuable', 'beneficial', 'effective', 
            'efficient', 'reliable', 'trustworthy', 'professional', 'recommend', 
            'worth', 'quality', 'premium', 'superior', 'exceptional', 
            'incredible', 'phenomenal', 'marvelous', 'fabulous', 'terrific'
        }
        
        self.negative_words = {
            'terrible', 'awful', 'horrible', 'disgusting', 'pathetic', 'worst', 
            'hate', 'dislike', 'despise', 'loathe', 'bad', 'poor', 'disappointing', 
            'unsatisfactory', 'inadequate', 'inferior', 'useless', 'worthless', 
            'pointless', 'annoying', 'frustrating', 'irritating', 'aggravating', 
            'boring', 'dull', 'bland', 'tasteless', 'cheap', 'broken', 'damaged', 
            'defective', 'faulty', 'unreliable', 'slow', 'expensive', 'overpriced', 
            'waste', 'problem', 'issue', 'trouble', 'difficulty', 'complaint', 
            'regret', 'mistake', 'error', 'fail', 'failed', 'nasty', 'ugly'
        }
        
        self.negation_words = {
            'not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'neither', 
            'nor', 'barely', 'hardly', 'scarcely', 'seldom', 'rarely', 'without'
        }
        
        self.intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'incredibly': 1.8, 'absolutely': 1.7,
            'completely': 1.6, 'totally': 1.5, 'quite': 1.2, 'really': 1.3,
            'truly': 1.4, 'highly': 1.3, 'deeply': 1.4, 'utterly': 1.8
        }
        
        self.diminishers = {
            'slightly': 0.7, 'somewhat': 0.8, 'kind of': 0.6, 'sort of': 0.6,
            'a bit': 0.7, 'a little': 0.7, 'rather': 0.8, 'fairly': 0.8
        }
    
    def preprocess_text(self, text: str) -> List[str]:
        """Clean and tokenize text."""
        if not text or not text.strip():
            return []
        
        text = text.lower().strip()
        
        # Handle contractions
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will", "'d": " would"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:]', '', text)
        words = re.findall(r'\b\w+\b', text)
        
        return words
    
    def calculate_sentiment_score(self, words: List[str]) -> Dict:
        """Calculate sentiment scores."""
        positive_score = 0.0
        negative_score = 0.0
        negation_flag = False
        intensifier_multiplier = 1.0
        word_sentiments = []
        
        for i, word in enumerate(words):
            current_multiplier = intensifier_multiplier
            
            if word in self.negation_words:
                negation_flag = True
                continue
                
            if word in self.intensifiers:
                intensifier_multiplier = self.intensifiers[word]
                continue
            elif word in self.diminishers:
                intensifier_multiplier = self.diminishers[word]
                continue
            
            word_sentiment = 0.0
            if word in self.positive_words:
                word_sentiment = 1.0 * current_multiplier
                if negation_flag:
                    word_sentiment = -word_sentiment
                    negation_flag = False
                positive_score += max(0, word_sentiment)
                negative_score += max(0, -word_sentiment)
                
            elif word in self.negative_words:
                word_sentiment = -1.0 * current_multiplier
                if negation_flag:
                    word_sentiment = -word_sentiment
                    negation_flag = False
                positive_score += max(0, word_sentiment)
                negative_score += max(0, -word_sentiment)
            
            if word_sentiment != 0:
                word_sentiments.append((word, round(word_sentiment, 2)))
            
            intensifier_multiplier = 1.0
        
        return {
            'positive_score': round(positive_score, 2),
            'negative_score': round(negative_score, 2),
            'word_sentiments': word_sentiments,
            'total_words': len(words)
        }
    
    def classify_sentiment(self, scores: Dict) -> Tuple[str, float]:
        """Classify sentiment and calculate confidence."""
        positive_score = scores['positive_score']
        negative_score = scores['negative_score']
        total_words = scores['total_words']
        
        total_sentiment_score = positive_score + negative_score
        
        if total_sentiment_score == 0:
            sentiment = 'neutral'
            confidence = 60.0
        else:
            positive_ratio = positive_score / total_sentiment_score
            
            if positive_ratio > 0.6:
                sentiment = 'positive'
                confidence = min(95, 70 + (positive_ratio - 0.6) * 62.5)
            elif positive_ratio < 0.4:
                sentiment = 'negative'
                confidence = min(95, 70 + (0.4 - positive_ratio) * 62.5)
            else:
                sentiment = 'neutral'
                confidence = 60 + abs(positive_ratio - 0.5) * 40
        
        # Adjust confidence based on text length
        if total_words < 5:
            confidence *= 0.8
        elif total_words > 50:
            confidence = min(confidence * 1.1, 95)
        
        return sentiment, round(confidence, 2)
    
    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of input text."""
        if not text or not text.strip():
            raise ValueError("Please enter some text to analyze")
        
        words = self.preprocess_text(text)
        if not words:
            raise ValueError("No valid words found in the text")
        
        scores = self.calculate_sentiment_score(words)
        sentiment, confidence = self.classify_sentiment(scores)
        
        # Update statistics
        self.stats['total'] += 1
        self.stats[sentiment] += 1
        
        result = SentimentResult(
            text=text.strip(),
            sentiment=sentiment,
            confidence=confidence,
            positive_score=scores['positive_score'],
            negative_score=scores['negative_score'],
            word_count=len(words),
            sentiment_words=scores['word_sentiments']
        )
        
        # Add to history
        self.history.append(result)
        
        return result
    
    def print_result(self, result: SentimentResult):
        """Print formatted analysis result."""
        print("\n" + "="*60)
        print("SENTIMENT ANALYSIS RESULT")
        print("="*60)
        print(f"Text: {result.text}")
        print(f"Sentiment: {result.sentiment.upper()}")
        print(f"Confidence: {result.confidence}%")
        print(f"Word Count: {result.word_count}")
        print(f"Positive Score: {result.positive_score}")
        print(f"Negative Score: {result.negative_score}")
        
        if result.sentiment_words:
            print(f"Key Sentiment Words:")
            for word, score in result.sentiment_words:
                sentiment_type = "positive" if score > 0 else "negative"
                print(f"  - '{word}': {score} ({sentiment_type})")
        
        print("="*60)
    
    def get_statistics(self) -> Dict:
        """Get analysis statistics."""
        total = self.stats['total']
        if total == 0:
            return {**self.stats, 'positive_percentage': 0.0, 'negative_percentage': 0.0, 'neutral_percentage': 0.0}
        
        return {
            **self.stats,
            'positive_percentage': round((self.stats['positive'] / total) * 100, 2),
            'negative_percentage': round((self.stats['negative'] / total) * 100, 2),
            'neutral_percentage': round((self.stats['neutral'] / total) * 100, 2)
        }
    
    def show_statistics(self):
        """Display current statistics."""
        stats = self.get_statistics()
        print("\n" + "="*40)
        print("ANALYSIS STATISTICS")
        print("="*40)
        print(f"Total Analyses: {stats['total']}")
        print(f"Positive: {stats['positive']} ({stats['positive_percentage']}%)")
        print(f"Negative: {stats['negative']} ({stats['negative_percentage']}%)")
        print(f"Neutral: {stats['neutral']} ({stats['neutral_percentage']}%)")
        print("="*40)
    
    def show_history(self):
        """Show analysis history."""
        if not self.history:
            print("\nNo analysis history available.")
            return
        
        print(f"\n{'='*50}")
        print("ANALYSIS HISTORY")
        print(f"{'='*50}")
        
        for i, result in enumerate(self.history[-10:], 1):  # Show last 10
            print(f"{i}. {result.sentiment.upper()} ({result.confidence}%): {result.text[:50]}...")
        
        if len(self.history) > 10:
            print(f"\n... and {len(self.history) - 10} more entries")
    
    def reset_stats(self):
        """Reset statistics and history."""
        self.stats = {'total': 0, 'positive': 0, 'negative': 0, 'neutral': 0}
        self.history = []
        print("Statistics and history cleared!")

def main():
    """Main interactive function."""
    analyzer = SentimentAnalyzer()
    
    print("🎭 INTERACTIVE SENTIMENT ANALYZER")
    print("="*50)
    print("Enter text to analyze its sentiment!")
    print("Commands:")
    print("  'stats' - Show statistics")
    print("  'history' - Show recent analyses")
    print("  'reset' - Clear statistics")
    print("  'quit' or 'exit' - Exit program")
    print("="*50)
    
    while True:
        try:
            # Get user input
            user_input = input("\n💬 Enter text to analyze (or command): ").strip()
            
            if not user_input:
                print("Please enter some text or a command.")
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Thanks for using the Sentiment Analyzer!")
                break
            elif user_input.lower() == 'stats':
                analyzer.show_statistics()
                continue
            elif user_input.lower() == 'history':
                analyzer.show_history()
                continue
            elif user_input.lower() == 'reset':
                analyzer.reset_stats()
                continue
            elif user_input.lower() in ['help', 'h']:
                print("\nCommands:")
                print("  'stats' - Show statistics")
                print("  'history' - Show recent analyses")
                print("  'reset' - Clear statistics")
                print("  'quit' or 'exit' - Exit program")
                continue
            
            # Analyze the text
            result = analyzer.analyze(user_input)
            analyzer.print_result(result)
            
        except ValueError as e:
            print(f"❌ Error: {e}")
        except KeyboardInterrupt:
            print("\n👋 Thanks for using the Sentiment Analyzer!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()