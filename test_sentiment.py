from yapp import SentimentAnalyzer

def main():
    # Initialize the analyzer
    analyzer = SentimentAnalyzer()
    
    # Test cases
    test_texts = [
        "I absolutely love this product! It's amazing and highly effective.",
        "This is the worst experience I've ever had. Completely useless.",
        "The product is okay, not too good or bad.",
        "I didn't expect it to be so helpful. Definitely worth it!",
        "Nothing about this was good, it was terrible and disappointing.",
        "The service was incredibly slow, but the food was good.",
        "It's kind of useful, but not worth the price.",
        ""
    ]
    
    # Analyze each text
    for i, text in enumerate(test_texts):
        try:
            result = analyzer.analyze(text)
            print(f"Test Case {i+1}:")
            print(f"Text: {text}")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Confidence: {result['confidence']}%")
            print(f"Positive Score: {result['positive_score']}")
            print(f"Negative Score: {result['negative_score']}")
            print(f"Word Count: {result['word_count']}")
            print(f"Sentiment Words: {result['sentiment_words']}")
            print("-" * 50)
        except ValueError as e:
            print(f"Error for Test Case {i+1}: {e}")
            print("-" * 50)
    
    # Display statistics
    print("Analysis Statistics:")
    print(analyzer.get_statistics())

if __name__ == "__main__":
    main()
