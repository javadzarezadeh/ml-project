# Studying Predictors of Movie Revenue

This project attempts to extract features relevant to predicting movie revenue from Letterboxd reviews.

# Instructions

```
python3 sentiment-analyzer.py --input raw-data/reviews.csv --output raw-data/review-sentiments.csv
python3 sentiment-analyzer.py --input raw-data/reviews-expanded.csv --output raw-data/reviews-sentiments-expanded.csv
python3 emotion-analyzer.py --input raw-data/reviews.csv --output raw-data/review-emotions.csv
python3 emotion-analyzer.py --input raw-data/reviews-expanded.csv --output raw-data/reviews-emotions-expanded.csv
python3 emotion-analyzer.py --input raw-data/review-sentiments.csv --output raw-data/review-sentiments-and-emotions.csv
python3 emotion-analyzer.py --input raw-data/reviews-sentiments-expanded.csv --output raw-data/reviews-sentiments-and-emotions-expanded.csv
```
