# Tesla Reputation Analysis 🚗⚡

A comprehensive dataset collection and sentiment analysis tool for Tesla's reputation tracking across Reddit, optimized for BERT-based sentiment analysis.

## 📋 Overview

This project analyzes Tesla's reputation evolution across different time periods, particularly focusing on the correlation between Elon Musk's political involvement and public perception of Tesla. The tool collects Reddit posts and comments mentioning Tesla or Elon Musk, processes them for sentiment analysis, and provides structured datasets ready for machine learning analysis.

## 🎯 Key Features

- **Smart Text Processing**: Intelligent truncation for BERT compatibility (512 token limit)
- **Dual Collection**: Extracts both posts AND relevant comments
- **Time Period Analysis**: Tracks reputation across 8 years, from 2017 to 2025
- **Multi-Subreddit Coverage**: Collects from Tesla-focused, financial, and general discussion subreddits
- **Sentiment Pre-filtering**: Identifies positive/negative indicators for enhanced analysis
- **BERT-Ready Output**: Optimized text lengths and structured format for transformer models

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Reddit API credentials (free)

### Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/tesla-reputation-analysis.git
   cd tesla-reputation-analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create Reddit API credentials**
   - Go to [Reddit Apps](https://www.reddit.com/prefs/apps)
   - Click "Create App" → "Script"
   - Note down: `client_id`, `client_secret`

4. **Setup environment variables**
   Create a `.env` file in the project root:
   ```env
   REDDIT_CLIENT_ID=your_client_id_here
   REDDIT_CLIENT_SECRET=your_client_secret_here
   REDDIT_USERNAME=your_reddit_username
   REDDIT_PASSWORD=your_reddit_password
   ```

## 📦 Dependencies

```txt
praw>=7.7.0
pandas>=1.5.0
python-dotenv>=0.19.0
transformers>=4.20.0
torch>=1.12.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

## 🔧 Usage

### Basic Data Collection
```bash
python tesla_reputation_collector.py
```

The script will:
1. Ask if you want to include comments (recommended: yes)
2. Collect data from multiple subreddits
3. Generate 3 CSV files:
   - `tesla_elon_reputation_dataset_enhanced.csv` (complete dataset)
   - `tesla_elon_reputation_dataset_enhanced_posts_only.csv`
   - `tesla_elon_reputation_dataset_enhanced_comments_only.csv`

### Advanced Usage
```python
from tesla_reputation_collector import TeslaReputationDatasetBuilder

# Initialize collector
builder = TeslaReputationDatasetBuilder()

# Custom subreddit list
custom_subreddits = ['Tesla', 'investing', 'technology']

# Collect data
data = builder.extract_multiple_subreddits(
    custom_subreddits, 
    limit_per_subreddit=500,
    include_comments=True
)

# Save dataset
df = builder.save_dataset(data, 'custom_tesla_dataset.csv')
```

## 📊 Dataset Structure

### Main Columns
| Column | Description | Type |
|--------|-------------|------|
| `type` | 'post' or 'comment' | string |
| `post_id` | Unique Reddit post ID | string |
| `parent_id` | Parent post ID (for comments) | string |
| `title` | Post title (empty for comments) | string |
| `text` | Post/comment content (truncated) | string |
| `was_truncated` | Whether text was truncated for BERT | boolean |
| `created_utc` | Unix timestamp | integer |
| `created_date` | Human-readable date | string |
| `score` | Reddit score (upvotes - downvotes) | integer |
| `author` | Reddit username | string |
| `subreddit` | Source subreddit | string |
| `period` | Time period classification | string |
| `mention_type` | 'tesla', 'elon', or 'both' | string |
| `sentiment_hint` | Pre-identified sentiment indicators | string |
| `upvote_ratio` | Ratio of upvotes (posts only) | float |

### Time Periods
- **pre_political**: 2017-2019 (Baseline Tesla reputation)
- **early_political**: 2020-2022 (Initial political involvement)
- **election_cycle**: 2023-Nov 2024 (Active election period)
- **post_election**: Nov 2024-Present (Current state)

### Target Subreddits
**High Priority** (Tesla-focused):
- r/Tesla, r/teslamotors, r/TeslaLounge, r/electricvehicles

**Medium Priority** (Financial perspective):
- r/investing, r/stocks, r/SecurityAnalysis, r/business

**Lower Priority** (General discussion):
- r/technology, r/cars, r/news

## 🤖 Sentiment Analysis with BERT

### Text Preprocessing
- **Maximum lengths**: 2000 chars (posts), 1500 chars (comments)
- **Smart truncation**: Preserves sentence boundaries when possible
- **Token optimization**: Stays within BERT's 512 token limit

### Subreddit Comparison
```python
# Compare sentiment across subreddits
subreddit_sentiment = df.groupby('subreddit')['sentiment_score'].mean().sort_values()
print("Most positive subreddits:", subreddit_sentiment.tail())
print("Most negative subreddits:", subreddit_sentiment.head())
```

## 🔍 Data Quality Features

- **Duplicate Removal**: Prevents same post collection across different sort methods
- **Rate Limiting**: Respects Reddit API limits (0.5s delay with comments, 0.1s without)
- **Error Handling**: Robust error handling with detailed logging
- **Text Validation**: Ensures all collected text mentions Tesla/Elon keywords
- **Relevance Filtering**: Only collects comments that mention Tesla/Elon

## ⚠️ Important Notes

### Rate Limits
- **With comments**: ~2-3 posts per minute (including comment processing)
- **Posts only**: ~20-30 posts per minute
- **Daily limit**: ~1000 API calls per day for free accounts

### Storage Requirements
- **Typical dataset**: 50-100MB for 5000 posts + comments
- **CSV format**: Human-readable, Excel-compatible
- **Memory usage**: ~500MB RAM during collection

### Ethical Considerations
- **Public data only**: Only collects publicly available Reddit content
- **No personal data**: Usernames are public Reddit handles
- **Research purpose**: Intended for academic/research use
- **Rate limiting**: Respects Reddit's API guidelines

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Clone your fork
git clone https://github.com/yourusername/tesla-reputation-analysis.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**⭐ If this project helps your research, please consider giving it a star!**

Made with ❤️ for Tesla reputation analysis and open science.
