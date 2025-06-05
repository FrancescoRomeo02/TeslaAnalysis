import praw
import pandas as pd
from datetime import datetime, timezone, timedelta
import time
import os
import re
from dotenv import load_dotenv
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

import hashlib

class TeslaReputationDatasetBuilder:
    def __init__(self, max_workers=3):
        """Initialize Reddit API connection with threading support"""
        
        # Thread-local storage for Reddit instances
        self.local = threading.local()
        self.max_workers = max_workers
        self.rate_limit_lock = threading.Lock()
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum seconds between requests
        
        
        # Define time periods with more granular search strategy
        self.periods = {
            '2017': {
                'start': datetime(2017, 1, 1, tzinfo=timezone.utc),
                'end': datetime(2017, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            },
            '2018': {
                'start': datetime(2018, 1, 1, tzinfo=timezone.utc),
                'end': datetime(2018, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            },
            '2019': {
                'start': datetime(2019, 1, 1, tzinfo=timezone.utc),
                'end': datetime(2019, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            },
            '2020': {
                'start': datetime(2020, 1, 1, tzinfo=timezone.utc),
                'end': datetime(2020, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            },
            '2021': {
                'start': datetime(2021, 1, 1, tzinfo=timezone.utc),
                'end': datetime(2021, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            },
            '2022': {
                'start': datetime(2022, 1, 1, tzinfo=timezone.utc),
                'end': datetime(2022, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            },
            '2023': {
                'start': datetime(2023, 1, 1, tzinfo=timezone.utc),
                'end': datetime(2023, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            },
            '2024': {
                'start': datetime(2024, 1, 1, tzinfo=timezone.utc),
                'end': datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            },
            '2025': {
                'start': datetime(2025, 1, 1, tzinfo=timezone.utc),
                'end': datetime(2025, 5, 30, 23, 59, 59, tzinfo=timezone.utc)  # Updated to current date
            }
        }
        
        # Enhanced Tesla/Elon keywords
        self.tesla_keywords = [
            'tesla', 'model s', 'model 3', 'model x', 'model y',
            'cybertruck', 'roadster', 'semi', 'powerwall', 'solar roof',
            'autopilot', 'fsd', 'full self driving',
            'gigafactory', 'supercharger', 'tsla', 'neuralink'
        ]
        
        self.elon_keywords = [
            'elon', 'musk', 'elon musk', 'elonmusk'
        ]
    
    def get_reddit_instance(self):
        """Get thread-local Reddit instance with 3-way round-robin"""
        if not hasattr(self.local, 'reddit'):
            thread_id = threading.get_ident()
            index = thread_id % 3  # round-robin tra 3 client

            if index == 0:
                client_id = os.getenv('REDDIT_CLIENT_ID')
                client_secret = os.getenv('REDDIT_CLIENT_SECRET')
                username = os.getenv('REDDIT_USERNAME')
                password = os.getenv('REDDIT_PASSWORD')
            elif index == 1:
                client_id = os.getenv('REDDIT_CLIENT_ID_2')
                client_secret = os.getenv('REDDIT_CLIENT_SECRET_2')
                username = os.getenv('REDDIT_USERNAME')
                password = os.getenv('REDDIT_PASSWORD')
            else:
                client_id = os.getenv('REDDIT_CLIENT_ID_3')
                client_secret = os.getenv('REDDIT_CLIENT_SECRET_3')
                username = os.getenv('REDDIT_USERNAME')
                password = os.getenv('REDDIT_PASSWORD')

            self.local.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                username=username,
                password=password,
                user_agent=f'Tesla_Reputation_Analysis_v3.0 by u/{username}'
            )
        return self.local.reddit
    
    def rate_limit_delay(self):
        """Thread-safe rate limiting"""
        with self.rate_limit_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last
                time.sleep(sleep_time)
            self.last_request_time = time.time()
    
    def contains_tesla_or_elon(self, text):
        """Check if text contains Tesla or Elon mentions"""
        if not text:
            return False, None
        
        text_lower = text.lower()
        
        tesla_found = any(keyword in text_lower for keyword in self.tesla_keywords)
        elon_found = any(keyword in text_lower for keyword in self.elon_keywords)
        
        if tesla_found and elon_found:
            return True, 'both'
        elif tesla_found:
            return True, 'tesla'
        elif elon_found:
            return True, 'elon'
        else:
            return False, None
    
    def classify_period(self, timestamp):
        """Classify post into time period"""
        post_date = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        
        for period_name, period_range in self.periods.items():
            if period_range['start'] <= post_date <= period_range['end']:
                return period_name
        return 'other'
    
    def get_top_comments(self, reddit, post, limit=3):
        """Get top comments from a post with intelligent filtering"""
        try:
            post.comments.replace_more(limit=2)
            all_comments = post.comments.list()
            
            relevant_comments = []
            for comment in all_comments:
                if hasattr(comment, 'body') and comment.body and len(comment.body.strip()) > 30:
                    contains_keywords, mention_type = self.contains_tesla_or_elon(comment.body)
                    if contains_keywords:
                        relevant_comments.append({
                            'comment': comment,
                            'mention_type': mention_type,
                        })
            
            relevant_comments.sort(key=lambda x: x['comment'].score, reverse=True)
            return relevant_comments[:limit]
            
        except Exception as e:
            logger.warning(f"Error getting comments for post {post.id}: {e}")
            return []
    
    def search_by_time_period(self, subreddit_name, period_name, target_posts_per_period=50):
        """Search for posts in a specific time period using multiple strategies"""
        reddit = self.get_reddit_instance()
        subreddit = reddit.subreddit(subreddit_name)
        period_info = self.periods[period_name]
        
        posts_data = []
        comments_data = []
        processed_ids = set()
        post_hashes = set()
        
        logger.info(f"Searching r/{subreddit_name} for period {period_name}")
        
        # Strategy 1: Search with Tesla/Elon keywords for the time period
        search_queries = self.tesla_keywords + self.elon_keywords
        
        for query in search_queries:
            try:
                self.rate_limit_delay()
                
                # Search posts with the query
                search_results = subreddit.search(
                    query, 
                    sort='new', 
                    time_filter='all',
                    limit=200
                )
                
                for post in search_results:
                    if post.id in processed_ids:
                        continue
                    # Filter out low-score or low-upvote-ratio posts
                    if post.score < 1 or (post.upvote_ratio is not None and post.upvote_ratio < 0.4):
                        continue
                    if str(post.author).lower() in ['automoderator', '[deleted]', 'bot']:
                        continue
                    
                    post_date = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
                    
                    # Check if post is in our target period
                    if not (period_info['start'] <= post_date <= period_info['end']):
                        continue
                    
                    # Check if post mentions Tesla or Elon
                    full_text = f"{post.title} {post.selftext}"
                    if len(full_text.strip()) < 30:
                        continue
                    post_hash = hashlib.md5(full_text.strip().encode('utf-8')).hexdigest()
                    if post_hash in post_hashes:
                        continue
                    post_hashes.add(post_hash)
                    contains_keywords, mention_type = self.contains_tesla_or_elon(full_text)
                    
                    if not contains_keywords:
                        continue
                    
                    processed_ids.add(post.id)
                    
                    # Process post
                    post_data = self.process_post(post, subreddit_name, mention_type, 'search')
                    posts_data.append(post_data)
                    
                    # Get comments if needed
                    if post.num_comments > 0:
                        top_comments = self.get_top_comments(reddit, post, limit=4)
                        for comment_data in top_comments:
                            comment_entry = self.process_comment(
                                comment_data['comment'], 
                                post.id, 
                                subreddit_name, 
                                comment_data['mention_type'], 
                                'search'
                            )
                            comments_data.append(comment_entry)
                    
                    if len(posts_data) >= target_posts_per_period:
                        break
                
                if len(posts_data) >= target_posts_per_period:
                    break
                    
            except Exception as e:
                logger.error(f"Error searching with query '{query}': {e}")
                continue
        
        # Strategy 2: If we still need more posts, try top posts from the entire subreddit
        if len(posts_data) < target_posts_per_period:
            try:
                self.rate_limit_delay()
                top_posts = subreddit.top(time_filter='all', limit=400)
                
                for post in top_posts:
                    if post.id in processed_ids:
                        continue
                    # Filter out low-score or low-upvote-ratio posts
                    if post.score < 1 or (post.upvote_ratio is not None and post.upvote_ratio < 0.4):
                        continue
                    if str(post.author).lower() in ['automoderator', '[deleted]', 'bot']:
                        continue
                    
                    post_date = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
                    
                    if not (period_info['start'] <= post_date <= period_info['end']):
                        continue
                    
                    full_text = f"{post.title} {post.selftext}"
                    if len(full_text.strip()) < 30:
                        continue
                    post_hash = hashlib.md5(full_text.strip().encode('utf-8')).hexdigest()
                    if post_hash in post_hashes:
                        continue
                    post_hashes.add(post_hash)
                    contains_keywords, mention_type = self.contains_tesla_or_elon(full_text)
                    
                    if not contains_keywords:
                        continue
                    
                    processed_ids.add(post.id)
                    
                    post_data = self.process_post(post, subreddit_name, mention_type, 'top')
                    posts_data.append(post_data)
                    
                    if post.num_comments > 0:
                        top_comments = self.get_top_comments(reddit, post, limit=4)
                        for comment_data in top_comments:
                            comment_entry = self.process_comment(
                                comment_data['comment'], 
                                post.id, 
                                subreddit_name, 
                                comment_data['mention_type'], 
                                'top'
                            )
                            comments_data.append(comment_entry)
                    
                    if len(posts_data) >= target_posts_per_period:
                        break
                        
            except Exception as e:
                logger.error(f"Error getting top posts: {e}")
        
        logger.info(f"Found {len(posts_data)} posts and {len(comments_data)} comments for {period_name} in r/{subreddit_name}")
        return posts_data + comments_data
    
    def process_post(self, post, subreddit_name, mention_type, sort_method):
        """Process a single post into our data format"""
        max_char_len = 2000
        cleaned_title = re.sub(r"http\S+", "", post.title).strip()
        cleaned_selftext = re.sub(r"http\S+", "", post.selftext).strip()
        cleaned_selftext = cleaned_selftext[:max_char_len]
        return {
            'type': 'post',
            'post_id': post.id,
            'parent_id': None,
            'title': cleaned_title,
            'text': cleaned_selftext,
            'created_utc': post.created_utc,
            'created_date': datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
            'score': post.score,
            'num_comments': post.num_comments,
            'author': str(post.author) if post.author else '[deleted]',
            'subreddit': subreddit_name,
            'url': post.url,
            'is_self': post.is_self,
            'upvote_ratio': post.upvote_ratio,
            'period': self.classify_period(post.created_utc),
            'mention_type': mention_type,
            'sort_method': sort_method,
            'is_stickied': post.stickied,
            'is_locked': post.locked,
            'num_awards': post.total_awards_received if hasattr(post, 'total_awards_received') else 0
        }
    
    def process_comment(self, comment, post_id, subreddit_name, mention_type, sort_method):
        """Process a single comment into our data format"""
        max_char_len = 2000
        cleaned_body = re.sub(r"http\S+", "", comment.body).strip()
        cleaned_body = cleaned_body[:max_char_len]
        return {
            'type': 'comment',
            'post_id': post_id,
            'parent_id': post_id,
            'comment_id': comment.id,
            'title': '',
            'text': cleaned_body,
            'created_utc': comment.created_utc,
            'created_date': datetime.fromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
            'score': comment.score,
            'num_comments': 0,
            'author': str(comment.author) if comment.author else '[deleted]',
            'subreddit': subreddit_name,
            'url': f"https://reddit.com{comment.permalink}",
            'is_self': True,
            'upvote_ratio': None,
            'period': self.classify_period(comment.created_utc),
            'mention_type': mention_type,
            'sort_method': sort_method,
            'is_stickied': comment.stickied if hasattr(comment, 'stickied') else False,
            'is_locked': False,
            'num_awards': comment.total_awards_received if hasattr(comment, 'total_awards_received') else 0
        }
    
    def extract_subreddit_period(self, args):
        """Thread worker function for extracting data from a subreddit in a specific period"""
        subreddit_name, period_name, target_posts = args
        try:
            return self.search_by_time_period(subreddit_name, period_name, target_posts)
        except Exception as e:
            logger.error(f"Error extracting r/{subreddit_name} period {period_name}: {e}")
            return []
    
    def extract_balanced_dataset(self, subreddit_list, posts_per_period_per_subreddit=20):
        """Extract data with balanced distribution across time periods"""
        all_data = []
        
        # Create tasks for all subreddit-period combinations
        tasks = []
        for subreddit in subreddit_list:
            for period in self.periods.keys():
                if period != '2025' or datetime.now().month >= 5:  # Include 2025 data if we're past May
                    tasks.append((subreddit, period, posts_per_period_per_subreddit))
        
        # Shuffle tasks to avoid hitting the same subreddit repeatedly
        random.shuffle(tasks)
        
        logger.info(f"Starting extraction with {len(tasks)} tasks across {self.max_workers} threads")
        logger.info(f"Target: {posts_per_period_per_subreddit} posts per period per subreddit")
        
        # Execute tasks with thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {executor.submit(self.extract_subreddit_period, task): task for task in tasks}
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                subreddit, period, _ = task
                
                try:
                    result = future.result()
                    all_data.extend(result)
                    logger.info(f"Completed r/{subreddit} - {period}: {len(result)} entries")
                except Exception as e:
                    logger.error(f"Task r/{subreddit} - {period} failed: {e}")
        
        return all_data
    
    def save_dataset(self, data, filename='tesla_elon_reputation_dataset.csv'):
        """Save filtered dataset to CSV"""
        df = pd.DataFrame(data)
        
        if len(df) == 0:
            logger.error("No data to save!")
            return None
        
        # Remove posts from 'other' period
        df = df[df['period'] != 'other']
        
        
        # Save to CSV
        df.to_csv(filename, index=False, encoding='utf-8')
        logger.info(f"\nDataset saved to {filename}")
        
        # Save separate files for posts and comments
        posts_df = df[df['type'] == 'post']
        comments_df = df[df['type'] == 'comment']
        
        if len(posts_df) > 0:
            posts_filename = filename.replace('.csv', '_posts_only.csv')
            posts_df.to_csv(posts_filename, index=False, encoding='utf-8')
            logger.info(f"Posts-only dataset saved to {posts_filename}")
        
        if len(comments_df) > 0:
            comments_filename = filename.replace('.csv', '_comments_only.csv')
            comments_df.to_csv(comments_filename, index=False, encoding='utf-8')
            logger.info(f"Comments-only dataset saved to {comments_filename}")
        
        return df

def main():
    # Create dataset builder with 6 threads (to maximize use of 3 clients)
    builder = TeslaReputationDatasetBuilder(max_workers=6)
    
    # Define subreddits - prioritized for Tesla reputation analysis
    subreddits = [
        'teslamotors', 'TeslaLounge', 'electricvehicles', 'elonmusk',
        'ElonMusketeers', 'Musk',
        'technology', 'cars'
    ]
    
    logger.info("Starting BALANCED Tesla/Elon reputation dataset collection...")
    logger.info("Using multi-threading for faster collection while respecting API limits")
    logger.info("Targeting balanced distribution across all time periods (2017-2025)")
    logger.info(f"Target subreddits: {subreddits}")
    
    # Ask user for posts per period per subreddit
    try:
        posts_per_period = int(input("\nPosts per period per subreddit (default=40): ") or "40")
    except ValueError:
        posts_per_period = 70
    
    logger.info(f"Target: {posts_per_period} posts per period per subreddit")
    logger.info(f"Estimated total posts: {len(subreddits)} subreddits × 9 periods × {posts_per_period} = {len(subreddits) * 9 * posts_per_period}")
    
    # Extract data with balanced approach
    all_data = builder.extract_balanced_dataset(
        subreddits, 
        posts_per_period_per_subreddit=posts_per_period
    )
    
    if not all_data:
        logger.error("No data collected! Check your Reddit API credentials.")
        return None
    
    # Save dataset
    df = builder.save_dataset(all_data, 'tesla_elon_reputation_dataset_balanced.csv')
    
    logger.info(f"\n{'='*50}")
    logger.info("COLLECTION COMPLETE!")
    logger.info(f"{'='*50}")
    logger.info(f"Dataset ready for BERT sentiment analysis!")
    logger.info(f"Balanced temporal distribution achieved!")
    
    return df

if __name__ == "__main__":
    dataset = main()
    if dataset is not None:
        logger.info("Dataset successfully created and saved.")
        logger.info("Ready for BERT sentiment analysis!")
    else:
        logger.error("Dataset creation failed.")
        logger.error("Please check the logs for errors.")