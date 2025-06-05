import praw
import os
from dotenv import load_dotenv

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID_2"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET_2"),
    username=os.getenv("REDDIT_USERNAME"),
    password=os.getenv("REDDIT_PASSWORD"),
    user_agent="TestApp"
)

print(reddit.user.me())