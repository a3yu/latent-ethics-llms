import os
import sys
import praw
import time
import json
from tqdm import tqdm
import requests

def check_rate_limit(reddit_instance):
    """
    Check Reddit API rate limit status and wait if necessary.
    Returns rate limit information.
    """
    try:
        # Make a simple request to get rate limit headers
        response = reddit_instance._core._requestor._http.get(
            'https://oauth.reddit.com/api/v1/me',
            headers=reddit_instance._core._requestor._http.headers
        )
        
        # Extract rate limit headers
        used = response.headers.get('X-Ratelimit-Used', 'Unknown')
        remaining = response.headers.get('X-Ratelimit-Remaining', 'Unknown')
        reset = response.headers.get('X-Ratelimit-Reset', 'Unknown')
        
        print(f"Rate Limit - Used: {used}, Remaining: {remaining}, Reset in: {reset}s")
        
        # If remaining requests are low, wait
        if remaining != 'Unknown' and remaining.isdigit():
            remaining_int = int(remaining)
            if remaining_int < 10:  # Wait if less than 10 requests remaining
                reset_time = int(reset) if reset.isdigit() else 60
                print(f"⚠️  Low remaining requests ({remaining_int}). Waiting {reset_time} seconds...")
                time.sleep(reset_time + 5)  # Add 5 second buffer
                return True
        
        return False
        
    except Exception as e:
        print(f"Could not check rate limit: {e}")
        return False

def scrape_posts_by_time_filters(subreddit, posts_per_filter=1000):
    """
    Scrape posts using different time filters to get diverse content.
    """
    all_posts = []
    seen_post_ids = set()  # Track duplicates
    total_count = 0
    
    # Different time filters to scrape
    time_filters = ['all', 'year', 'month', 'week', 'day']
    
    print(f"🎯 Target: {posts_per_filter} posts per time filter")
    print(f"📅 Time filters: {', '.join(time_filters)}")
    print("🔍 Checking initial rate limit status...")
    check_rate_limit(reddit)
    
    for time_filter in time_filters:
        print(f"\n🕒 Scraping TOP posts from: {time_filter.upper()}")
        filter_count = 0
        filter_posts = []
        
        try:
            posts_generator = subreddit.top(time_filter=time_filter, limit=posts_per_filter)
            
            with tqdm(desc=f"Scraping {time_filter}", total=posts_per_filter) as pbar:
                for submission in posts_generator:
                    try:
                        # Skip if we've already seen this post
                        if submission.id in seen_post_ids:
                            pbar.update(1)
                            continue
                        
                        # Check rate limit every 50 posts
                        if total_count % 50 == 0 and total_count > 0:
                            print(f"\n📊 Rate limit check at post {total_count}:")
                            waited = check_rate_limit(reddit)
                            if waited:
                                print("✅ Resumed after waiting for rate limit reset")
                        
                        submission.comment_sort = 'top'
                        submission.comment_limit = 100
                        submission.comments.replace_more(limit=0)

                        top_comments = [
                            {
                                "comment_id": comment.id,
                                "body": comment.body,
                                "score": comment.score
                            }
                            for comment in submission.comments[:100]
                        ]

                        post_data = {
                            "post_id": submission.id,
                            "title": submission.title,
                            "selftext": submission.selftext,
                            "score": submission.score,
                            "num_comments": submission.num_comments,
                            "created_utc": submission.created_utc,
                            "url": f"https://www.reddit.com{submission.permalink}",
                            "time_filter": time_filter,  # Track which filter this came from
                            "top_comments": top_comments
                        }

                        filter_posts.append(post_data)
                        seen_post_ids.add(submission.id)
                        filter_count += 1
                        total_count += 1
                        pbar.update(1)
                    
                    except Exception as e:
                        print(f"Error on post {submission.id}: {e}")
                        # Check if it's a rate limit error
                        if "429" in str(e) or "rate" in str(e).lower():
                            print("🚫 Rate limit hit! Waiting 60 seconds...")
                            time.sleep(60)
                        else:
                            time.sleep(5)
            
            # Add this filter's posts to the main collection
            all_posts.extend(filter_posts)
            print(f"✅ {time_filter.upper()} complete: {filter_count} unique posts added")
            
            # Brief pause between time filters
            time.sleep(5)
            
        except Exception as e:
            print(f"❌ Error scraping {time_filter}: {e}")
            if "429" in str(e) or "rate" in str(e).lower():
                print("🚫 Rate limit hit! Waiting 120 seconds...")
                time.sleep(120)
            else:
                time.sleep(10)
    
    return all_posts, total_count

# Get Reddit API credentials from environment variables
client_id = os.getenv('REDDIT_CLIENT_ID')
client_secret = os.getenv('REDDIT_CLIENT_SECRET')
user_agent = os.getenv('REDDIT_USER_AGENT', 'python:aita-research:v1.0 (by /u/your_username)')

if not client_id or not client_secret:
    print("❌ Error: Reddit API credentials not found in environment variables.")
    print("Please set the following environment variables:")
    print("  export REDDIT_CLIENT_ID='your-client-id'")
    print("  export REDDIT_CLIENT_SECRET='your-client-secret'")
    print("  export REDDIT_USER_AGENT='your-user-agent'  # Optional, has default")
    sys.exit(1)

reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent,
)

subreddit = reddit.subreddit('amitheasshole')

# Configuration
POSTS_PER_FILTER = 1000  # Posts to get from each time filter

# Scrape posts from multiple time filters
all_posts, final_count = scrape_posts_by_time_filters(subreddit, POSTS_PER_FILTER)

# Final rate limit check
print("\n🏁 Final rate limit status:")
check_rate_limit(reddit)

# Save entire list as a JSON file
output_filename = f"reddit_top_posts_{final_count}.json"
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(all_posts, f, ensure_ascii=False, indent=2)

print(f"✅ Finished! Saved {final_count} unique posts to {output_filename}")
print(f"📊 Breakdown by time filter:")
filter_counts = {}
for post in all_posts:
    filter_name = post['time_filter']
    filter_counts[filter_name] = filter_counts.get(filter_name, 0) + 1

for filter_name, count in filter_counts.items():
    print(f"   {filter_name.upper()}: {count} posts")