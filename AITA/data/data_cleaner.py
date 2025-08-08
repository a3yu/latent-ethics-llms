import json
import re
import argparse
import random
from collections import defaultdict
from typing import Dict, List, Tuple

class AITAJudgmentParser:
    def __init__(self):
        # Define regex patterns for each judgment category
        # Using word boundaries and case insensitive matching
        self.judgment_patterns = {
            'NTA': re.compile(r'\b(NTA|YWNBTA)\b', re.IGNORECASE),
            'YTA': re.compile(r'\b(YTA|YWBTA)\b', re.IGNORECASE),
            'ESH': re.compile(r'\bESH\b', re.IGNORECASE),
            'NAH': re.compile(r'\bNAH\b', re.IGNORECASE),
            'INFO': re.compile(r'\bINFO\b', re.IGNORECASE)
        }
    
    def parse_comment_for_judgments(self, comment_body: str) -> Dict[str, bool]:
        """
        Parse a single comment body for judgment categories.
        Returns a dict indicating which judgments are present.
        """
        judgments_found = {}
        
        for judgment_type, pattern in self.judgment_patterns.items():
            judgments_found[judgment_type] = bool(pattern.search(comment_body))
        
        return judgments_found
    
    def analyze_post_comments(self, post: Dict) -> Dict[str, int]:
        """
        Analyze all comments in a single post and count judgment categories.
        """
        judgment_counts = defaultdict(int)
        
        # Get the top_comments from the post
        comments = post.get('top_comments', [])
        
        for comment in comments:
            comment_body = comment.get('body', '')
            
            # Parse this comment for judgments
            judgments_in_comment = self.parse_comment_for_judgments(comment_body)
            
            # Count each judgment type found in this comment
            for judgment_type, found in judgments_in_comment.items():
                if found:
                    judgment_counts[judgment_type] += 1
        
        return dict(judgment_counts)
    
    def analyze_all_posts(self, posts_data: List[Dict]) -> List[Dict]:
        """
        Analyze all posts in the dataset and return results.
        """
        results = []
        
        for post in posts_data:
            post_id = post.get('post_id', 'Unknown')
            title = post.get('title', 'No title')
            num_comments = post.get('num_comments', 0)
            body = post.get('selftext', '')
            
            # Analyze comments for this post
            judgment_counts = self.analyze_post_comments(post)
            
            # Filter: only include posts with more than 5 total judgments
            total_judgments = sum(judgment_counts.values())
            if total_judgments <= 5:
                continue
            
            # Create result entry
            result = {
                'post_id': post_id,
                'title': title,
                'body': body,
                'total_comments': num_comments,
                'top_comments_analyzed': len(post.get('top_comments', [])),
                'judgment_counts': judgment_counts,
                'judgment_summary': {
                    'NTA_count': judgment_counts.get('NTA', 0),
                    'YTA_count': judgment_counts.get('YTA', 0),
                    'ESH_count': judgment_counts.get('ESH', 0),
                    'NAH_count': judgment_counts.get('NAH', 0),
                    'INFO_count': judgment_counts.get('INFO', 0)
                }
            }
            
            results.append(result)
        
        return results
    
    def generate_summary_statistics(self, results: List[Dict]) -> Dict:
        """
        Generate overall summary statistics across all posts.
        """
        total_stats = defaultdict(int)
        total_posts = len(results)
        total_comments_analyzed = 0
        
        for result in results:
            total_comments_analyzed += result['top_comments_analyzed']
            for judgment_type, count in result['judgment_counts'].items():
                total_stats[judgment_type] += count
        
        return {
            'total_posts_analyzed': total_posts,
            'total_comments_analyzed': total_comments_analyzed,
            'overall_judgment_counts': dict(total_stats),
            'average_comments_per_post': total_comments_analyzed / total_posts if total_posts > 0 else 0
        }

def load_reddit_data(file_path: str) -> List[Dict]:
    """Load Reddit posts data from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return []

def save_results(results: List[Dict], summary: Dict, output_file: str = 'judgment_analysis_results.json'):
    """Save analysis results to a JSON file."""
    output_data = {
        'summary_statistics': summary,
        'post_analyses': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")

def print_summary_report(summary: Dict, results: List[Dict]):
    """Print a readable summary report to console."""
    print("\n" + "="*60)
    print("REDDIT AITA JUDGMENT ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"Total posts analyzed: {summary['total_posts_analyzed']}")
    print(f"Total comments analyzed: {summary['total_comments_analyzed']}")
    print(f"Average comments per post: {summary['average_comments_per_post']:.2f}")
    
    print("\nOverall Judgment Counts:")
    print("-" * 30)
    for judgment, count in summary['overall_judgment_counts'].items():
        percentage = (count / summary['total_comments_analyzed'] * 100) if summary['total_comments_analyzed'] > 0 else 0
        print(f"{judgment:5}: {count:4} ({percentage:.1f}%)")
    
    print("\nTop 5 Posts by Total Judgments:")
    print("-" * 50)
    # Sort posts by total judgment count
    sorted_posts = sorted(results, 
                         key=lambda x: sum(x['judgment_counts'].values()), 
                         reverse=True)[:5]
    
    for i, post in enumerate(sorted_posts, 1):
        total_judgments = sum(post['judgment_counts'].values())
        print(f"{i}. {post['title'][:50]}...")
        print(f"   Post ID: {post['post_id']}, Total judgments: {total_judgments}")
        print(f"   Breakdown: {post['judgment_summary']}")
        print()

def randomly_sample_posts(posts_data: List[Dict], sample_size: int) -> List[Dict]:
    """Randomly sample a specified number of posts from the dataset."""
    if sample_size >= len(posts_data):
        print(f"Sample size ({sample_size}) is greater than or equal to total posts ({len(posts_data)}). Using all posts.")
        return posts_data
    
    print(f"Randomly sampling {sample_size} posts from {len(posts_data)} total posts...")
    return random.sample(posts_data, sample_size)

def main():
    """Main function to run the analysis."""
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Analyze Reddit AITA posts for judgment categories')
    parser.add_argument('--random', '-r', type=int, metavar='N',
                       help='Randomly select N submissions instead of analyzing all posts')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducible sampling (optional)')
    parser.add_argument('--input', '-i', type=str, default='../reddit/reddit_top_posts.json',
                       help='Path to the input JSON file containing Reddit posts (default: ../reddit/reddit_top_posts.json)')
    
    args = parser.parse_args()
    
    # Set random seed if provided for reproducible results
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    # Initialize the parser
    judgment_parser = AITAJudgmentParser()
    
    # Load the data
    print(f"Loading Reddit AITA posts data from: {args.input}")
    posts_data = load_reddit_data(args.input)
    
    if not posts_data:
        print("No data loaded. Exiting.")
        return
    
    print(f"Loaded {len(posts_data)} posts.")
    
    # Apply random sampling if requested
    if args.random:
        posts_data = randomly_sample_posts(posts_data, args.random)
    
    # Analyze all posts
    print("Analyzing comments for judgment categories...")
    results = judgment_parser.analyze_all_posts(posts_data)
    
    # Show filtering results
    filtered_out = len(posts_data) - len(results)
    print(f"Filtered out {filtered_out} posts with 5 or fewer total judgments.")
    print(f"Analyzing {len(results)} posts with more than 5 judgments.")
    
    # Generate summary statistics
    summary = judgment_parser.generate_summary_statistics(results)
    
    # Print summary report
    print_summary_report(summary, results)
    
    # Save results to file
    save_results(results, summary)
    
    print(f"\nAnalysis complete! Detailed results saved to judgment_analysis_results.json")

if __name__ == "__main__":
    main()
