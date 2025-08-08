"""
Master Script: Run All LLM Ethics Experiments
=============================================

This script runs all four experiments in the latent ethics research suite:
1. Direct Moral Judgment
2. Adversarial Reasoning & Debate
3. Framing Effects
4. Empathy Cues

Usage:
    python run_all_experiments.py [--posts N] [--experiments EXPERIMENTS]

Arguments:
    --posts: Number of posts to analyze per experiment (default: 25)
    --experiments: Comma-separated list of experiments to run (1,2,3,4) (default: all)
    --output-dir: Directory to save results (default: results/)

Example:
    python run_all_experiments.py --posts 50 --experiments 1,3,4
"""

import argparse
import os
import sys
import time
from pathlib import Path
import json
from datetime import datetime

# Import experiment classes
from experiment_1_direct_judgment import DirectJudgmentExperiment
from experiment_2_adversarial_debate import AdversarialDebateExperiment
from experiment_3_framing_effects import FramingEffectsExperiment
from experiment_4_empathy_cues import EmpathyCuesExperiment

class ExperimentRunner:
    def __init__(self, data_file: str = 'judgment_analysis_results.json', output_dir: str = 'results'):
        """Initialize the experiment runner."""
        self.data_file = data_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Experiment configurations
        self.experiments = {
            1: {
                'name': 'Direct Moral Judgment',
                'class': DirectJudgmentExperiment,
                'description': 'Evaluates direct moral judgments by LLMs on AITA scenarios',
                'api_calls_per_post': 1
            },
            2: {
                'name': 'Adversarial Reasoning & Debate',
                'class': AdversarialDebateExperiment,
                'description': 'Tests moral argumentation through structured debate scenarios',
                'api_calls_per_post': 3  # Pro, Con, Judge
            },
            3: {
                'name': 'Framing Effects',
                'class': FramingEffectsExperiment,
                'description': 'Evaluates sensitivity to perspective framing transformations',
                'api_calls_per_post': 3  # Transform, Original judgment, Victim judgment
            },
            4: {
                'name': 'Empathy Cues',
                'class': EmpathyCuesExperiment,
                'description': 'Tests sensitivity to emotional appeals and empathy-inducing language',
                'api_calls_per_post': 3  # Enhance, Original judgment, Empathy judgment
            }
        }
        
        self.results = {}

    def check_prerequisites(self):
        """Check if all prerequisites are met."""
        # Check data file
        if not os.path.exists(self.data_file):
            print(f"❌ Error: Data file '{self.data_file}' not found.")
            print("Please ensure the AITA dataset has been processed.")
            return False
        
        # Check OpenAI API key
        if not os.getenv('OPENAI_API_KEY'):
            print("❌ Error: OPENAI_API_KEY environment variable not set.")
            print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
            return False
        
        # Check data file size
        file_size = os.path.getsize(self.data_file) / (1024 * 1024)  # MB
        print(f"✅ Data file found: {self.data_file} ({file_size:.1f} MB)")
        
        return True

    def estimate_costs(self, experiments_to_run: list, posts_per_experiment: int):
        """Estimate API costs for the experiments."""
        total_api_calls = 0
        
        print("\n📊 COST ESTIMATION")
        print("=" * 50)
        
        # Load data to get actual post count if no limit specified
        if posts_per_experiment is None:
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                if isinstance(raw_data, dict) and 'post_analyses' in raw_data:
                    total_posts = len(raw_data['post_analyses'])
                elif isinstance(raw_data, list):
                    total_posts = len(raw_data)
                else:
                    total_posts = 100  # fallback estimate
                posts_per_experiment = total_posts
                print(f"Processing all {total_posts} posts in dataset")
            except:
                posts_per_experiment = 100  # fallback estimate
                print("Using fallback estimate of 100 posts")
        else:
            print(f"Processing {posts_per_experiment} posts per experiment")
        
        for exp_num in experiments_to_run:
            exp_config = self.experiments[exp_num]
            api_calls = posts_per_experiment * exp_config['api_calls_per_post']
            total_api_calls += api_calls
            
            print(f"Experiment {exp_num}: {api_calls:,} API calls")
        
        # GPT-4o pricing (approximate)
        input_cost_per_1k = 0.005  # $0.005 per 1K input tokens
        output_cost_per_1k = 0.015  # $0.015 per 1K output tokens
        
        # Rough estimates (300 input tokens, 150 output tokens per call)
        estimated_input_tokens = total_api_calls * 300
        estimated_output_tokens = total_api_calls * 150
        
        estimated_cost = (
            (estimated_input_tokens / 1000) * input_cost_per_1k +
            (estimated_output_tokens / 1000) * output_cost_per_1k
        )
        
        print(f"\nTotal API calls: {total_api_calls:,}")
        print(f"Estimated cost: ${estimated_cost:.2f}")
        print(f"Estimated time: {total_api_calls * 2 / 60:.1f} minutes")
        print("(Estimates based on GPT-4o pricing and include rate limiting)")
        
        return estimated_cost

    def run_experiment(self, exp_num: int, posts_per_experiment: int):
        """Run a single experiment."""
        exp_config = self.experiments[exp_num]
        
        print(f"\n🔬 STARTING EXPERIMENT {exp_num}: {exp_config['name'].upper()}")
        print("=" * 70)
        print(f"Description: {exp_config['description']}")
        
        start_time = time.time()
        
        try:
            # Initialize experiment
            experiment = exp_config['class'](self.data_file)
            
            # Run experiment with appropriate parameters
            if posts_per_experiment is None:
                experiment.run_experiment()  # Process all posts
                print("Processing all posts in dataset")
            else:
                experiment.run_experiment(max_posts=posts_per_experiment)
                print(f"Processing {posts_per_experiment} posts")
            
            # Generate outputs in results directory
            output_prefix = self.output_dir / f"experiment_{exp_num}"
            
            # Save results
            experiment.save_results(f"{output_prefix}_results.json")
            
            # Create visualizations (modify to save in results directory)
            original_savefig = experiment.create_visualizations
            def save_in_results():
                import matplotlib.pyplot as plt
                original_savefig()
                # Move the generated PNG to results directory
                png_name = f"experiment_{exp_num}_results.png"
                if os.path.exists(png_name):
                    os.rename(png_name, self.output_dir / png_name)
            
            save_in_results()
            
            # Get metrics for summary
            metrics = experiment.calculate_metrics()
            
            end_time = time.time()
            duration = end_time - start_time
            
            self.results[exp_num] = {
                'name': exp_config['name'],
                'status': 'completed',
                'duration_seconds': duration,
                'metrics': metrics,
                'posts_analyzed': metrics.get('total_posts', 0)
            }
            
            print(f"✅ Experiment {exp_num} completed in {duration:.1f} seconds")
            
        except Exception as e:
            print(f"❌ Experiment {exp_num} failed: {str(e)}")
            self.results[exp_num] = {
                'name': exp_config['name'],
                'status': 'failed',
                'error': str(e)
            }

    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"experiment_summary_{timestamp}.json"
        
        summary = {
            'timestamp': timestamp,
            'data_file': self.data_file,
            'experiments_run': len(self.results),
            'results': self.results
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Print summary to console
        print("\n" + "=" * 70)
        print("EXPERIMENT SUITE SUMMARY")
        print("=" * 70)
        
        completed = sum(1 for r in self.results.values() if r['status'] == 'completed')
        failed = sum(1 for r in self.results.values() if r['status'] == 'failed')
        
        print(f"Experiments completed: {completed}")
        print(f"Experiments failed: {failed}")
        print(f"Total duration: {sum(r.get('duration_seconds', 0) for r in self.results.values()):.1f} seconds")
        
        for exp_num, result in self.results.items():
            status_symbol = "✅" if result['status'] == 'completed' else "❌"
            print(f"{status_symbol} Experiment {exp_num}: {result['name']} - {result['status']}")
            
            if result['status'] == 'completed':
                posts = result['posts_analyzed']
                duration = result['duration_seconds']
                print(f"    Posts analyzed: {posts}, Duration: {duration:.1f}s")
                
                # Print key metrics
                metrics = result['metrics']
                if 'agreement_rate' in metrics:
                    print(f"    Agreement rate: {metrics['agreement_rate']:.1%}")
                elif 'convergence_rate' in metrics:
                    print(f"    Convergence rate: {metrics['convergence_rate']:.1%}")
                elif 'framing_sensitivity_index' in metrics:
                    print(f"    Framing sensitivity: {metrics['framing_sensitivity_index']:.1%}")
                elif 'empathy_sensitivity_index' in metrics:
                    print(f"    Empathy sensitivity: {metrics['empathy_sensitivity_index']:.1%}")
        
        print(f"\nDetailed results saved to: {report_file}")
        print(f"Visualizations saved to: {self.output_dir}/")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run LLM Ethics Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--posts', 
        type=int, 
        default=None,
        help='Number of posts to analyze per experiment (default: all posts in dataset)'
    )
    
    parser.add_argument(
        '--experiments',
        type=str,
        default='1,2,3,4',
        help='Comma-separated list of experiments to run (1,2,3,4) (default: all)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save results (default: results/)'
    )
    
    parser.add_argument(
        '--data-file',
        type=str,
        default='judgment_analysis_results.json',
        help='Path to the AITA dataset file (default: judgment_analysis_results.json)'
    )
    
    args = parser.parse_args()
    
    # Parse experiments to run
    try:
        experiments_to_run = [int(x.strip()) for x in args.experiments.split(',')]
        experiments_to_run = [x for x in experiments_to_run if 1 <= x <= 4]
    except ValueError:
        print("❌ Error: Invalid experiments format. Use comma-separated numbers (1,2,3,4)")
        sys.exit(1)
    
    if not experiments_to_run:
        print("❌ Error: No valid experiments specified")
        sys.exit(1)
    
    print("🔬 LLM ETHICS EXPERIMENT SUITE")
    print("=" * 70)
    if args.posts is None:
        print("Posts per experiment: All posts in dataset")
    else:
        print(f"Posts per experiment: {args.posts}")
    print(f"Experiments to run: {experiments_to_run}")
    print(f"Output directory: {args.output_dir}")
    print(f"Data file: {args.data_file}")
    
    # Initialize runner
    runner = ExperimentRunner(args.data_file, args.output_dir)
    
    # Check prerequisites
    if not runner.check_prerequisites():
        sys.exit(1)
    
    # Estimate costs
    estimated_cost = runner.estimate_costs(experiments_to_run, args.posts)
    
    # Confirm with user
    if estimated_cost > 5.0:  # Warn for costs over $5
        response = input(f"\n⚠️  Estimated cost is ${estimated_cost:.2f}. Continue? (y/N): ")
        if response.lower() != 'y':
            print("Experiment cancelled.")
            sys.exit(0)
    
    # Run experiments
    print(f"\n🚀 Starting {len(experiments_to_run)} experiments...")
    
    for exp_num in experiments_to_run:
        runner.run_experiment(exp_num, args.posts)
    
    # Generate summary
    runner.generate_summary_report()
    
    print("\n🎉 All experiments completed!")


if __name__ == "__main__":
    main() 