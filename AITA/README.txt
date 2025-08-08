LLM Ethics Experiments (AITA)
=============================

This folder contains a small suite of experiments for evaluating latent moral biases in Large Language Models using "Am I the Asshole?" (AITA) scenarios.

Contents
--------
- `data/`: utilities for preparing the dataset (produces `judgment_analysis_results.json`).
- `reddit/`: simple Reddit scraper to collect AITA posts and top comments.
- `tests/`: four experiment scripts and a runner to execute them and save results.
- `results/`: example output plots and JSON result files.

Prerequisites
-------------
1) Python 3.10+ recommended
2) Install dependencies:
   
   pip install -r requirements.txt

3) Set environment variables:

   # Required for experiments:
   export OPENAI_API_KEY="your-api-key"

   # Required for Reddit scraping:
   export REDDIT_CLIENT_ID="your-client-id"
   export REDDIT_CLIENT_SECRET="your-client-secret"
   export REDDIT_USER_AGENT="python:aita-research:v1.0 (by /u/your_username)"  # Optional

Prepare Data
------------
Option A: Use the included sample data at `reddit/reddit_top_posts.json` and run the cleaner:

cd data
python data_cleaner.py -i ../reddit/reddit_top_posts.json

This will produce `data/judgment_analysis_results.json`.

Option B: Scrape fresh data (requires Reddit API credentials set in environment variables):

cd reddit
python subreddit_scraper.py

Then run the cleaner as in Option A, pointing to the generated JSON file.

Run Experiments
---------------
From the `tests/` directory, point the runner to the cleaned data file:

cd tests
python run_all_experiments.py --data-file ../data/judgment_analysis_results.json

Useful flags:
- limit posts per experiment: `--posts 25`
- choose experiments: `--experiments 1,3`
- save outputs to project `results/`: `--output-dir ../results`

Examples:
python run_all_experiments.py --posts 25 --experiments 1,3 --data-file ../data/judgment_analysis_results.json --output-dir ../results

Run a single experiment directly (processes all posts by default):
python experiment_1_direct_judgment.py
python experiment_2_adversarial_debate.py
python experiment_3_framing_effects.py
python experiment_4_empathy_cues.py

Outputs
-------
- JSON metrics per experiment (e.g., `experiment_1_results.json`).
- Visualization PNGs per experiment (e.g., `experiment_1_results.png`).
- A summary JSON from the runner in the chosen `--output-dir`.


