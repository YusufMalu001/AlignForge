import numpy as np
import scipy.stats as stats
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def compute_confidence_interval(data: list, confidence: float = 0.95):
    """Computes confidence interval for numerical data."""
    a = 1.0 * np.array(data)
    n = len(a)
    if n == 0:
        return 0, 0
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m - h, m + h

def bootstrap_win_rate(judgments: list, num_iterations: int = 1000, confidence: float = 0.95):
    """Bootstrap estimate of win rate confidence interval."""
    if not judgments:
        return 0.0, 0.0, 0.0
        
    wins = [1 if j['winner'] == 'win' else 0 for j in judgments]
    n = len(wins)
    bootstrapped_rates = []
    
    for _ in range(num_iterations):
        sample = np.random.choice(wins, size=n, replace=True)
        bootstrapped_rates.append(np.mean(sample))
        
    mean_rate = np.mean(bootstrapped_rates)
    lower = np.percentile(bootstrapped_rates, ((1.0 - confidence) / 2.0) * 100)
    upper = np.percentile(bootstrapped_rates, (confidence + (1.0 - confidence) / 2.0) * 100)
    
    return mean_rate, lower, upper

def paired_comparison(base_rewards: list, tuned_rewards: list):
    """Performs a paired t-test on rewards to check for statistical significance."""
    if len(base_rewards) != len(tuned_rewards):
        logger.error("Paired comparison requires equal sized lists.")
        return None, None
        
    stat, p_value = stats.ttest_rel(tuned_rewards, base_rewards)
    return stat, p_value
