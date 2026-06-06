import json
import csv
import logging
from pathlib import Path
import plotly.graph_objects as go
import numpy as np
from collections import Counter
import re

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def distinct_n(text, n):
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    return len(set(ngrams)) / len(ngrams)

class BenchmarkReportGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.output_dir / "benchmark_plots"
        self.plots_dir.mkdir(exist_ok=True)
        
    def generate_report(self, judgments: list, latencies: dict):
        logger.info("Generating comprehensive benchmark report...")
        total = len(judgments)
        if total == 0: return
        
        wins = sum(1 for j in judgments if j['winner'] == 'win')
        
        metrics = {
            "Win Rate (DPO > Base)": wins / total,
            "Avg Baseline Reward": np.mean([j['baseline_reward'] for j in judgments]),
            "Avg DPO Reward": np.mean([j['dpo_reward'] for j in judgments]),
            "Reward Gain": np.mean([j['dpo_reward'] - j['baseline_reward'] for j in judgments]),
            "Avg Length (Base)": np.mean([len(j['baseline_response'].split()) for j in judgments]),
            "Avg Length (DPO)": np.mean([len(j['dpo_response'].split()) for j in judgments]),
            "Distinct-1 (Base)": np.mean([distinct_n(j['baseline_response'], 1) for j in judgments]),
            "Distinct-1 (DPO)": np.mean([distinct_n(j['dpo_response'], 1) for j in judgments]),
            "Distinct-2 (Base)": np.mean([distinct_n(j['baseline_response'], 2) for j in judgments]),
            "Distinct-2 (DPO)": np.mean([distinct_n(j['dpo_response'], 2) for j in judgments]),
            "Latency (Base)": latencies.get('base', 0.0),
            "Latency (DPO)": latencies.get('dpo', 0.0)
        }
        
        # Save JSON
        with open(self.output_dir / "benchmark_report.json", "w") as f:
            json.dump(metrics, f, indent=4)
            
        # Save CSV
        with open(self.output_dir / "benchmark_report.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            for k, v in metrics.items():
                writer.writerow([k, f"{v:.4f}"])
                
        self._generate_plots(metrics)
        logger.info("Benchmark report and plots generated successfully.")
        
    def _generate_plots(self, metrics):
        # Comparison Bar Chart
        categories = ['Avg Length', 'Distinct-1', 'Distinct-2', 'Latency (s)']
        base_vals = [metrics['Avg Length (Base)'], metrics['Distinct-1 (Base)'], metrics['Distinct-2 (Base)'], metrics['Latency (Base)']]
        dpo_vals = [metrics['Avg Length (DPO)'], metrics['Distinct-1 (DPO)'], metrics['Distinct-2 (DPO)'], metrics['Latency (DPO)']]
        
        fig = go.Figure(data=[
            go.Bar(name='Baseline', x=categories, y=base_vals, marker_color='#8b949e'),
            go.Bar(name='DPO', x=categories, y=dpo_vals, marker_color='#58a6ff')
        ])
        fig.update_layout(barmode='group', template='plotly_dark', title="Baseline vs DPO Metrics")
        fig.write_html(str(self.plots_dir / "comparison_metrics.html"))
