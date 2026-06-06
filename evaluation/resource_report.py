import os
import psutil
import time
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ResourceTracker:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.peak_ram_mb = 0.0
        self.process = psutil.Process(os.getpid())
        
    def start(self):
        self.start_time = time.time()
        self.peak_ram_mb = self.process.memory_info().rss / (1024 * 1024)
        logger.info(f"Started resource tracking. Initial RAM: {self.peak_ram_mb:.2f} MB")
        
    def record_peak_ram(self):
        current_ram = self.process.memory_info().rss / (1024 * 1024)
        if current_ram > self.peak_ram_mb:
            self.peak_ram_mb = current_ram
            
    def stop_and_report(self, stage_name: str, output_dir: str):
        self.record_peak_ram()
        self.end_time = time.time()
        duration_sec = self.end_time - self.start_time
        
        # Calculate disk usage of the output directory
        disk_usage_mb = 0.0
        output_path = Path(output_dir)
        if output_path.exists():
            disk_usage_mb = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / (1024 * 1024)
            
        report = {
            "stage": stage_name,
            "training_time_seconds": duration_sec,
            "peak_ram_mb": self.peak_ram_mb,
            "disk_usage_mb": disk_usage_mb
        }
        
        logger.info(f"Resource Report [{stage_name}]: {duration_sec:.2f}s | Peak RAM: {self.peak_ram_mb:.2f}MB | Disk: {disk_usage_mb:.2f}MB")
        
        report_path = output_path / "resource_report.json"
        
        # Append if exists
        reports = []
        if report_path.exists():
            with open(report_path, "r") as f:
                reports = json.load(f)
                if not isinstance(reports, list):
                    reports = [reports]
        reports.append(report)
        
        with open(report_path, "w") as f:
            json.dump(reports, f, indent=4)
