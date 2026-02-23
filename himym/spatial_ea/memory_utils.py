"""
Memory monitoring utilities for tracking resource usage during experiments.

This module provides tools to:
- Monitor current memory usage
- Log memory consumption at key points
- Detect potential memory leaks
- Provide memory usage warnings
"""

import gc
import os
import psutil
from datetime import datetime
from pathlib import Path


class MemoryMonitor:
    """Monitor and log memory usage during experiments."""
    
    def __init__(self, log_file: str | None = None, warning_threshold_mb: float = 8000):
        """
        Initialize memory monitor.
        
        Args:
            log_file: Optional file path to log memory usage
            warning_threshold_mb: Memory usage threshold in MB to trigger warnings
        """
        self.process = psutil.Process(os.getpid())
        self.log_file = Path(log_file) if log_file else None
        self.warning_threshold_mb = warning_threshold_mb
        self.measurements: list[dict] = []
        
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            # Write header
            with open(self.log_file, 'w') as f:
                f.write("timestamp,context,rss_mb,vms_mb,percent,available_mb\n")
    
    def get_memory_usage(self) -> dict:
        """
        Get current memory usage statistics.
        
        Returns:
            Dictionary with memory usage metrics
        """
        memory_info = self.process.memory_info()
        virtual_memory = psutil.virtual_memory()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': self.process.memory_percent(),
            'available_mb': virtual_memory.available / 1024 / 1024,
            'total_mb': virtual_memory.total / 1024 / 1024,
            'system_percent': virtual_memory.percent
        }
    
    def log_memory(self, context: str = "", force_gc: bool = False) -> dict:
        """
        Log current memory usage with optional context.
        
        Args:
            context: Description of what's happening (e.g., "After generation 10")
            force_gc: Whether to run garbage collection before measuring
            
        Returns:
            Dictionary with memory usage metrics
        """
        if force_gc:
            gc.collect()
        
        stats = self.get_memory_usage()
        stats['timestamp'] = datetime.now().isoformat()
        stats['context'] = context
        
        self.measurements.append(stats)
        
        # Log to file if specified
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"{stats['timestamp']},{context},{stats['rss_mb']:.1f},"
                       f"{stats['vms_mb']:.1f},{stats['percent']:.2f},"
                       f"{stats['available_mb']:.1f}\n")
        
        # Check for warning threshold
        if stats['rss_mb'] > self.warning_threshold_mb:
            print(f"\n⚠️  MEMORY WARNING: Using {stats['rss_mb']:.1f} MB "
                  f"({stats['percent']:.1f}% of process limit)")
            print(f"   Context: {context}")
            print(f"   System memory available: {stats['available_mb']:.1f} MB "
                  f"({100 - stats['system_percent']:.1f}%)\n")
        
        return stats
    
    def print_summary(self) -> None:
        """Print a summary of memory usage statistics."""
        if not self.measurements:
            print("No memory measurements recorded.")
            return
        
        rss_values = [m['rss_mb'] for m in self.measurements]
        
        print("\n" + "="*60)
        print("MEMORY USAGE SUMMARY")
        print("="*60)
        print(f"Measurements: {len(self.measurements)}")
        print(f"RSS (Resident Set Size):")
        print(f"  Initial: {rss_values[0]:.1f} MB")
        print(f"  Final: {rss_values[-1]:.1f} MB")
        print(f"  Peak: {max(rss_values):.1f} MB")
        print(f"  Average: {sum(rss_values)/len(rss_values):.1f} MB")
        print(f"  Increase: {rss_values[-1] - rss_values[0]:.1f} MB")
        
        if self.log_file:
            print(f"\nDetailed log saved to: {self.log_file}")
        print("="*60 + "\n")
    
    def get_summary_dict(self) -> dict:
        """Get summary statistics as a dictionary."""
        if not self.measurements:
            return {}
        
        rss_values = [m['rss_mb'] for m in self.measurements]
        
        return {
            'num_measurements': len(self.measurements),
            'rss_initial_mb': rss_values[0],
            'rss_final_mb': rss_values[-1],
            'rss_peak_mb': max(rss_values),
            'rss_average_mb': sum(rss_values) / len(rss_values),
            'rss_increase_mb': rss_values[-1] - rss_values[0]
        }


def log_memory_usage(context: str = "") -> dict:
    """
    Standalone function to quickly log memory usage.
    
    Args:
        context: Description of current operation
        
    Returns:
        Dictionary with memory usage metrics
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    virtual_memory = psutil.virtual_memory()
    
    stats = {
        'rss_mb': memory_info.rss / 1024 / 1024,
        'vms_mb': memory_info.vms / 1024 / 1024,
        'percent': process.memory_percent(),
        'available_mb': virtual_memory.available / 1024 / 1024,
        'system_percent': virtual_memory.percent
    }
    
    print(f"[Memory] {context}: RSS={stats['rss_mb']:.1f} MB, "
          f"Available={stats['available_mb']:.1f} MB "
          f"({100 - stats['system_percent']:.1f}% free)")
    
    return stats


def check_memory_available(required_mb: float = 1000, context: str = "") -> bool:
    """
    Check if sufficient memory is available.
    
    Args:
        required_mb: Required memory in MB
        context: Description for logging
        
    Returns:
        True if sufficient memory available, False otherwise
    """
    virtual_memory = psutil.virtual_memory()
    available_mb = virtual_memory.available / 1024 / 1024
    
    if available_mb < required_mb:
        print(f"\n⚠️  LOW MEMORY WARNING")
        print(f"   Context: {context}")
        print(f"   Available: {available_mb:.1f} MB")
        print(f"   Required: {required_mb:.1f} MB")
        print(f"   Shortfall: {required_mb - available_mb:.1f} MB\n")
        return False
    
    return True


def force_cleanup(verbose: bool = True) -> dict:
    """
    Force aggressive memory cleanup.
    
    Args:
        verbose: Whether to print cleanup statistics
        
    Returns:
        Dictionary with before/after memory usage
    """
    # Get memory before cleanup
    process = psutil.Process(os.getpid())
    before_mb = process.memory_info().rss / 1024 / 1024
    
    # Run garbage collection multiple times for thorough cleanup
    for _ in range(3):
        gc.collect()
    
    # Get memory after cleanup
    after_mb = process.memory_info().rss / 1024 / 1024
    freed_mb = before_mb - after_mb
    
    if verbose:
        print(f"[Memory Cleanup] Before: {before_mb:.1f} MB, "
              f"After: {after_mb:.1f} MB, "
              f"Freed: {freed_mb:.1f} MB")
    
    return {
        'before_mb': before_mb,
        'after_mb': after_mb,
        'freed_mb': freed_mb
    }
