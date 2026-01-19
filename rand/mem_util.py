#!/usr/bin/env python3
"""
System Resource Monitor
Checks system memory and disk usage and exits loudly if above thresholds.
"""

import psutil
import sys
import argparse
import time
import os


def get_memory_usage():
    """Get current memory usage as a percentage."""
    memory = psutil.virtual_memory()
    return memory.percent


def get_disk_usage(path="."):
    """Get current disk usage as a percentage for the given path."""
    disk = psutil.disk_usage(path)
    return (disk.used / disk.total) * 100


def loud_exit_memory(memory_percent, threshold):
    """Exit loudly when memory threshold is exceeded."""
    print("🚨" * 50)
    print("🚨 CRITICAL MEMORY ALERT! 🚨")
    print("🚨" * 50)
    print(f"MEMORY USAGE: {memory_percent:.2f}%")
    print(f"THRESHOLD: {threshold}%")
    print("SYSTEM MEMORY USAGE IS ABOVE ACCEPTABLE LIMITS!")
    print("🚨" * 50)
    print("EXITING IMMEDIATELY!")
    print("🚨" * 50)
    sys.exit(1)


def loud_exit_disk(disk_percent, threshold, path):
    """Exit loudly when disk threshold is exceeded."""
    print("💾" * 50)
    print("💾 CRITICAL DISK SPACE ALERT! 💾")
    print("💾" * 50)
    print(f"DISK USAGE: {disk_percent:.2f}%")
    print(f"THRESHOLD: {threshold}%")
    print(f"PATH: {os.path.abspath(path)}")
    print("DISK SPACE USAGE IS ABOVE ACCEPTABLE LIMITS!")
    print("💾" * 50)
    print("EXITING IMMEDIATELY!")
    print("💾" * 50)
    sys.exit(1)


def loud_exit_both(memory_percent, memory_threshold, disk_percent, disk_threshold, path):
    """Exit loudly when both memory and disk thresholds are exceeded."""
    print("⚠️" * 50)
    print("⚠️ CRITICAL SYSTEM RESOURCE ALERT! ⚠️")
    print("⚠️" * 50)
    print(f"MEMORY USAGE: {memory_percent:.2f}% (threshold: {memory_threshold}%)")
    print(f"DISK USAGE: {disk_percent:.2f}% (threshold: {disk_threshold}%)")
    print(f"DISK PATH: {os.path.abspath(path)}")
    print("BOTH MEMORY AND DISK USAGE ARE ABOVE ACCEPTABLE LIMITS!")
    print("⚠️" * 50)
    print("EXITING IMMEDIATELY!")
    print("⚠️" * 50)
    sys.exit(1)


def check_resources_and_exit(memory_threshold=80.0, disk_threshold=85.0, disk_path=".", verbose=False):
    """
    Check if memory and/or disk usage is above thresholds.
    
    Args:
        memory_threshold (float): Memory usage percentage threshold (default: 80%)
        disk_threshold (float): Disk usage percentage threshold (default: 85%)
        disk_path (str): Path to check disk usage for (default: current directory)
        verbose (bool): Print current usage even if below thresholds
    """
    memory_percent = get_memory_usage()
    disk_percent = get_disk_usage(disk_path)
    
    if verbose:
        print(f"Current memory usage: {memory_percent:.2f}%")
        print(f"Current disk usage: {disk_percent:.2f}% ({os.path.abspath(disk_path)})")
    
    memory_exceeded = memory_percent > memory_threshold
    disk_exceeded = disk_percent > disk_threshold
    
    if memory_exceeded and disk_exceeded:
        loud_exit_both(memory_percent, memory_threshold, disk_percent, disk_threshold, disk_path)
    elif memory_exceeded:
        loud_exit_memory(memory_percent, memory_threshold)
    elif disk_exceeded:
        loud_exit_disk(disk_percent, disk_threshold, disk_path)
    
    return memory_percent, disk_percent


def monitor_resources(memory_threshold=80.0, disk_threshold=90.0, disk_path=".", interval=5.0, max_checks=None):
    """
    Continuously monitor memory and disk usage.
    
    Args:
        memory_threshold (float): Memory usage percentage threshold
        disk_threshold (float): Disk usage percentage threshold
        disk_path (str): Path to check disk usage for
        interval (float): Check interval in seconds
        max_checks (int): Maximum number of checks (None for infinite)
    """
    check_count = 0
    
    print(f"Starting resource monitor:")
    print(f"  Memory threshold: {memory_threshold}%")
    print(f"  Disk threshold: {disk_threshold}%")
    print(f"  Disk path: {os.path.abspath(disk_path)}")
    print(f"  Check interval: {interval}s")
    
    try:
        while True:
            check_count += 1
            memory_percent, disk_percent = check_resources(
                memory_threshold, disk_threshold, disk_path, verbose=True
            )
            
            if max_checks and check_count >= max_checks:
                print(f"Completed {max_checks} checks. All resources OK.")
                break
                
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nResource monitoring stopped by user.")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Check system memory and disk usage and exit loudly if above thresholds"
    )
    parser.add_argument(
        "--memory-threshold", "-mt", 
        type=float, 
        default=80.0,
        help="Memory usage threshold percentage (default: 80.0)"
    )
    parser.add_argument(
        "--disk-threshold", "-dt",
        type=float,
        default=90.0,
        help="Disk usage threshold percentage (default: 90.0)"
    )
    parser.add_argument(
        "--disk-path", "-dp",
        type=str,
        default=".",
        help="Path to check disk usage for (default: current directory)"
    )
    parser.add_argument(
        "--monitor", "-m",
        action="store_true",
        help="Continuously monitor resource usage"
    )
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=5.0,
        help="Check interval in seconds for monitoring mode (default: 5.0)"
    )
    parser.add_argument(
        "--max-checks", "-c",
        type=int,
        help="Maximum number of checks in monitoring mode"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print current resource usage"
    )
    
    # Legacy support for old --threshold argument
    parser.add_argument(
        "--threshold", "-t", 
        type=float,
        help="Memory usage threshold percentage (deprecated, use --memory-threshold)"
    )
    
    args = parser.parse_args()
    
    # Handle legacy threshold argument
    memory_threshold = args.threshold if args.threshold is not None else args.memory_threshold
    
    if args.monitor:
        monitor_resources(
            memory_threshold, args.disk_threshold, args.disk_path, 
            args.interval, args.max_checks
        )
    else:
        memory_percent, disk_percent = check_resources(
            memory_threshold, args.disk_threshold, args.disk_path, args.verbose
        )
        if args.verbose:
            print(f"✅ Memory usage OK: {memory_percent:.2f}% (threshold: {memory_threshold}%)")
            print(f"✅ Disk usage OK: {disk_percent:.2f}% (threshold: {args.disk_threshold}%)")


if __name__ == "__main__":
    main()
