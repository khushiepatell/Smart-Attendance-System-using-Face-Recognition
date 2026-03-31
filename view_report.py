"""
view_report.py — Display and summarize attendance logs.
Usage: python view_report.py [--date 2026-03-31] [--all]
"""

import os
import csv
import argparse
from datetime import datetime
from collections import defaultdict

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "attendance_logs")


def list_logs():
    if not os.path.isdir(OUTPUT_DIR):
        print("[INFO] No attendance logs found yet.")
        return []
    return sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".csv")])


def display_log(filepath: str):
    print(f"\n{'='*60}")
    print(f"  Report: {os.path.basename(filepath)}")
    print(f"{'='*60}")

    rows = []
    try:
        with open(filepath, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {filepath}")
        return

    if not rows:
        print("  (No records in this log.)")
        return

    # Header
    print(f"  {'ID':<12} {'Name':<20} {'Date':<12} {'Time':<10} {'Conf.'}")
    print(f"  {'-'*12} {'-'*20} {'-'*12} {'-'*10} {'-'*6}")
    for row in rows:
        print(f"  {row.get('Student ID',''):<12} "
              f"{row.get('Name',''):<20} "
              f"{row.get('Date',''):<12} "
              f"{row.get('Time',''):<10} "
              f"{row.get('Confidence','')}")

    print(f"\n  Total present: {len(rows)}")


def main():
    parser = argparse.ArgumentParser(description="View attendance reports.")
    parser.add_argument("--date", type=str, default=None,
                        help="Date to view (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--all", action="store_true",
                        help="Show all available logs.")
    args = parser.parse_args()

    if args.all:
        logs = list_logs()
        if not logs:
            return
        for log in logs:
            display_log(os.path.join(OUTPUT_DIR, log))
    else:
        date_str = args.date or datetime.now().strftime("%Y-%m-%d")
        filepath = os.path.join(OUTPUT_DIR, f"attendance_{date_str}.csv")
        if not os.path.exists(filepath):
            print(f"[INFO] No attendance log found for {date_str}.")
            available = list_logs()
            if available:
                print(f"[INFO] Available logs: {', '.join(available)}")
        else:
            display_log(filepath)


if __name__ == "__main__":
    main()
