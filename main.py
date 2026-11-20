"""
Traffic Sign Recognition - Main Entry Point
Built for VAYDA SYSTEM
"""

import sys

def main():
    print("Traffic Sign Recognition System Initialized.")
    if len(sys.argv) > 1:
        print(f"Target image provided: {sys.argv[1]}")

if __name__ == "__main__":
    main()
