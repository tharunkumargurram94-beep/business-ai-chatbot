"""
Vamadeva automatic knowledge-base synchronization.

Runs the incremental website crawler immediately and then every 3 hours.
Run from the project root with:
    python app\auto_sync.py
"""

from crawler_auto_sync import run_forever


if __name__ == "__main__":
    run_forever()
