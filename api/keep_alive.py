"""
OmniDrive Render Keep-Alive Daemon.

Runs in the background and pings the Render backend every 10 minutes
to prevent the free-tier service from spinning down.
"""
import time
import requests
from datetime import datetime

ENDPOINT = "https://omnidrive.onrender.com/health"
INTERVAL_SECONDS = 10 * 60  # 10 minutes

def ping():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        t0 = time.time()
        res = requests.get(ENDPOINT, timeout=60)
        elapsed = round((time.time() - t0) * 1000, 2)
        if res.status_code == 200:
            print(f"[{now}] Success! {ENDPOINT} responded in {elapsed}ms. (Status: 200)")
        else:
            print(f"[{now}] Warning: {ENDPOINT} responded with status {res.status_code}")
    except requests.exceptions.Timeout:
        print(f"[{now}] Timeout: Server took over 60s to respond (cold-start waking up).")
    except Exception as e:
        print(f"[{now}] Error pinging server: {e}")

if __name__ == "__main__":
    print(f"OmniDrive Keep-Alive service started. Target: {ENDPOINT}")
    print(f"Pinging every {INTERVAL_SECONDS // 60} minutes. Press Ctrl+C to stop.\n")
    while True:
        ping()
        time.sleep(INTERVAL_SECONDS)
