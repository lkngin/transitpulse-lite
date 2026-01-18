import requests
import sys
import os
import pandas as pd
from google.transit import gtfs_realtime_pb2

def check_gtfs_rt():
    url = "https://api.data.gov.my/gtfs-realtime/vehicle-position/prasarana/?category=rapid-bus-kl"
    print(f"Checking GTFS RT: {url}")
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        print(f"✅ Status Code: {r.status_code}")
        
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(r.content)
        print(f"✅ Successfully parsed {len(feed.entity)} entities.")
    except Exception as e:
        print(f"❌ Failed to fetch or parse GTFS RT: {e}")

def check_gtfs_static():
    url = "https://api.data.gov.my/gtfs-static/prasarana/?category=rapid-bus-kl"
    print(f"\nChecking GTFS Static: {url}")
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        print(f"✅ Status Code: {r.status_code}")
        print(f"✅ Downloaded {len(r.content)} bytes.")
    except Exception as e:
        print(f"❌ Failed to fetch GTFS Static: {e}")

def check_env():
    print("\nChecking Environment:")
    key = os.getenv("GROQ_API_KEY")
    if key:
        print(f"✅ GROQ_API_KEY is set (starts with {key[:4]}...)")
    else:
        print("⚠️ GROQ_API_KEY is NOT set in environment.")

if __name__ == "__main__":
    check_gtfs_rt()
    check_gtfs_static()
    check_env()
