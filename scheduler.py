import schedule
import time
from realtime_data.update_mldata import update_mldata

# Schedule the update every 10 seconds
schedule.every(10).seconds.do(update_mldata)

if __name__ == "__main__":
    print("Scheduler started. Updating data every 10 seconds...")
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every second for better accuracy
