import datetime 
import os

def results_dir():
    # Create a subdirectory for today's date, if that subdir doesn't already exist:
    today = str(datetime.date.today())
    results_dir = f"results/{today}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir

print(results_dir())