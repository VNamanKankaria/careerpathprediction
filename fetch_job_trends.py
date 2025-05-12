import requests
import random
import time
from datetime import datetime

def get_job_trends():
    """Fetch and return a trending job title from the API."""
    url = "https://jsearch.p.rapidapi.com/search"
    headers = {
        "X-RapidAPI-Key": "5b70bfea19msh6c8af742a8261dep1f9cdcjsn887bb64e2f10",
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }
    
    queries = ["trending jobs", "latest tech jobs", "healthcare careers", 
               "finance jobs", "best engineering roles", "AI job openings"]
    
    params = {"query": random.choice(queries), "num_pages": 1}

    try:
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            job_list = data.get("data", [])
            
            if job_list:
                selected_job = random.choice(job_list).get("job_title", "Developer").split(" - ")[0] 
                print("the selected job is : " , selected_job) 
                return selected_job
            else:
                print("No job titles found in API response.")  
        
        else:
            print(f"API request failed with status code {response.status_code}")

    except Exception as e:
        print(f"Error: {e}")

    return "Developer"

# Test the function
if __name__ == "__main__":
    while True:
        get_job_trends()
        time.sleep(2)
