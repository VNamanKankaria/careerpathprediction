import requests
import random

# Keyword-based classification of company types with weights for balancing
company_type_keywords = {
    "Startup": ["startup", "junior", "developer", "software", "engineer", "intern", "associate"],
    "MNC": ["senior", "manager", "lead", "scientist", "consultant", "executive", "director", "global"],
    "Government": ["government", "public sector", "civil", "officer", "policy", "administration"],
    "Freelance": ["freelance", "contract", "gig", "self-employed"],
    "Remote": ["remote", "work from home", "distributed", "telecommute"]
}

# Adjust probability weights (higher value = less frequent)
company_type_weights = {
    "Startup": 3,
    "MNC": 2,  # Reduce MNC dominance
    "Government": 3,
    "Freelance": 2,
    "Remote": 3
}

def classify_company_type(job_title, job_type):
    job_title = job_title.lower() if job_title else "Senior Software Engineer"
    job_type = job_type.lower() if job_type else "developer"

    matched_types = []

    # Check for keywords in job title
    for company_type, keywords in company_type_keywords.items():
        if any(keyword in job_title for keyword in keywords):
            matched_types.append(company_type)

    # If no match, classify based on job type
    if not matched_types:
        if "remote" in job_type:
            matched_types.append("Remote")
        elif "contract" in job_type:
            matched_types.append("Freelance")
        elif "government" in job_type:
            matched_types.append("Government")

    # Balance selection based on weights
    if matched_types:
        weighted_choices = [
            (company_type, company_type_weights.get(company_type, 1))
            for company_type in matched_types
        ]
        weighted_choices.sort(key=lambda x: x[1])  # Prioritize lower-weighted categories
        chosen_type = random.choices(
            [wt[0] for wt in weighted_choices], 
            weights=[wt[1] for wt in weighted_choices]
        )[0]
        return chosen_type

    return "Other"

def get_company_type():
    """Fetch and return a company type based on job trends."""
    url = "https://jsearch.p.rapidapi.com/search"
    headers = {
        "X-RapidAPI-Key": "955832e2d0msh93638f4a5766764p106cf1jsnc7918d69ed51",  # Replace with your RapidAPI key
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }
    params = {"query": "jobs", "num_pages": 2}  # Fetch more jobs for better distribution

    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if "data" in data and len(data["data"]) > 0:
            job_listings = data["data"]

            # Classify company type from multiple job listings
            company_types = [
                classify_company_type(job["job_title"], job.get("job_employment_type", ""))
                for job in job_listings
            ]

            # Remove dominant MNCs if others exist
            non_mnc_types = [ct for ct in company_types if ct != "MNC"]
            return random.choice(non_mnc_types) if non_mnc_types else "MNC"

    return "Remote"
