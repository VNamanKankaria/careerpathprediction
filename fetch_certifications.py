import requests
import random

def get_random_certification():
    """Fetch and return a random certification from Coursera API."""
    url = "https://api.coursera.org/api/courses.v1?q=search&query=certification"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        courses = [course["name"] for course in data.get("elements", [])]

        if courses:
            return random.choice(courses)  # Pick one random certification
    return "Certificate in Business & Operations Management Excellence"
