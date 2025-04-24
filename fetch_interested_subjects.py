import requests
from bs4 import BeautifulSoup
import random

# Wikipedia URL for academic subjects
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_academic_fields"

def get_interested_subjects():
    """Fetch and return a random interested subject from Wikipedia."""
    response = requests.get(WIKI_URL)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        subjects = []

        # Extract subjects from Wikipedia page
        for li in soup.select("div.mw-parser-output ul li"):
            text = li.get_text().strip()
            if len(text.split()) <= 4:  # Ensure it's a subject name, not a description
                subjects.append(text)

        if subjects:
            return random.choice(subjects)  # Pick a random subject

    return "Artificial Intelligence"
