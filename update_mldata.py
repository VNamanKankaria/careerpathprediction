import csv
import random
import time
from datetime import datetime
from fetch_job_trends import get_job_trends
from fetch_suggested_roles import get_suggested_roles
from fetch_certifications import get_random_certification
from fetch_workshop import get_workshop
from fetch_interested_subjects import get_interested_subjects
from fetch_company_preferences import get_company_type

# Define static columns that need random values
static_columns = {
    "Logical quotient rating": lambda: random.randint(1, 10),
    "hackathons": lambda: random.randint(0, 5),
    "coding skills rating": lambda: random.randint(1, 10),
    "public speaking points": lambda: random.randint(1, 10),
    "self-learning capability?": lambda: random.choice(["Yes", "No"]),
    "Extra-courses did": lambda: random.choice(["Yes", "No"]),
    "reading and writing skills": lambda: random.randint(1, 10),
    "memory capability score": lambda: random.randint(1, 10),
    "Taken inputs from seniors or elders": lambda: random.choice(["Yes", "No"]),
    "Interested Type of Books": lambda: random.choice(["Fiction", "Non-Fiction", "Self-Help", "Biography"]),
    "Management or Technical": lambda: random.choice(["Management", "Technical"]),
    "hard/smart worker": lambda: random.choice(["Hard", "Smart"]),
    "worked in teams ever?": lambda: random.choice(["Yes", "No"]),
    "Introvert": lambda: random.choice(["Yes", "No"])
}

# Fetch dynamic data from other scripts
def get_dynamic_data():
    return {
        "certifications": get_random_certification(),
        "workshops": get_workshop(),
        "Interested subjects": get_interested_subjects(),
        "interested career area": get_job_trends(),
        "Type of company want to settle in?": get_company_type(),
        "Suggested Job Role": get_suggested_roles()
    }

# Column order as specified
columns_order = [
    "Logical quotient rating",
    "hackathons",
    "coding skills rating",
    "public speaking points",
    "self-learning capability?",
    "Extra-courses did",
    "certifications",
    "workshops",
    "reading and writing skills",
    "memory capability score",
    "Interested subjects",
    "interested career area",
    "Type of company want to settle in?",
    "Taken inputs from seniors or elders",
    "Interested Type of Books",
    "Management or Technical",
    "hard/smart worker",
    "worked in teams ever?",
    "Introvert",
    "Suggested Job Role"
]

# Function to update mldata.csv
def update_mldata_csv():
    filename = "data/mldata.csv"

    # Generate random data
    row_data = {col: static_columns[col]() for col in static_columns}
    row_data.update(get_dynamic_data())

    # Ensure the order of columns
    row = [row_data[col] for col in columns_order]

    # Append the row to the CSV file
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(row)

    print(f"[{datetime.now()}] New row added successfully to mldata.csv")

# Run the script indefinitely
if __name__ == "__main__":
    while True:
        update_mldata_csv()
        time.sleep(0.1)  # Wait for 60 seconds before adding the next row
