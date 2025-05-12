import sqlite3
import random

# Connect to SQLite database
conn = sqlite3.connect("cbrsdata.db")
cursor = conn.cursor()

# Sample data pools
yes_no = ['Yes', 'No']
rating = ['poor', 'medium', 'excellent']
smart_or_hard = ['Smart worker', 'Hard Worker']
management_or_technical = ['Management', 'Technical']
books = ['Productivity Hacks', 'Quantum Physics', 'Fictional Epics', 'Mindfulness Guide', 'Classic Literature']
certifications = ['Azure Developer Associate', 'CompTIA Security+', 'TensorFlow Developer', 'DataRobot ML', 'Docker Certified']
workshops = ['AI Workshop', 'Data Science Bootcamp', 'Ethical Hacking']
subjects = ['Data Analytics', 'Cloud Computing', 'AI Ethics', 'Edge Computing', 'Blockchain']
companies = ['Remote-first Startup', 'Unicorn Tech Company', 'Mid-size SaaS Firm', 'Global Bank IT Wing', 'NGO Tech Division']
career_areas = ['UX/UI Design', 'Cyber Forensics', 'Game Development', 'DevOps Engineer', 'AI Product Lead']
feedbacks = ['Satisfied', 'Not satisfied']

# Function to generate a single row of data
def generate_row():
    return (
        f'User{random.randint(1000,9999)}',  # Name
        str(random.randint(6000000000, 9999999999)),  # Contact_Number
        f'user{random.randint(1000,9999)}@example.com',  # Email_address
        random.randint(0, 10),  # Logical_quotient_rating
        random.randint(0, 10),  # coding_skills_rating
        random.randint(0, 10),  # hackathons
        random.randint(0, 10),  # public_speaking_points
        random.choice(yes_no),  # self_learning_capability
        random.choice(yes_no),  # Team_Worker
        random.choice(yes_no),  # Taken_inputs_from_seniors_or_elders
        random.choice(yes_no),  # worked_in_teams_ever
        random.choice(yes_no),  # Introvert
        random.choice(rating),  # reading_and_writing_skills
        random.choice(rating),  # memory_capability_score
        random.choice(smart_or_hard),  # smart_or_hard_work
        random.choice(management_or_technical),  # Management_or_Techinical
        random.choice(subjects),  # Interested_subjects
        random.choice(books),  # Interested_Type_of_Books
        random.choice(certifications),  # certifications
        random.choice(workshops),  # workshops
        random.choice(companies),  # Type_of_company_want_to_settle_in
        random.choice(career_areas),  # interested_career_area
        random.choice(career_areas),  # Result
        random.choice(feedbacks)  # Feedback
    )

# Insert 100 rows
for _ in range(100):
    row = generate_row()
    cursor.execute('''
        INSERT INTO predictiontable (
            Name, Contact_Number, Email_address, Logical_quotient_rating, coding_skills_rating,
            hackathons, public_speaking_points, self_learning_capability, Team_Worker,
            Taken_inputs_from_seniors_or_elders, worked_in_teams_ever, Introvert, reading_and_writing_skills,
            memory_capability_score, smart_or_hard_work, Management_or_Techinical, Interested_subjects,
            Interested_Type_of_Books, certifications, workshops, Type_of_company_want_to_settle_in,
            interested_career_area, Result, Feedback
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', row)

# Commit changes and close
conn.commit()
conn.close()

print("✅ Successfully inserted 100 random rows into predictiontable.")