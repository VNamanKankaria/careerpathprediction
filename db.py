import sqlite3

# Establish the connection to the database
conn = sqlite3.connect('CBRSdata.db', check_same_thread=False)
c = conn.cursor()

# Function to create the table if it doesn't already exist
def create_table():
    c.execute('''CREATE TABLE IF NOT EXISTS predictiontable (
                    Name TEXT,
                    Contact_Number TEXT,  -- Changed to TEXT for better handling of phone numbers
                    Email_address TEXT,
                    Logical_quotient_rating INTEGER,
                    coding_skills_rating INTEGER,
                    hackathons INTEGER,
                    public_speaking_points INTEGER,
                    self_learning_capability TEXT,
                    Team_Worker TEXT,
                    Taken_inputs_from_seniors_or_elders TEXT,
                    worked_in_teams_ever TEXT,
                    Introvert TEXT,
                    reading_and_writing_skills TEXT,
                    memory_capability_score TEXT,
                    smart_or_hard_work TEXT,
                    Management_or_Techinical TEXT,
                    Interested_subjects TEXT,
                    Interested_Type_of_Books TEXT,
                    certifications TEXT,
                    workshops TEXT,
                    Type_of_company_want_to_settle_in TEXT,
                    interested_career_area TEXT,
              		Result TEXT , Feedback TEXT)''')
    conn.commit()

# Function to add data into the predictiontable
def add_data(Name, Contact_Number, Email_address, Logical_quotient_rating, 
             coding_skills_rating, hackathons, public_speaking_points, 
             self_learning_capability, Team_Worker, 
             Taken_inputs_from_seniors_or_elders, worked_in_teams_ever, Introvert, 
             reading_and_writing_skills, memory_capability_score, smart_or_hard_work, 
             Management_or_Techinical, Interested_subjects, Interested_Type_of_Books, 
             certifications, workshops, Type_of_company_want_to_settle_in, 
             interested_career_area,Result,Feedback):
    c.execute('''INSERT INTO predictiontable(
                    Name, Contact_Number, Email_address, Logical_quotient_rating, 
                    coding_skills_rating, hackathons, public_speaking_points, 
                    self_learning_capability, Team_Worker, 
                    Taken_inputs_from_seniors_or_elders, worked_in_teams_ever, Introvert, 
                    reading_and_writing_skills, memory_capability_score, smart_or_hard_work, 
                    Management_or_Techinical, Interested_subjects, Interested_Type_of_Books, 
                    certifications, workshops, Type_of_company_want_to_settle_in, 
                    interested_career_area,Result,Feedback) 
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
              (Name, Contact_Number, Email_address, Logical_quotient_rating, 
               coding_skills_rating, hackathons, public_speaking_points, 
               self_learning_capability, Team_Worker, 
               Taken_inputs_from_seniors_or_elders, worked_in_teams_ever, Introvert, 
               reading_and_writing_skills, memory_capability_score, smart_or_hard_work, 
               Management_or_Techinical, Interested_subjects, Interested_Type_of_Books, 
               certifications, workshops, Type_of_company_want_to_settle_in, 
               interested_career_area,Result,Feedback))
    conn.commit()

# Example usage (be sure to call create_table() first to ensure the table exists)
# create_table()
# add_data("John Doe", "1234567890", "john.doe@example.com", 120, 4, 3, 7, "High", "Yes", "Yes", "Yes", "No", "Good", "Average", "Smart", "Technical", "Math, Science", "Fiction, Fantasy", "CS101", "Workshops1", "Startup", "Software Engineering")
