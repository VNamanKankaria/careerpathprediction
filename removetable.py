import sqlite3
from db import *  # Assumes create_table() is defined in db.py

# Establish the connection to the database
conn = sqlite3.connect('CBRSdata.db', check_same_thread=False)
c = conn.cursor()

# Drop the old table (if any)
c.execute("DROP TABLE IF EXISTS predictiontable")
conn.commit()

create_table()
# Recreate the table with the correct schema