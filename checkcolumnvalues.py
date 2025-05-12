import sqlite3
import pandas as pd

# Connect to the SQLite database
conn = sqlite3.connect('CBRSdata.db')  # Replace with your actual database path

# Query the 'predictiontable' for the corrupted rows (binary data in the 'Result' column)
query = "SELECT * FROM predictiontable WHERE Result LIKE 'b%'"
df = pd.read_sql_query(query, conn)

# Print the corrupted rows with binary data
print("Corrupted Rows with Binary Values:")
print(df)

# Remove the rows with corrupted 'Result' values
query_clean = "SELECT * FROM predictiontable WHERE Result NOT LIKE 'b%'"
df_cleaned = pd.read_sql_query(query_clean, conn)

# Print cleaned data
print("\nCleaned Data (without corrupted rows):")
print(df_cleaned['Result'].value_counts())

# Optionally, update the table to remove the corrupted rows
# Uncomment to remove corrupted rows directly in the database
# conn.execute("DELETE FROM predictiontable WHERE Result LIKE 'b%'")
# conn.commit()

# Or update the corrupted rows with a default value like 'Unknown'
# Uncomment to replace the corrupted rows with a default value
# conn.execute("UPDATE predictiontable SET Result = 'Unknown' WHERE Result LIKE 'b%'")
# conn.commit()

# Close the database connection
conn.close()

# After cleaning the data, continue with model training using the cleaned dataframe (df_cleaned)
# Example:
# X = df_cleaned.drop('Result', axis=1)
# y = df_cleaned['Result']
# Train models...