import pandas as pd

# Sample data as a string (replace this with reading from a file or database)
data = """
GVKEY,IID,COMPANYID,COMPANYNAME,TRADINGITEMID,EXCHANGETICKER,PRICECLOSE,ISOCOUNTRY3,SIMPLEINDUSTRYDESCRIPTION,FACTORID,FACTORNAME,FACTORVALUE,ASOFDATE
239620,01W,93982,SimCorp A/S,20211924,CPSE:SIM,84.2,DNK,Software,143,Cash to Price,0.07817458,03/01/2011
239620,01W,93982,SimCorp A/S,20211924,CPSE:SIM,84.2,DNK,Software,144,Another Factor,0.12345678,03/01/2011
239620,01W,93982,SimCorp A/S,20211924,CPSE:SIM,84.2,DNK,Software,145,Yet Another Factor,0.98765432,03/01/2011
239620,01W,93982,SimCorp A/S,20211924,CPSE:SIM,85.0,DNK,Software,143,Cash to Price,0.08000000,03/02/2011
239620,01W,93982,SimCorp A/S,20211924,CPSE:SIM,85.0,DNK,Software,144,Another Factor,0.13000000,03/02/2011
239620,01W,93982,SimCorp A/S,20211924,CPSE:SIM,85.0,DNK,Software,145,Yet Another Factor,0.99000000,03/02/2011
999999,02X,12345,Another Company,20211925,CPSE:ANOTHER,50.0,DNK,Hardware,143,Cash to Price,0.05000000,03/01/2011
999999,02X,12345,Another Company,20211925,CPSE:ANOTHER,50.0,DNK,Hardware,144,Another Factor,0.10000000,03/01/2011
999999,02X,12345,Another Company,20211925,CPSE:ANOTHER,50.0,DNK,Hardware,145,Yet Another Factor,0.90000000,03/01/2011
"""


df = pd.read_csv("2025-01-24 4_32am.csv")
# Read the data into a pandas DataFrame
# df = pd.read_csv(pd.compat.StringIO(data))

# Filter for CPSE exchange (assuming all EXCHANGETICKERs start with 'CPSE:')
df_filtered = df[df['EXCHANGETICKER'].str.startswith('CPSE:')]

# Pivot the table to organize by stock, date, and factor
df_pivot = df_filtered.pivot_table(
    index=['COMPANYNAME', 'ASOFDATE'],  # Group by stock and date
    columns='FACTORNAME',  # Factors become columns
    values='FACTORVALUE',  # Values are the factor values
    aggfunc='first'  # Use the first value if there are duplicates
).reset_index()

# Display the organized data
print(df_pivot)