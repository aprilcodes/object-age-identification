# gathers SKU, Name or Title, Era, Description from all files
# appends data into one file as output
# ALL DONE except Microsoft DBs and masterList31911

import os
import pandas as pd
import re
import pyodbc

# get the sku out of title
def extract_number(text): 
    match = re.search(r'\b\d{4,5}\b$', str(text)) 
    if match:
        # print(f"MATCH: {int(match.group())}")
        return int(match.group())
    return None

# os.chdir(r'C:/ncf-graduate-school/semester-3/deep-learning/final-project')
os.chdir(r'C:/vvcc/archive-dump-Sept-2024/compiled-catalogs/_csvs-with-sku-name-era-description')

cwd = os.getcwd()

file_names = [file for file in os.listdir(cwd) if os.path.isfile(os.path.join(cwd, file))]
# print(f"file_names: {file_names}")

complete_df = pd.DataFrame(columns=["sku", "name", "title", "era", "description"])

for file in file_names:
    complete_file_path = os.path.join(cwd, file)
    print(f"Processing file: {complete_file_path}")

    if complete_file_path.endswith('.csv'):
        df = pd.read_csv(complete_file_path, encoding='utf-8', encoding_errors='ignore', dtype={'sku': str})
    elif complete_file_path.endswith('.xls'):
        df = pd.read_excel(complete_file_path, engine='xlrd', converters={'SKU': str})
    elif complete_file_path.endswith('.xlsx'):
        df = pd.read_excel(complete_file_path, engine='openpyxl', converters={'SKU': str})
    elif complete_file_path.endswith('.adp') or complete_file_path.endswith('.mdb'): 
        conn = pyodbc.connect(rf'Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={complete_file_path};')
        cursor = conn.cursor()
        cursor.execute("SELECT Name FROM MSysObjects WHERE Type=1 AND Flags=0;")
        tables = cursor.fetchall()
        for table in tables:
            print(table[0])  # The Name column is the first item in each row
            cursor.close()
            conn.close()
    else:
        print(f"Skipping unsupported file type: {file}")
        continue

    if file == "products_export_1.csv" or file == "products_export_2.csv": # Shopify files
        df['sku'] = df['Variant SKU']
        df.drop(columns=['Variant SKU'])

    if "Etsy" in file or "sellbrite" in file: # Etsy & Sellbrite files may have sku in Name, needs to move to sku
        #if "sellbrite" in file: # sellbrite files start with two unnecessary rows
            #print("BEFORE:")
            #print(df.iloc[3:3, :])
            # df = df.iloc[1:].reset_index(drop=True)
            # df = df[1:].reset_index(drop=True)
            # df.columns = df.iloc[0]  # set the first row as column names
            # df = df[0:].reset_index(drop=False)
            # print("AFTER:")
            # print(df.columns)
        df.columns = df.columns.str.lower()
        for index, row in df.iterrows():
            # print(f"row[name] is {row['name']}")
            name_number = extract_number(row['name']) 
            if name_number:
                if pd.isna(df.loc[index, 'sku']): # if sku is empty, replace it for both Etsy/Sellbrite files
                    df.at[index, 'sku'] = name_number
                if "sellbrite" in file:
                    if any(char.isalpha() for char in str(df.loc[index, 'sku'])): # if sku is alphanumeric, replace it in Sellbrite file
                        df.at[index, 'sku'] = name_number

    #if df['sku'].dtype(int): # make all skus strings temporarily
    #    df['sku'].astype(str)
    df = df.dropna(how='all') # get rid of empty rows to speed up processing
    df.columns = df.columns.str.lower()

    if "sku" not in df.columns: # if this particular file doesn't have any sku reference, we don't need it
        continue
    else:
        columns_to_keep = ["sku", "name", "title", "era", "description"]
        df = df[[col for col in columns_to_keep if col in df.columns]] # get whatever columns already exist, skip those that don't exist

    if df.columns.tolist() == ['sku']: # if sku is the only one in the list, move on
        continue
    else:
        df['sku'] = df['sku'].astype(str)
        df = df[~df['sku'].apply(lambda x: any(char.isalpha() for char in x))] # remove any row where sku has alphabet characters
        df['sku'] = df['sku'].apply(lambda x: int(x.split('-')[0]) if '-' in x else int(x)) # removes hyphen in sku, and anything after it

        for _, row in df.iterrows():
            sku = row['sku']
            #if sku == '340841' or sku == 340841:
            #    print("FUNKY FILE: ")
            #    print(complete_file_path)
            
            if sku in complete_df['sku'].values: # if sku exists, update rows with new info 
                for col in df.columns:
                    if pd.isna(complete_df.loc[complete_df['sku'] == sku, col].values[0]):
                        complete_df.loc[complete_df['sku'] == sku, col] = row[col]
            else:
                complete_df = pd.concat([complete_df, pd.DataFrame([row])], ignore_index=True) # make a new entry

complete_df.reset_index(drop=True, inplace=True)

complete_df.to_csv("../complete_catalog.csv", index=False)
print("complete_df saved.")