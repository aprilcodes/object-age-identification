# gathers SKU, Name or Title, Era, Description, itemType from all files
# appends data into one file as output

import os
import pandas as pd
import re
import pyodbc

# get the sku out of title
def extract_number(text): 
    match = re.search(r'\b\d{4,5}\b$', str(text)) 
    if match:
        return int(match.group())
    return None

os.chdir(r'C:/vvcc/archive-dump-Sept-2024/compiled-catalogs/_csvs-with-sku-name-era-description')

cwd = os.getcwd()

file_names = [file for file in os.listdir(cwd) if os.path.isfile(os.path.join(cwd, file))]

complete_df = pd.DataFrame(columns=["sku", "name", "title", "era", "description", "itemtype"])

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
        df.columns = df.columns.str.lower()
        for index, row in df.iterrows():
            name_number = extract_number(row['name']) 
            if name_number:
                if pd.isna(df.loc[index, 'sku']): # if sku is empty, replace it for both Etsy/Sellbrite files
                    df.at[index, 'sku'] = name_number
                if "sellbrite" in file:
                    if any(char.isalpha() for char in str(df.loc[index, 'sku'])): # if sku is alphanumeric, replace it in Sellbrite file
                        df.at[index, 'sku'] = name_number
    
    # find item type category values and edit all name variations to itemtype
    columns_to_rename = ["category_name", "TypeItem", "sub_category_child", "Parent Category", "Type"] # omitting Category Name in this list
    for col in columns_to_rename:
        if col in df.columns:
            print(f"renaming {col} to itemtype")
            df.rename(columns={col: "itemtype"}, inplace=True) 
            if "itemtype" in df.columns:
                print(f"itemtype column exists and has {df['itemtype'].count()} non-empty rows.")   

    df.columns = df.columns.str.lower()

    if "sku" not in df.columns: # if this particular file doesn't have any sku reference, we don't need it
        continue
    else:
        columns_to_keep = ["sku", "name", "title", "era", "description", "itemtype"]
        df = df[[col for col in columns_to_keep if col in df.columns]] # get whatever columns already exist, skip those that don't exist

    if df.columns.tolist() == ['sku']: # if sku is the only one in the list, move on
        continue
    else:
        df['sku'] = df['sku'].astype(str)
        df = df[~df['sku'].apply(lambda x: any(char.isalpha() for char in x))] # remove any row where sku has alphabet characters
        df['sku'] = df['sku'].apply(lambda x: int(x.split('-')[0]) if '-' in x else int(x)) # removes hyphen in sku, and anything after it

        for _, row in df.iterrows():
            sku = row['sku']
            
            if sku in complete_df['sku'].values: # if sku exists, update rows with new info 
                for col in df.columns:
                    if pd.isna(complete_df.loc[complete_df['sku'] == sku, col].values[0]):
                        complete_df.loc[complete_df['sku'] == sku, col] = row[col]
            else:
                new_row_df = pd.DataFrame([row.to_dict()])
                complete_df = pd.concat([complete_df, new_row_df], ignore_index=True) # make a new entry

complete_df.reset_index(drop=True, inplace=True)

complete_df.to_csv("../complete_catalog.csv", index=False)
print("complete_df saved.")