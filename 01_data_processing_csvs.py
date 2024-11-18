# gathers SKU, Name or Title, Era, Description from all files
# appends data into one file as output

import os
import pandas as pd

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
    else:
        print(f"Skipping unsupported file type: {file}")
        continue

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

complete_df.to_csv("../complete_catalog_productcart.csv", index=False)
print("complete_df saved.")