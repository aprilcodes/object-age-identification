# compare sku to images, make a list of images without skus
# compare image to sku, make a list of skus without images

import os
import pandas as pd
import re

def extract_numbers_with_s(text):
    if pd.isna(text): 
        return None
    # regex to match two-digit or four-digit numbers ending with 's'
    matches = re.findall(r'\b(\d{2}|\d{4})s\b', text)
    return matches if matches else None

def add_century_to_era(match_list):
    if not match_list:  # if None or empty
        return match_list
    return [f"19{item}" if len(item) == 2 else item for item in match_list]

def process_matches(match_list):
    if not match_list:  # if None or empty
        return None
    if len(match_list) == 1: 
        return int(match_list[0])
    else:
        return max(map(int, match_list))

os.chdir(r'C:/vvcc/archive-dump-Sept-2024')
cwd = os.getcwd()

photos_folder = cwd + '/compiled-photos/_master_set_photos'

csv_filepath = cwd + '/compiled-catalogs/complete_catalog.csv'
csv_file = pd.read_csv(csv_filepath)

# find rows where 'title' has a value and 'name' is empty, replace empty name with title
csv_file.loc[csv_file['title'].notna() & csv_file['name'].isna(), 'name'] = csv_file['title']
csv_file.drop(columns=['title'], inplace=True)

# strip alphabet characters and ampersands from era if needed
csv_file['era'] = [
    ''.join(char for char in str(cell) if not (char.isalpha() or char == '&'))
    for cell in csv_file['era']
]

# take any strings that are integers only and end in "s", remove the "s"
# if there's > 1 string, take the bigger one

# extract era from title if era is empty
csv_file['matches'] = csv_file['name'].apply(extract_numbers_with_s)

# csv_file['matches'] = csv_file['description'].apply(extract_numbers_with_s)

# extract era from description if era is still empty
csv_file['matches'] = csv_file.apply(
    lambda row: extract_numbers_with_s(row['description']) if not (row['matches']) else row['matches'],
    axis=1
)

csv_file['matches'] = csv_file['matches'].apply(add_century_to_era)
csv_file['matches'] = csv_file['matches'].apply(process_matches) # keep only the later era if there's > 1

csv_file['era'] = csv_file.apply(
    lambda row: row['matches'] if (pd.isna(row['era']) or row['era'] in [None, '', []]) else row['era'],
    axis = 1
)

csv_file_columns = ['sku', 'name', 'era', 'description']
csv_file = csv_file[csv_file_columns]

# done with munging, now look at what photos vs. records are present
unmatched_files = []

# match valid filenames 
valid_pattern = re.compile(r'^v(\d{3,5})([a-zA-Z0-9]*)\.jpg$')
csv_skus = set(csv_file['sku'].dropna().astype(int)) 
# print(f"csv_skus: {csv_skus}")

# find skus that are included in photos but not in csv
for filename in os.listdir(photos_folder):
    # print(f"processing: {filename}")
    match = valid_pattern.match(filename)
    if match:
        # extract the numeric part of the filename
        image_sku = int(match.group(1))
        # print(image_sku)
        # check if the SKU exists in the CSV
        if image_sku not in csv_skus and image_sku not in unmatched_files:
            # print(f"found one: image_sku is {image_sku} and csv_skus is {csv_skus}")
            # print(f"Appending filename: {filename}")
            unmatched_files.append(image_sku) # sb image_sku not filename?
            # print(f"Unmatched files so far: {unmatched_files}")

print(f"Unmatched files: {unmatched_files}")
print(len(unmatched_files))
print(len(csv_file))

csv_file.to_csv("complete_catalog_cleaned.csv")
print("Saved to complete_catalog_cleaned.csv")