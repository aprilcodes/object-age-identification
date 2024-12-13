# compare sku to images, make separate lists of images & skus 
# clean up data in catalog
# remove non-women's items: men's, children's, sewing patterns, home goods
# locate items were imported without itemtypes, type them (this got involved!) 
# tried using an AI assistant to make this easier through Llama
# it couldn't reliably access Google Sheets and/or couldn't save its results to Sheets

import os
import pandas as pd
import re
import shutil

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

def map_itemtype(value):
    if pd.isna(value):  
        return value, "no"
    for key, replacement in itemtype_dict.items():
        if key in value.lower():  
            return replacement, "x"  # replace value and mark 'typed' column
    return value, "no"

def map_nametypes(row):
    if row['typed'] == "x":
        return row['itemtype'], row['typed']
    name_value = row['name']
    if pd.isna(name_value): 
        return row['itemtype'], row['typed']
    name_value = str(name_value).lower()

    for key, value_list in nametypes_dict.items():
        for value in value_list:
            if value in name_value: 
                return key, "x" 
    return row['itemtype'], row['typed']

os.chdir(r'C:/vvcc/archive-dump-Sept-2024')
cwd = os.getcwd()

photos_folder = cwd + '/compiled-photos/_master_set_photos'

output_folder = cwd + '/compiled-photos/unmatched-photo-subset'
os.makedirs(output_folder, exist_ok=True)

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

# strip semi-colons from description
csv_file['description'] = csv_file['description'].str.replace(r'\n|\t|;', ' ', regex=True)

# strip dashes and ampersands, substitute with "and"
csv_file['name'] = (
    csv_file['name']
    .str.replace('-', '', regex=False)
    .str.replace('&', 'and', regex=False)
)

# take any strings that are integers only and end in "s", remove the "s"
# if there's > 1 string, take the bigger one

# extract era from title if era is empty
csv_file['matches'] = csv_file['name'].apply(extract_numbers_with_s)

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

# extract itemtype from same column, or (if needed) from name column
csv_file['name'] = csv_file['name'].str.lower()
csv_file['itemtype'] = csv_file['itemtype'].str.lower()

csv_file = csv_file[~csv_file['itemtype'].str.contains("men|children|girl|boy|infant|toys|supplies|collectibles|watches|living", case=False, na=False)] # exclude rows with men, children, non-garments in itemtype
csv_file = csv_file[~csv_file['name'].str.contains("nan|sewing pattern| men's|men's small|men's suit|black wool tuxedo|tuxedo jacket|tux jacket|smoking jacketx|tuxedox|dinner jacketx|leisure suitx|silk jacketx|newsboy capx|metal collar|smoking jacket|sailor uniform|boxers|spanish leather jacket|the airman|cummerbund|pig buckle|size small|size medium|size large|size xl|men small|baby girl|toddler|baby romper|baby shoe|boy's|girl's|kids'|infant|layette|baby girl|christening|baby dress|baby's top|diaper|floatie|boy scout|2t|3t|4t|months|money clip|fur clip|hair rat|battenburg lace collar|jabot|eyeglass chain|50s vanity gift|yards|tablecloth|suitcase|toiletry|paperback|hardback|towel|cufflinks|keychain|brag book|flagon|barrette|hair jewel|hair clip|hair pins|hair ornament|hair accent|bobby pins|binoculars|charm with carnelian|cuff links", case=False, na=False)] # exclude rows with men's/children's garments in name
csv_file = csv_file[~csv_file['description'].str.contains("little girl's dress|kid's tank|baby's overall", case=False, na=False)]
csv_file = csv_file[~csv_file['name'].str.contains("child", case=False, na=False)] # exclude rows with child in name

csv_file['typed'] = 'no'

csv_file['name'] = csv_file['name'].astype(str)
csv_file['typed'] = csv_file['typed'].astype(str)
csv_file['description'] = csv_file['description'].astype(str)
csv_file['typed'] = csv_file['typed'].fillna('').astype(str)

# this dictionary (1 of 2) uses values in itemtype to standardize item types 
itemtype_dict = {"bsuit": "swimsuit",
                 "access glv": "gloves",
                 "access shoe": "shoes",
                 "shoes": "shoes",
                 "access purse": "purse",
                 "access set misc purse": "purse",
                 "access scarf": "scarf",
                 "access hat": "hat",
                 "hats": "hat",
                 "access hose": "hosiery",
                 "access hosiery": "hosiery",
                 "access belt": "belt",
                 "access misc belt": "belt",
                 "access misc belt": "belt",
                 "accessories > belts & suspenders > belts": "belt",
                 "jewelry > bracelets": "bracelet",
                 "bott sep skt": "skirt",
                 "bott sep pant": "pant",
                 "pants": "pant",
                 "shorts": "shorts",
                 "sun dr": "dress",
                 "dd": "dress",
                 "day dresses": "dress",
                 "ctail": "dress",
                 "von": "dress",
                 "eve": "dress",
                 "mini": "dress",
                 "swt card": "sweater",
                 "coat outerwear": "coat",
                 "coats & outerwear": "coat",
                 "reg sht": "shirt",
                 "retro sht": "shirt",
                 "shirts": "shirt",
                 "overlay": "shirt",
                 "apron": "apron",
                 "ptsuit": "pantsuit",
                 "pant outfits": "pantsuit",
                 "pant suits": "pantsuit",
                 "wmn bus": "business",
                 "business wear": "business",
                 "business & office": "business",
                 "necklaces": "necklace",
                 "pins": "pin",
                 "bracelets": "bracelet",
                 "sets (parures)": "jewelry set"
                 # "wedd": "wedding gown"
                 }

# 'gown' is ambiguous unless paired with 'ling' category, or 'nightgown' is found in description; must have its own filtering condition

# this dictionary (2 of 2) uses values in name to standardize item types
nametypes_dict = {"sweater": ["sweater", "turtleneck", "pullover", "cardigan", "twin set", "shrug"],
                 "bra": ["bra ", "bustier", "merry widow"],
                 "girdle": ["girdle", "corset", "corselette", "foundation", "waist cincher", "allinone", "enhancer"],
                 "half slip": ["half slip", "petticoat", "crinoline"],
                 "nightgown": ["nightie", "night gown", "nightgown", "gown and robe"],
                 "pajama": ["pajama", "pajama and robe", "long johns", "lounger"],
                 "robe": ["robe", "dressing gown", "bed jacket", "bedjacket", "boudoir jacket", "kimono", "morning jacket", "peignoir", "lounging jacket", "coverup", "cover up", "house coat", "duster", "lingerie jacket"],
                 "wedding gown": ["wedding dress", "wedding gown", "bridal gown", "empire gown with pearl clusters"],
                 "wedding veil": ["wedding veil", "bridal veil", "orange blossoms veil"],
                 "dress": ["day dress", "flapper dress", "1910s dress", "20s dress", "30s dress", "40s dress", "50s dress", "60s dress", "70s dress", "80s dress", "frock with lace collar", "dressx", "frockx", "dirndl", "kaftan", "rhumba dress", "pc dress", "piece dress", "gown with beaded", "mini and jacket", "rust halter gown", "chiffon ball gown", "moss green floral dress", "1930s black chiffon and lace dress", "thai silk tailored dress", "knit dress set", "office dress with pleated", "rockabilly dress", "lingerie dress", "polka dot dress", "secretary dress", "chiffon tailored dress", "black seersucker ensemble", "white pleated dress", "organdy dress", "couture quality cream wool dress", "mocha net dress", "swishy print dress", "halter dress with lace", "pewter silk tailored dress", "merino wool dress", "marilyn style 1950s tailored black fitted dress", "hourglass dress", "faux weave knit dress", "crafty stitcher 1960s", "hostess dress", "dress with slanted", "gypsy dress", "medallion print dress", "violet chiffon gown", "magenta print gown", "fantasy dress", "black velvet dress with pink", "floral satin empire gown", "white jersey gown", "canvas dress", "mini dress", "dressmaker frock", "teal striped dress", "navy fleck dress", "cheerleader uniform", "tea gown", "lawn frock", "taffeta frock", "homespun dress", "cut velvet dress with satin roses", "gauze dress", "lounge dress", "stripe halter dress", "color block dress", "strapless dress", "strapless frock", "18th century costume", "strapless gown", "linen tailored dress", "cotton frock", "muumuu", "butterflies halter dress", "black pinstripe dress", "hawaiian dress", "debutante gown", "debutante style gown", "diner dress", "waitress uniform", "uniform dress", "paisley dress", "tent dress", "gown with beadwork", "bandeau gown and overlay", "satin separates", "mini with bell sleeves", "patchwork dress", "black and striped dress", "dinner dress", "royal purple dress", "cotton halter dress", "antique dress", "lawn dress", "nurse uniform", "crepe and satin dress", "velvet dress with tatted daisies", "beaded 1950s gown", "1930s look dress", "1950sinspired dress", "jumper", "summer dress", "bombshell dress", "empire dress", "party frock", "chiffon empire gown", "blouson dress", "chiffon frock", "casual dress", "linen dress", "crepe dress", "silk dress", "silk frock", "terrycloth dress", "terrycloth knit dress", "swing style dress", "drop waist dress", "egyptian goddess gown", "dress in denim", "floral chiffon dress", "beaded gown", "cotton dress", "polyester dress", "jersey dress", "blue dress with", "wrap dress", "house dress", "tricot 1960s lounge dress", "party dress", "design print dress", "pucci look dress", "tennis dress", "sport dress", "sports dress", "aline dress", "maxi dress", "afternoon dress", "afternoon gown", "bohemian dress", "hippie dress", "evening gown", "gown with detachable train", "geisha girls dress", "silk chiffon dress", "evening dress", "sun dress", "cocktail", "shift", "sheath", "tunic dress", "diamond knit dress", "velvet halter dress", "1980s navy tailored dress", "sexy spring green dress", "chocolate cut velvet dress", "pixel panel dress", "mini dress by david hayes", "haute quality cardinal red wool dress", "india silk print dress", "brown knit dress in checks", "mustard plaid dress", "sari", "lavender haze dress", "garden path tailored dress", "baremidriff summer set", "violet knit dress", "apple green knit dress", "1960s mocha knit dress", "violet plaid dress", "diane fres", "hanae mori dress", "1960s designer dress", "1960s mod dress  long", "1980s red striped dress", "1930s floral dress", "1930s white dress", "1970s disco dress", "1950s purple dress", "1980s retro dress", "linen 1960s tailored office dress", "inkblot rayon dress", "calico peasant dress", "fitnflare dress", "emma peel style dress", "charcoal tailored dress", "maroon floral dress", "brown squares knit dress", "1970s muslin dress", "nymph dress", "little black dress", "daisies knit dress", "floral stripe dress", "sand knit dress", "avocado knit dress", "1920s navy satin dress", "blue knit office dress", "toffee gabardine dress", "white minimalist dress", "navy knit dress with silver", "peel 1960s black tailored wool dress", "daisy print frock", "1970s peplum dress"], # "dress
                 "suit": ["1940s suit", "smart 1950s black linen suit", "1940s navy suit with", "mocha raw silk suit", "mocha linen suit", "tone brown suit with square buttons", "dress and coat", "ensemble", "dress and jacket", "dress and tailored jacket", "dress and embroidered jacket", "sports suit", "crochet suit", "summer suit", "ultrasuede suit", "gray plaid cotton suit", "cadet blue suit", "camel linen suit", "1960s navy knit suit", "1960s beige suit", "gabardine suit", "wool suit ", "silver suit", "lime green suit", "black and white wool suit", "beige striped suit", "suit with suede trim", "orange wool suit", "hearts linen suit", "sports suit", "day suit", "evening suit", "textured wool suit", "cadet gray suit", "1960s suit", "pc suit", "ivory suit", "wool tweed suit", "polyester 1970s suit", "dress set", "satin suit", "sharkskin suit", "style suitx"],
                 "full slip": ["full slip", "slip", "chemise"],
                 "garter belt": ["garter belt"],""
                 "underwear": ["undies", "underwear", "bloomers", "panty", "panties", "brief"],
                 "lingerie": [ "negligee", "teddy", "baby doll set"],
                 "skirt": ["skirt", "skort"],
                 "romper": ["romper", "play suit", "playsuit", "play set", "playclothes", "top and short", "a and f casual set", "baremidriff set"],
                 "shirt": ["shirt", "weskit", "top", "blouse", "vest", "tshirt", "tee", "tunic", "poncho", "kaftan shirt", "polo", "sleeveless tank", "knit tank", "cotton tank", "victorian oriental ensemble", "cranberry shell", "powder pink shell", "1960s shell", "1960s overlay", "lace bodice overlay", "1890s waist", "tweed waist", "silk waist", "lawn waist", "mauve waist", "pink waist", "victorian black lace waist", "camisole", "1900s waist", "edwardian waist", "day bodice", "teal silk bodice", "football jersey", "cotton bodice with bell sleeves"],
                 "handbag": ["handbag", "bag", "purse", "minaudiere", "pochette", "muff", "leather hobo", "lucite clutch", "tote", "satchel", "evening clutch", "clutchx"],
                 "gloves": ["gloves", "mittens"],
                 "shorts": ["shorts", "hot pants", "culotte"],
                 "pant": ["pant", "trousers", "capri", "jeans", "clam digger", "hip hugger", "bellbottom", "bell bottom", "corduroy flares", "dungaree", "denim flares", "flares by wrangler", "elephant bells", "polyester levi", "button fly levi", "breeches", "jodphurs", "gauchos"],
                 "shoes": ["shoe", "platform", "heels", "pumps", "sandal", "stiletto", "stilleto", "slingback", "mary jane", "peeptoe", "peep toe", "mules", "flats", "sneaker", "spectators", "slides", "wedges", "espadrille", "loafers", "oxford"], 
                 "boots": ["boots"],
                 "coat": ["coat", "slicker", "cape"],
                 "jacket": ["canvas jacket", "suit jacket", "denim jacket", "fur jacket", "field jacket", "crochet jacket", "vinyl jacket", "track jacket", "stage jacket", "stage wear jacket", "brocade jacket", "plaid jacket", "jacket with zipped hood", "tyrolean jacket", "hostess jacket", "titanic era 1910s black serge ladies", "hourglass jacket", "quilted jacketx", "slouch jacketx", "jacket with atomic buttons", "overjacket", "bolero", "boucle` jacket", "linen jacket", "cotton jacket", "biker jacket", "wrap with bell sleeves", "shaggy jacket", "icelandic", "utility jacket", "cadet jacket", "nike jacket", "sueded jacket", "patchwork jacket", "windbreaker", "wind breaker", "striped jacket", "maxi jacket", "leather jacket", "jacket with hood", "swing jacket", "mohair jacket", "velvet jacket", "corduroy jacket", "cropped jacket", "summer jacket", "summer 70s jacket", "knit jacket", "suede jacket", "wrap jacket", "weekend jacket", "blazer", "1970s jacket", "retro jacket", "sun yatsen jacket", "jacket with sunflowers print", "purple jacket with butterfly", "denim blue tailored jacket", "tawny stripe jacket", "tweed jacket", "green silk tailored jacket", "red polyester jacket"],
                 "scarf": ["scarf", "shawl", "stole", "hanky", "challis wrap", "mohair wrap", "kerchief"],
                 "hat": ["hat", "visor", "pillbox", "beret", "tam", "skullcap", "bowler", "turban", "halo", "breton", "tweed cap", "boater", "hat and box", "cossack", "chapeau", "cloche", "toque", "headband", "street cap", "swim cap", "riding hood", "homburg", "fedora", "leather cap"],
                 "swimsuit": ["swimsuit", "swim suit", "bathing suit", "bikini", "swim dress", "swimplay"],
                 "sun glasses": ["glasses", "sunglasses"],
                 "fan": ["victorian fan", "souvenir fan"],
                 "umbrella": ["umbrella"],
                 "pantsuit": ["jumpsuit", "polyester suit", "catsuit", "palazzo set"],
                 "wallet": ["wallet", "cigarette case"],
                 "overalls": ["overalls"], 
                 "coveralls": ["work uniform"],
                 "hosiery": ["hosiery", "stockings", "stocking"],
                 "apron": ["apron", "smock"],
                 "hair comb": ["hair comb"],
                 "cigarette holder": ["cigarette holder"],
                 "compact": ["compact", "powder box"],
                 "jewelry set": ["necklace and earrings", "parure", "bracelet set"],
                 "earrings": ["earrings"],
                 "necklace": ["necklace", "pendant and chain", "pendant and rope chain", "pendant by celebrity", "enamel pendant", "choker"],
                 "bracelet": ["charm bracelet", "bangle", "bracelet from sanborn", "bracelet with poodle", "talons bracelet", "motif silvertone bracelet", "sunflower bracelet", "seashells bracelet", "metal bead bracelet", "chainmail bracelet", "bamboo cuff bracelet", "bells bracelet", "silver leaves bracelet", "slider bracelet", "corsage bracelet", "anklet", "strand bracelet", "daisy bracelet", "motif copper bracelet", "basket pin", "penny cuff bracelet"], 
                 "brooch": ["brooch", "cameo", "feather pin", "medallion pin", "sterling pin", "initial pin", "lapel pin", "stick pin", "grapes pin", "leaves pin", "lily pin", "fan 1960s pin", "curlique pin", "starburst pin", "crescent pin", "friend pin", "pink berriesx", "plumage pin", "fish pin", "trio pin", "insect pin", "spray pin", "crab pin", "candlelight pin", "pineapple pin", "sun pin", "pin with moveable parts", "sweethearts pin", "owl pin", "spider pin", "schnauzer pin", "donkey pin", "knot pin", "seahorses pin", "maltese cross by trifari", "dagger by trifari", "key by trifari", "crown by trifari", "cat pin", "cat with sapphire", "ostrich pin", "daisy pin", "filigree pin", "songbirds pin", "lucite pin", "dragonfly pin", "poodle pin", "tree pin", "dome pin", "laurel pin", "stone pin", "vintage pin", "art nouveau pin", "square pin", "oval pin", "carousel pin", "leaf pin", "metal pin", "flower pin", "circle pin", "bow pin", "pear pin", "scatter pins", "cornucopia pin"],
                 "spats": ["spats"],
                 "belt": ['chain belt with filigree', 'chevron belt', 'chainlink belt with tassel', 'hippie belt with faux tortoise', 'velvet belt with metallic gold', 'suede slouchy belt', 'belt with silver metal buckle', 'braided maroon belt', '1990s belt in suede', 'triple belt', 'brocade belt with fringe', 'beaded belt with tassels'],
                 "bra": ["bra"],
                 "ring": ["ring"],
                 "tie": ["tie"]
                 }

# combine dresses with suits (need more data and they are most similar)
nametypes_dict['dress'] += nametypes_dict['suit']

# match itemtype_dict & then nametype_dict keys to 'itemtype'
# if found, replace itemtype & put an "x" in typed
csv_file[['itemtype', 'typed']] = csv_file['itemtype'].apply(
    lambda cell_value: pd.Series(map_itemtype(cell_value))
)

csv_file[['itemtype', 'typed']] = csv_file.apply(
    lambda row: pd.Series(map_nametypes(row)),
    axis=1
)

# some items have ambiguous names and won't get labeled using the dictionaries above; they need a check from two columns to identify correct label

condition_ling = (csv_file['name'].str.contains("gown", case=False, na=False)) & (csv_file['itemtype'] == 'ling')
condition_gown = (csv_file['name'].str.contains("gown", case=False, na=False)) & (csv_file['description'].str.contains("nightgown", case=False, na=False))
condition_belt = (csv_file['name'].str.contains("belt", case=False, na=False)) & (csv_file['itemtype'] == 'accessories')
condition_purse = (csv_file['name'].str.contains("clutch", case=False, na=False)) & (csv_file['itemtype'].isin(['accessories', 'handbags & purses']))
condition_dress = (csv_file['name'].str.contains(r"gown|frock|dress")) & (csv_file['itemtype'] == 'formal wear')
condition_dress2 = (csv_file['name'].str.contains(r"gown|frock|dress")) & (csv_file['itemtype'] == 'formals')
condition_suit = (csv_file['name'].str.contains("suit", case=False, na=False)) & (csv_file['itemtype'] == 'business')
condition_suit2 = (csv_file['name'].str.contains(r"(?=.*dress)(?=.*jacket)", case=False, na=False)) & (csv_file['itemtype'] == 'business')

csv_file.loc[condition_ling, 'itemtype'] = 'nightgown'
csv_file.loc[condition_ling, 'typed'] = 'x'

csv_file.loc[condition_gown, 'itemtype'] = 'nightgown'
csv_file.loc[condition_gown, 'typed'] = 'x'

csv_file.loc[condition_belt, 'itemtype'] = 'belt'
csv_file.loc[condition_belt, 'typed'] = 'x'

csv_file.loc[condition_purse, 'itemtype'] = 'handbag'
csv_file.loc[condition_purse, 'typed'] = 'x'

csv_file.loc[condition_dress, 'itemtype'] = 'dress'
csv_file.loc[condition_dress, 'typed'] = 'x'

csv_file.loc[condition_dress2, 'itemtype'] = 'dress'
csv_file.loc[condition_dress2, 'typed'] = 'x'

# changes all suits to dress category for larger n for dress dataset
csv_file.loc[condition_suit, 'itemtype'] = 'dress'
csv_file.loc[condition_dress2, 'typed'] = 'x'

csv_file.loc[condition_suit2, 'itemtype'] = 'dress'
csv_file.loc[condition_suit2, 'typed'] = 'x'

# remove straggling stuff
csv_file = csv_file[csv_file['name'] != "blank"]
csv_file = csv_file[csv_file['itemtype'] != "blank"]
csv_file = csv_file[csv_file['sku'] != 1]
csv_file = csv_file[csv_file['itemtype'] != "giftcert"]

csv_file_columns = ['sku', 'name', 'era', 'description', 'itemtype', 'typed']
csv_file = csv_file[csv_file_columns]

# done with munging, now look at what photos vs. records are present
unmatched_files = []

# match valid filenames 
valid_pattern = re.compile(r'^v(\d{3,5})([a-zA-Z0-9]*)\.jpg$')
csv_skus = set(csv_file['sku'].dropna().astype(int)) 

csv_file['ml'] = ''

# find skus that are included in photos but not in csv
for filename in os.listdir(photos_folder):
    match = valid_pattern.match(filename)
    if match:
        image_sku = int(match.group(1))
        # check if the SKU exists in the CSV
        if image_sku not in csv_skus and image_sku not in unmatched_files:
            unmatched_files.append(image_sku) 
            new_row = {'sku': image_sku, 'ml': 'test'} # those skus without labels will be test set
            new_row_df = pd.DataFrame([new_row])
            csv_file = pd.concat([csv_file, new_row_df], ignore_index=True) 
            src_path = os.path.join(photos_folder, filename)
            dst_path = os.path.join(output_folder, filename)
            shutil.copy2(src_path, dst_path)

# last bit to munge: a few descriptions bump out one column to the right, unsure why
csv_file.loc[csv_file["ml"] == "x", "itemtype"] = csv_file["typed"]
csv_file.loc[csv_file["ml"] == "x", "typed"] = csv_file["ml"]

save_filepath = os.path.join(cwd, "compiled-catalogs")
save_file = os.path.join(save_filepath, "complete_catalog_cleaned.csv")
csv_file.to_csv(save_file)

# sending the rest to Llama to save me time; this list gives Llama a menu of name choices to choose from 
# unique_names = set(list(nametypes_dict.keys()) + list(itemtype_dict.values()))
# print(unique_names)

print(f"Saved: complete_catalog_cleaned.csv")