import sqlite3
import pandas as pd
from unidecode import unidecode
import jellyfish
import re

# Configuration
DB_PATH = "../data/clients.db"

def normalize_text(text):
    """Standardizes text: lowercase, ascii only, stripped."""
    if text is None:
        return ""
    return unidecode(str(text)).lower().strip()

def normalize_national_id(text):
    """Removes dashes, spaces, and non-alphanumeric chars from IDs."""
    if text is None:
        return ""
    # Keep only letters and numbers
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', str(text))
    return cleaned.upper() if cleaned else ""

def normalize_email(text):
    """Lowercase, strip, and remove dots from the username part (gmail style)."""
    if not text:
        return ""
    text = str(text).lower().strip()
    if '@' in text:
        user, domain = text.split('@', 1)
        # Remove dots only from username (common strategy for fuzzy matching)
        user = user.replace('.', '')
        return f"{user}@{domain}"
    return text

def normalize_phone(text):
    """
    Robust normalization for PL, US, ES, DE numbers.
    1. Strip non-digits.
    2. Strip leading zeros (handles '00' international and '0' trunk prefixes).
    3. Strip known country codes if length matches expected pattern.
    """
    if not text:
        return ""
    
    # 1. Keep only digits
    digits = re.sub(r'\D', '', str(text))
    
    # 2. Strip leading zeros (e.g., 0048... -> 48..., 0176... -> 176...)
    digits = digits.lstrip('0')
    
    # 3. Heuristic Country Code Stripping
    # We only strip if the resulting length makes sense for that country.
    
    # PL: 9 digits. CC(48) + 9 = 11 digits.
    if len(digits) == 11 and digits.startswith('48'):
        return digits[2:]
        
    # US: 10 digits. CC(1) + 10 = 11 digits.
    if len(digits) == 11 and digits.startswith('1'):
        return digits[1:]
        
    # ES: 9 digits. CC(34) + 9 = 11 digits.
    if len(digits) == 11 and digits.startswith('34'):
        return digits[2:]
        
    # DE: Variable length (usually 10-11 without CC). 
    # CC(49) + 10/11 = 12/13 digits.
    if len(digits) >= 12 and digits.startswith('49'):
        return digits[2:]
        
    return digits

def normalize_address(text):
    """
    Standardizes address by expanding common abbreviations to full words.
    This is a standard NLP technique (canonicalization), not data leakage.
    We map ALL known variations (St, St., Str) to a single canonical form (street).
    """
    if not text:
        return ""
    text = unidecode(str(text)).lower().strip()
    
    # Remove punctuation (dots, commas) to make matching easier
    # "St." -> "st", "Apt." -> "apt"
    text = re.sub(r'[^\w\s]', '', text)
    
    # Canonical Mapping (Abbreviation -> Full Word)
    # We ONLY map Short -> Long. We do NOT map Long -> Short. 
    # This ensures "Street" stays "street", and "St" becomes "street".
    # NOTE: Would look into Libpostal for production-grade solution.
    replacements = {
        # English
        r'\bst\b': 'street',
        r'\bave\b': 'avenue',
        r'\brd\b': 'road',
        r'\bblvd\b': 'boulevard',
        r'\bdr\b': 'drive',
        r'\bln\b': 'lane',
        r'\bapt\b': 'apartment',
        r'\bste\b': 'suite',
        
        # Polish
        r'\bul\b': 'ulica',
        r'\bal\b': 'aleja',
        r'\bos\b': 'osiedle',
        r'\bm\b': 'mieszkanie',
        
        # Spanish
        r'\bc\b': 'calle',   # C/ -> c after punctuation removal
        r'\bav\b': 'avenida',
        r'\bpza\b': 'plaza',
        r'\bpso\b': 'paseo',
        
        # German
        r'\bstr\b': 'strasse',
        r'\bpl\b': 'platz'
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
        
    return text

def get_soundex(text):
    """Returns Soundex code (e.g., Smith -> S530)."""
    if not text:
        return "0000"
    return jellyfish.soundex(normalize_text(text))

def create_blocking_keys(df):
    """Adds blocking columns to the DataFrame."""
    
    # --- 1. CLEANING ---
    df['norm_first_name'] = df['first_name'].apply(normalize_text)
    df['norm_last_name'] = df['last_name'].apply(normalize_text)
    df['norm_nid'] = df['national_id'].apply(normalize_national_id)
    df['norm_email'] = df['email'].apply(normalize_email)
    df['norm_phone'] = df['phone_number'].apply(normalize_phone)
    df['norm_address'] = df['address'].apply(normalize_address)
    df['norm_city'] = df['city'].apply(normalize_text)

    # --- DATE PARSING OPTIMIZATION ---
    # We use errors='coerce' to turn garbage into NaT (Not a Time)
    # We use format='mixed' to handle US/EU mix automatically
    temp_dates = pd.to_datetime(df['dob'], errors='coerce', format='mixed')
    
    # Standardize DOB to YYYY-MM-DD string (or None if invalid)
    df['norm_dob'] = temp_dates.dt.strftime('%Y-%m-%d').replace({pd.NaT: None})
    
    # Extract year, fill NaT with 0, convert to string "1990" or "0000"
    df['norm_dob_year'] = temp_dates.dt.year.fillna(0).astype(int).astype(str).replace('0', '0000')

    # --- 2. BLOCKING KEYS ---
    # CRITICAL: We return None if any component is missing to avoid "Garbage Blocks"
    # (e.g., a block of 10M people with missing DOB).
    
    # KEY 1: Phonetic Surname + Year
    # Require both Surname and Year to be present.
    def make_bk_phonetic(row):
        if not row['norm_last_name'] or row['norm_dob_year'] == '0000':
            return None
        return get_soundex(row['norm_last_name']) + "_" + row['norm_dob_year']
        
    df['bk_phonetic_year'] = df.apply(make_bk_phonetic, axis=1)

    # KEY 2: National ID (Strong)
    # Only block if ID is valid (len > 4)
    df['bk_nid'] = df['norm_nid'].apply(lambda x: x if len(x) > 4 else None) 

    # KEY 3: First Initial + Full Surname
    # Require both First Name and Last Name.
    def make_bk_initials(row):
        if not row['norm_first_name'] or not row['norm_last_name']:
            return None
        return row['norm_first_name'][:1] + "_" + row['norm_last_name']

    df['bk_initials'] = df.apply(make_bk_initials, axis=1)

    # KEY 4: Contact Info (Phone/Email)
    # Catches cases where Name AND Date are both wrong/typo'd.
    # We use the last 6 digits of phone (very specific) or the normalized email.
    def make_bk_contact(row):
        # Try Phone first (Last 6 digits)
        if row['norm_phone'] and len(row['norm_phone']) >= 6:
            return "ph_" + row['norm_phone'][-6:]
        # Fallback to Email if Phone not available
        if row['norm_email']:
            return "em_" + row['norm_email']
        return None

    df['bk_contact'] = df.apply(make_bk_contact, axis=1)

    return df

def run_preprocessing():
    conn = sqlite3.connect(DB_PATH)
    
    print("Loading raw data...")
    df = pd.read_sql("SELECT * FROM clients", conn)
    
    print("Normalizing and generating keys...")
    df = create_blocking_keys(df)
    
    # Verification
    print("\n--- Date Parsing Check ---")
    print(df[['dob', 'norm_dob_year']].head(10))
    
    print("\n--- Blocking Keys Check ---")
    print(df[['bk_nid', 'bk_phonetic_year', 'bk_initials', 'bk_contact']].head(5))
    
    # Coverage Report
    # NOTE: Count how many keys are present per record. Useful for monitoring to see if data quality degrades -> more blocking keys may be needed.  
    df['key_count'] = df[['bk_nid', 'bk_phonetic_year', 'bk_initials', 'bk_contact']].notna().sum(axis=1)
    print("\n--- Blocking Key Coverage ---")
    print(df['key_count'].value_counts().sort_index())
    print(f"Orphans (0 keys): {len(df[df['key_count'] == 0])}")

    print("\nSaving processed data...")
    df.to_sql("clients_processed", conn, if_exists="replace", index=False)
    conn.close()

if __name__ == "__main__":
    run_preprocessing()
    
