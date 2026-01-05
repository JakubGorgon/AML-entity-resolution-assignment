import sqlite3
import pandas as pd
from unidecode import unidecode
import jellyfish
import re
import os
import json
import pickle
import numpy as np
from datasketch import MinHash, MinHashLSH
from pathlib import Path

from src.settings import settings

# Configuration
DB_PATH = settings.db_path

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

def compute_minhash_signature(text, num_perm=None):
    """
    Computes MinHash signature for LSH.
    Returns a list of integers (hash values) serialized as JSON string.
    """
    if not text:
        return None
    
    if num_perm is None:
        num_perm = settings.lsh_num_perm

    # NOTE: We intentionally do not pass precomputed permutations here.
    # datasketch changed the type/shape of `permutations` across versions,
    # which makes cross-environment reproducibility brittle.
    m = MinHash(num_perm=num_perm)
    
    # Create 3-char shingles (n-grams)
    # e.g. "alex" -> "ale", "lex"
    text = str(text).lower().strip()
    if len(text) < 3:
        m.update(text.encode('utf8'))
    else:
        for i in range(len(text) - 2):
            m.update(text[i:i+3].encode('utf8'))
            
    # Return the hash values as a list
    return json.dumps(m.hashvalues.tolist())

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
    
    # KEY 1: MinHash Signature (Scalable Fuzzy Matching)
    # Replaces Soundex and Initials which are O(N^2) risks.
    # OPTIMIZATION: Use list comprehension instead of apply for 10x speedup
    print("  > Generating MinHash signatures...")
    full_names = (df['norm_first_name'].fillna('') + ' ' + df['norm_last_name'].fillna('')).str.strip()
    df['bk_minhash'] = [compute_minhash_signature(name) if name else None for name in full_names]

    # KEY 2: National ID (Strong)
    # Only block if ID is valid (len > 4)
    print("  > Generating ID keys...")
    df['bk_nid'] = df['norm_nid'].apply(lambda x: x if len(x) > 4 else None) 

    # KEY 3: Phone (Last 6 digits)
    # Good for catching typos in names/DOBs
    print("  > Generating Phone keys...")
    def make_bk_phone(row):
        if row['norm_phone'] and len(row['norm_phone']) >= 6:
            return row['norm_phone'][-6:]
        return None
    df['bk_phone'] = df.apply(make_bk_phone, axis=1)

    # KEY 4: Email (Exact)
    # Very strong key, catches almost all digital interactions
    # Ensure we don't block on empty strings or None
    print("  > Generating Email keys...")
    df['bk_email'] = df['norm_email'].apply(lambda x: x if x else None)
    
    # KEY 5: Initials + Last Name + DOB (Safe Loose Blocking)
    # Catches "M. Kilar" vs "Marek Kilar" without O(N^2) explosion.
    # Format: "m|kilar|1960-06-24"
    print("  > Generating Initial+DOB keys...")
    def make_bk_initial_dob(row):
        if not row['norm_first_name'] or not row['norm_last_name'] or not row['norm_dob']:
            return None
        initial = row['norm_first_name'][0]
        return f"{initial}|{row['norm_last_name']}|{row['norm_dob']}"
        
    df['bk_initial_dob'] = df.apply(make_bk_initial_dob, axis=1)

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
    print(df[['bk_nid', 'bk_minhash', 'bk_phone', 'bk_email']].head(5))
    
    # Coverage Report
    # NOTE: Count how many keys are present per record. Useful for monitoring to see if data quality degrades -> more blocking keys may be needed.  
    df['key_count'] = df[['bk_nid', 'bk_minhash', 'bk_phone', 'bk_email']].notna().sum(axis=1)
    print("\n--- Blocking Key Coverage ---")
    print(df['key_count'].value_counts().sort_index())
    print(f"Orphans (0 keys): {len(df[df['key_count'] == 0])}")

    print("\nSaving processed data...")
    df.to_sql("clients_processed", conn, if_exists="replace", index=False)
    
    # --- CREATE INDEXES ---
    # Critical for performance on large datasets (50M records).
    # Without these, the blocking join is O(N^2) or O(N*M) full scan.
    print("Creating indexes for blocking keys...")
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bk_nid ON clients_processed(bk_nid)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bk_phone ON clients_processed(bk_phone)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bk_email ON clients_processed(bk_email)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bk_initial_dob ON clients_processed(bk_initial_dob)")
    conn.commit()
    
    # --- BUILD & SAVE LSH INDEX ---
    # Pre-building this saves massive time in matching/training
    print("Building and saving LSH Index...")
    os.makedirs(os.path.dirname(settings.lsh_index_path) or "models", exist_ok=True)
        
    lsh = MinHashLSH(threshold=settings.lsh_threshold, num_perm=settings.lsh_num_perm)
    minhashes = {}
    
    # We iterate over the DataFrame we just saved
    # Optimization: Use insertion_session for faster inserts
    with lsh.insertion_session() as session:
        for idx, row in df.iterrows():
            if row['bk_minhash']:
                mh = MinHash(num_perm=settings.lsh_num_perm)
                mh.hashvalues = np.array(json.loads(row['bk_minhash']), dtype='uint64')
                session.insert(row['record_id'], mh)
                minhashes[row['record_id']] = mh
                
    with open(settings.lsh_index_path, "wb") as f:
        pickle.dump(lsh, f)
        
    with open(settings.minhashes_path, "wb") as f:
        pickle.dump(minhashes, f)
        
    print("LSH Index and MinHash objects saved to models/")
    
    conn.close()
    print("Preprocessing complete.")
if __name__ == "__main__":
    run_preprocessing()
    
