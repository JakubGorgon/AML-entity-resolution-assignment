import pandas as pd
from faker import Faker
import random
import uuid
import string
from datetime import datetime
import sqlite3
import os
from unidecode import unidecode 

# Config
RANDOM_SEED = 42
# Assuming script is run from project root
OUTPUT_DIR = "data"
DB_NAME = "clients.db"
CSV_NAME = "messy_data.csv"

# Separate Faker Instances per country
fakers = {
    'PL': Faker('pl_PL'),
    'US': Faker('en_US'),
    'ES': Faker('es_ES'),
    'DE': Faker('de_DE')
}

# Seed them all for reproducibility
for key in fakers:
    fakers[key].seed_instance(RANDOM_SEED)

random.seed(RANDOM_SEED)

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def introduce_date_noise(date_obj):
    """Simulates inconsistent date formats."""
    if date_obj is None:
        return None
    
    rand = random.random()
    if rand < 0.6:
        return date_obj.strftime('%Y-%m-%d')
    elif rand < 0.75:
        return date_obj.strftime('%d/%m/%Y')
    elif rand < 0.85:
        return date_obj.strftime('%m-%d-%Y')
    elif rand < 0.95:
        return date_obj.strftime('%Y/%m/%d')
    else:
        return None

def introduce_string_noise(text, probability=0.3):
    """Introduces typos, transpositions, or deletions."""
    if text is None or random.random() > probability:
        return text
    
    text = list(text)
    noise_type = random.choice(['typo', 'swap', 'delete', 'insert'])
    
    if len(text) < 2:
        return "".join(text)

    idx = random.randint(0, len(text) - 1)
    
    if noise_type == 'typo':
        # Replace with random char
        text[idx] = random.choice(string.ascii_letters)
    elif noise_type == 'swap' and idx < len(text) - 1:
        text[idx], text[idx+1] = text[idx+1], text[idx]
    elif noise_type == 'delete':
        del text[idx]
    elif noise_type == 'insert':
        text.insert(idx, random.choice(string.ascii_letters))
        
    return "".join(text)

def introduce_address_noise(address, probability=0.4):
    """Introduces address-specific noise."""
    if address is None or random.random() > probability:
        return address
    
    # Common abbreviations (US, PL, ES, DE)
    replacements = {
        # US / Generic English
        'Street': 'St.', 'St': 'Street',
        'Avenue': 'Ave.', 'Ave': 'Avenue',
        'Road': 'Rd.', 'Rd': 'Road',
        'Boulevard': 'Blvd.', 'Blvd': 'Boulevard',
        'Drive': 'Dr.', 'Dr': 'Drive',
        'Lane': 'Ln.', 'Ln': 'Lane',
        'Apartment': 'Apt.', 'Apt': 'Apartment',
        'Suite': 'Ste.', 'Ste': 'Suite',
        
        # PL (Poland)
        'ulica': 'ul.', 'ul': 'ulica',
        'aleja': 'al.', 'al': 'aleja',
        'osiedle': 'os.', 'os': 'osiedle',
        'mieszkanie': 'm.', 'm': 'mieszkanie',
        
        # ES (Spain)
        'Calle': 'C/', 'C/': 'Calle',
        'Avenida': 'Av.', 'Av': 'Avenida',
        'Plaza': 'Pza.', 'Pza': 'Plaza',
        'Paseo': 'Pso.', 'Pso': 'Paseo',
        
        # DE (Germany)
        'Straße': 'Str.', 'Str': 'Straße',
        'Platz': 'Pl.', 'Pl': 'Platz'
    }
    
    words = address.split()
    new_words = []
    for word in words:
        # Remove punctuation for checking
        clean_word = word.rstrip('.,')
        if clean_word in replacements and random.random() > 0.5:
            new_words.append(replacements[clean_word])
        else:
            new_words.append(word)
            
    # Occasional typo in the address string itself
    result = " ".join(new_words)
    if random.random() > 0.7:
        result = introduce_string_noise(result, probability=1.0)
        
    return result

def get_alternate_national_id(country_code, current_fake):
    """Generates a completely different ID type (e.g., Passport) or just a different number."""
    # 30% chance to return a Passport number instead of the standard ID
    if random.random() < 0.3:
        # Faker doesn't always have a dedicated passport provider for every locale,
        # but we can simulate it or use a generic one.
        return current_fake.bothify('??#######') # Generic Passport format
    
    # Otherwise, just generate a NEW standard ID (simulating a different document or major error)
    if country_code == 'US':
        return current_fake.ssn()
    elif country_code == 'PL':
        return current_fake.pesel()
    elif country_code == 'ES':
        return current_fake.nif()
    elif country_code == 'DE':
        return current_fake.rvnr()
    return current_fake.bothify('??######')


def generate_ground_truth(n_entities=1000, collision_rate=0.05):
    ground_truth = []
    
    for _ in range(n_entities):
        # DECISION: Create a "Doppelganger" (Hard Negative)?
        # We copy Name + DOB from an existing person, but generate new ID/Contact.
        # This forces the model to learn that Name+DOB is not enough.
        is_collision = False
        if len(ground_truth) > 100 and random.random() < collision_rate:
            is_collision = True
            base_entity = random.choice(ground_truth)
            country_code = base_entity['country']
            current_fake = fakers[country_code]
            
            fname = base_entity['first_name']
            lname = base_entity['last_name']
            dob = base_entity['dob']
            gender = 'M' # Simplified, doesn't matter for collision
        else:
            # 1. Pick Country
            country_code = random.choice(['PL', 'US', 'ES', 'DE'])
            current_fake = fakers[country_code]
            
            gender = random.choice(['M', 'F'])
            
            # 2. Generate Name (Strictly from that country's generator)
            if country_code == 'ES':
                # Spanish: often two surnames
                fname = current_fake.first_name()
                lname = f"{current_fake.last_name()} {current_fake.last_name()}"
            else:
                if gender == 'M':
                    fname = current_fake.first_name_male()
                    lname = current_fake.last_name_male()
                else:
                    fname = current_fake.first_name_female()
                    lname = current_fake.last_name_female()
            
            dob = current_fake.date_of_birth(minimum_age=18, maximum_age=80)

        # 3. Generate Email
        # Use Faker's locale-specific email provider for realism (e.g., .pl, .de domains)
        # But occasionally force a global provider (gmail, etc.)
        if random.random() < 0.7:
            email = current_fake.email()
        else:
            safe_fname = unidecode(fname.split(' ')[0].lower())
            safe_lname = unidecode(lname.split(' ')[-1].lower())
            domain = random.choice(['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com'])
            email = f"{safe_fname}.{safe_lname}@{domain}"

        # 4. National ID & Phone
        # Faker providers vary by locale
        if country_code == 'US':
            nat_id = current_fake.ssn()
        elif country_code == 'PL':
            nat_id = current_fake.pesel()
        elif country_code == 'ES':
            nat_id = current_fake.nif()
        elif country_code == 'DE':
            nat_id = current_fake.rvnr()

        phone = current_fake.phone_number()

        entity = {
            'entity_id': str(uuid.uuid4()),
            'first_name': fname,
            'last_name': lname,
            'dob': current_fake.date_of_birth(minimum_age=18, maximum_age=80),
            'email': email,
            'country': country_code,
            'national_id': nat_id,
            'phone_number': phone,
            'address': current_fake.street_address(),
            'city': current_fake.city()
        }
        ground_truth.append(entity)
    
    return ground_truth

def generate_messy_dataset(ground_truth, noise_multiplier=3):
    records = []
    
    for entity in ground_truth:
        # Base Record (Golden)
        base_record = entity.copy()
        base_record['record_id'] = str(uuid.uuid4())
        base_record['dob'] = base_record['dob'].strftime('%Y-%m-%d')
        records.append(base_record)
        
        # Duplicate Records (Messy)
        num_duplicates = random.randint(0, noise_multiplier)
        
        for _ in range(num_duplicates):
            record = entity.copy()
            record['record_id'] = str(uuid.uuid4())
            
            # Get the faker for the current country to generate "hard" alternatives
            current_fake = fakers[record['country']]

            # --- NOISE INJECTION ---
            
            # 1. Name Variations
            if random.random() > 0.7:
                # Initial only
                record['first_name'] = record['first_name'][0] + "."
            else:
                # Typos
                record['first_name'] = introduce_string_noise(record['first_name'], probability=0.2)
                record['last_name'] = introduce_string_noise(record['last_name'], probability=0.2)
            
            # 2. Date Noise
            record['dob'] = introduce_date_noise(record['dob'])
            
            # 3. Address Noise
            if random.random() > 0.95:
                record['address'] = None
            else:
                record['address'] = introduce_address_noise(record['address'])
            
            record['city'] = introduce_string_noise(record['city'], probability=0.1)

            # 4. ID & Phone Noise (Hard vs Soft)
            # National ID
            rand_id = random.random()
            if rand_id > 0.9:
                record['national_id'] = None
            elif rand_id > 0.8: # 10% chance of completely different ID (Hard Noise)
                record['national_id'] = get_alternate_national_id(record['country'], current_fake)
            else:
                record['national_id'] = introduce_string_noise(record['national_id'], probability=0.1)

            # Phone Number
            rand_phone = random.random()
            if rand_phone > 0.85:
                record['phone_number'] = None
            elif rand_phone > 0.75: # 10% chance of completely different Phone (Hard Noise)
                record['phone_number'] = current_fake.phone_number()
            else:
                record['phone_number'] = introduce_string_noise(record['phone_number'], probability=0.1)

            # 5. Email Noise (Hard vs Soft)
            rand_email = random.random()
            if rand_email > 0.8:
                record['email'] = None
            elif rand_email > 0.7: # 10% chance of completely different Email (Hard Noise)
                record['email'] = current_fake.email()
            else:
                record['email'] = introduce_string_noise(record['email'], probability=0.05)
            
            # 6. Country Noise (Rare)
            if random.random() > 0.98: # 2% chance of wrong country code
                record['country'] = random.choice(['PL', 'US', 'ES', 'DE'])

            records.append(record)
            
    return pd.DataFrame(records)

if __name__ == "__main__":
    ensure_output_dir()
    
    # Target: ~100k records
    # Multiplier 3 -> Avg 2.5 records per entity (1 base + avg 1.5 dupes)
    # 40,000 * 2.5 = 100,000
    print("Generating Ground Truth Entities (with 5% Doppelgangers)...")
    ground_truth = generate_ground_truth(n_entities=1000, collision_rate=0.05)
    
    print("Generating Messy Observations...")
    df_messy = generate_messy_dataset(ground_truth, noise_multiplier=3)
    
    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, CSV_NAME)
    df_messy.to_csv(csv_path, index=False)
    
    # Save SQL
    db_path = os.path.join(OUTPUT_DIR, DB_NAME)
    conn = sqlite3.connect(db_path)
    df_messy.to_sql("clients", conn, if_exists="replace", index=False)
    conn.close()
    
    print("-" * 30)
    print(f"Generated {len(df_messy)} records.")
    print("Sample Data (First 10 rows):")
    # Show specific columns to verify the fix
    print(df_messy[['first_name', 'last_name', 'national_id', 'phone_number', 'address']].head(10))
    
    
