import sqlite3
import pandas as pd
import jellyfish
import networkx as nx
from datetime import datetime
import joblib
import os

# Configuration
# Use absolute paths relative to this file to ensure it works from any CWD
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "clients.db")
MODEL_PATH = os.path.join(BASE_DIR, "models", "entity_resolution_model.pkl")

def get_candidates(conn):
    """
    Generates candidate pairs using SQL Blocking.
    Returns a DataFrame with columns: [id_a, id_b]
    """
    print("Generating candidates via Blocking Keys...")
    
    # We perform a Self-Join on the blocking keys.
    # We use 'record_id < record_id' to ensure:
    # 1. No self-matches (A=A)
    # 2. No duplicate pairs (A=B and B=A)
    query = """
    SELECT DISTINCT t1.record_id as id_a, t2.record_id as id_b
    FROM clients_processed t1
    JOIN clients_processed t2 ON
        (t1.bk_nid = t2.bk_nid AND t1.bk_nid IS NOT NULL) OR
        (t1.bk_phonetic_year = t2.bk_phonetic_year AND t1.bk_phonetic_year IS NOT NULL) OR
        (t1.bk_initials = t2.bk_initials AND t1.bk_initials IS NOT NULL) OR
        (t1.bk_contact = t2.bk_contact AND t1.bk_contact IS NOT NULL)
    WHERE t1.record_id < t2.record_id
    """
    
    pairs = pd.read_sql(query, conn)
    print(f"Found {len(pairs)} candidate pairs.")
    return pairs

def calculate_features(pairs, df_dict):
    """
    Calculates similarity features for each pair.
    df_dict: Dictionary mapping record_id -> row (dict) for fast lookup.
    """
    print("Calculating comparison features...")
    
    features = []
    
    for _, row in pairs.iterrows():
        id_a = row['id_a']
        id_b = row['id_b']
        
        rec_a = df_dict[id_a]
        rec_b = df_dict[id_b]
        
        feat = {
            'id_a': id_a,
            'id_b': id_b,
        }
        
        # --- 1. EXACT MATCH FLAGS ---
        # National ID (Strongest)
        # We use Damerau-Levenshtein to catch transpositions (81 -> 18) which count as dist=1
        nid_a = rec_a['norm_nid'] or ""
        nid_b = rec_b['norm_nid'] or ""
        if nid_a and nid_b:
            # Damerau-Levenshtein handles swaps (12 -> 21) as 1 edit
            dist = jellyfish.damerau_levenshtein_distance(nid_a, nid_b)
            max_len = max(len(nid_a), len(nid_b))
            
            if dist == 0:
                feat['nid_score'] = 1.0
            elif max_len > 0:
                # Normalize to 0-1 score
                # e.g. Dist=1, Len=9 -> 1 - (1/9) = 0.88
                # e.g. Dist=2, Len=9 -> 1 - (2/9) = 0.77
                feat['nid_score'] = 1.0 - (dist / max_len)
            else:
                feat['nid_score'] = 0.0
        else:
            feat['nid_score'] = 0.0
            
        # Email (Strong)
        # We use Jaro-Winkler for Email to catch typos
        feat['email_score'] = jellyfish.jaro_winkler_similarity(
            rec_a['norm_email'] or "", 
            rec_b['norm_email'] or ""
        )
            
        # Phone (Strong)
        # Exact match only for now (normalization handles most issues)
        if rec_a['norm_phone'] and rec_b['norm_phone']:
            feat['phone_match'] = 1 if rec_a['norm_phone'] == rec_b['norm_phone'] else 0
        else:
            feat['phone_match'] = 0
            
        # --- 2. FUZZY STRING METRICS ---
        # Names (Jaro-Winkler is good for typos/short names)
        feat['first_name_score'] = jellyfish.jaro_winkler_similarity(
            rec_a['norm_first_name'] or "", 
            rec_b['norm_first_name'] or ""
        )
        feat['last_name_score'] = jellyfish.jaro_winkler_similarity(
            rec_a['norm_last_name'] or "", 
            rec_b['norm_last_name'] or ""
        )
        
        # Address (Levenshtein is better for longer strings, normalized to 0-1)
        addr_a = rec_a['norm_address'] or ""
        addr_b = rec_b['norm_address'] or ""
        if addr_a and addr_b:
            max_len = max(len(addr_a), len(addr_b))
            dist = jellyfish.levenshtein_distance(addr_a, addr_b)
            feat['addr_score'] = 1 - (dist / max_len)
        else:
            feat['addr_score'] = 0
            
        # City
        feat['city_score'] = jellyfish.jaro_winkler_similarity(
            rec_a['norm_city'] or "", 
            rec_b['norm_city'] or ""
        )
        
        # --- 3. DATE METRICS ---
        # Exact DOB Match (using normalized YYYY-MM-DD)
        if rec_a['norm_dob'] and rec_b['norm_dob']:
            feat['dob_match'] = 1 if rec_a['norm_dob'] == rec_b['norm_dob'] else 0
        else:
            feat['dob_match'] = 0
        
        # Year Match (Soft check)
        feat['year_match'] = 1 if rec_a['norm_dob_year'] == rec_b['norm_dob_year'] and rec_a['norm_dob_year'] != '0000' else 0
        
        features.append(feat)
        
    return pd.DataFrame(features)

def classify_pairs(features_df):
    """
    Decides if a pair is a match based on feature scores.
    Returns the DataFrame with a 'is_match' column.
    """
    print("Classifying pairs...")
    
    # Try to load ML Model
    model = None
    if os.path.exists(MODEL_PATH):
        try:
            print(f"Loading ML model from {MODEL_PATH}...")
            model = joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"Failed to load model: {e}")
    
    # Predict with Model if available
    if model:
        feature_cols = [
            'nid_score', 'email_score', 'phone_match', 
            'first_name_score', 'last_name_score', 
            'addr_score', 'city_score', 
            'dob_match', 'year_match'
        ]
        # Ensure columns exist and are in correct order
        # Fill NaNs just in case, though calculate_features handles most
        X = features_df[feature_cols].fillna(0)
        # Get probabilities (class 1)
        features_df['ml_prob'] = model.predict_proba(X)[:, 1]
    else:
        features_df['ml_prob'] = 0.0
    
    def is_match(row):
        # Calculate Name Similarity Average
        name_avg = (row['first_name_score'] + row['last_name_score']) / 2
        
        # --- RULE BASED LOGIC (High Precision) ---
        
        # RULE 1: Strong National ID Match
        # National IDs are generally unique to an individual.
        # We trust this highly, but still require minimal name plausibility to catch data entry errors.
        # Score > 0.85 allows for 1 transposition in a 9-digit ID (0.88) or 1 substitution (0.88)
        if row['nid_score'] >= 0.85 and name_avg > 0.5:
            return 1
            
        # RULE 2: Email/Phone Match (Corroborated)
        # Shared email/phone is common (families, fraud). 
        # We require a decent name match to confirm it's the same person, not just a relative.
        # If name is VERY different, it might be a family member -> Manual Review (Not implemented here, so we treat as 0)
        if (row['email_score'] > 0.95 or row['phone_match'] == 1) and name_avg > 0.8:
            return 1
            
        # RULE 3: Strong Name + Date Match
        # If names are very similar AND DOB is exact
        if name_avg > 0.85 and row['dob_match'] == 1:
            return 1
            
        # RULE 4: Strong Name + Address Match
        # If names are similar AND Address is similar (catches cases with wrong DOB)
        if name_avg > 0.85 and row['addr_score'] > 0.7:
            return 1
            
        # RULE 5: Very Strong Name Match (Rare Name assumption)
        # If names are almost identical (typo) and Year matches
        if name_avg > 0.93 and row['year_match'] == 1:
            return 1
            
        # --- ML MODEL RESCUE (High Recall) ---
        # If rules didn't catch it, but model is confident
        # We use a threshold of 0.5 (standard) or higher if we want to be conservative.
        # Since we want to improve recall for edge cases, 0.5 is a good start.
        if row['ml_prob'] > 0.5:
             return 1
            
        return 0

    features_df['is_match'] = features_df.apply(is_match, axis=1)
    print(f"Classified {features_df['is_match'].sum()} matches out of {len(features_df)} pairs.")
    return features_df

def resolve_entities(features_df, all_record_ids):
    """
    Uses Graph Connected Components to resolve final Entity IDs.
    """
    print("Resolving entities via Graph Clustering...")
    
    # 1. Build Graph
    G = nx.Graph()
    G.add_nodes_from(all_record_ids) # Add ALL nodes, even singletons
    
    # 2. Add Edges for Matches
    matches = features_df[features_df['is_match'] == 1]
    edges = list(zip(matches['id_a'], matches['id_b']))
    G.add_edges_from(edges)
    
    # 3. Find Connected Components
    # Each component is a unique person
    entity_map = {}
    for i, component in enumerate(nx.connected_components(G)):
        entity_id = f"ENT_{i}"
        for record_id in component:
            entity_map[record_id] = entity_id
            
    print(f"Resolved {len(all_record_ids)} records into {len(set(entity_map.values()))} unique entities.")
    return pd.DataFrame(list(entity_map.items()), columns=['record_id', 'predicted_entity_id'])

def evaluate_results(predictions_df, conn):
    """
    Compares predictions against Ground Truth.
    """
    print("Evaluating results...")
    
    # Load Ground Truth (entity_id)
    ground_truth = pd.read_sql("SELECT record_id, entity_id as true_entity_id FROM clients", conn)
    
    # Merge
    merged = pd.merge(predictions_df, ground_truth, on='record_id')
    
    # --- METRICS CALCULATION ---
    # We use Pairwise Precision/Recall because Cluster IDs are arbitrary strings.
    # We generate all pairs from True Clusters and all pairs from Predicted Clusters.
    
    def get_pairs(df, col):
        pairs = set()
        for _, group in df.groupby(col):
            ids = sorted(group['record_id'].tolist())
            for i in range(len(ids)):
                for j in range(i+1, len(ids)):
                    pairs.add((ids[i], ids[j]))
        return pairs

    true_pairs = get_pairs(merged, 'true_entity_id')
    pred_pairs = get_pairs(merged, 'predicted_entity_id')
    
    tp = len(true_pairs.intersection(pred_pairs))
    fp = len(pred_pairs - true_pairs)
    fn = len(true_pairs - pred_pairs)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("-" * 30)
    print(f"True Pairs: {len(true_pairs)}")
    print(f"Predicted Pairs: {len(pred_pairs)}")
    print(f"Correct Matches (TP): {tp}")
    print(f"False Matches (FP): {fp}")
    print(f"Missed Matches (FN): {fn}")
    print("-" * 30)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("-" * 30)

def run_matching():
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Load Data for Lookup
    print("Loading processed data for lookup...")
    df_all = pd.read_sql("SELECT * FROM clients_processed", conn)
    # Convert to dict for O(1) lookup: {record_id: {col: val, ...}}
    df_dict = df_all.set_index('record_id').to_dict('index')
    all_record_ids = df_all['record_id'].tolist()
    
    # 2. Generate Candidates
    pairs_df = get_candidates(conn)
    
    # 3. Feature Engineering
    features_df = calculate_features(pairs_df, df_dict)
    
    # 4. Classification
    classified_df = classify_pairs(features_df)
    
    # 5. Clustering
    predictions_df = resolve_entities(classified_df, all_record_ids)
    
    # 6. Evaluation
    evaluate_results(predictions_df, conn)
    
    conn.close()
    return predictions_df

if __name__ == "__main__":
    run_matching()

