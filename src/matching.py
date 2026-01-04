import sqlite3
import pandas as pd
import jellyfish
import networkx as nx
from datetime import datetime
import joblib
import os

# Configuration
DB_PATH = "data/clients.db"
MODEL_PATH = "models/entity_resolution_model.pkl"

def analyze_blocking_stats(conn, pairs_count):
    """
    Calculates and prints health metrics for the blocking strategy.
    Critical for ensuring the system scales to millions of records.
    """
    print("\n--- Blocking Performance Report ---")
    
    # 1. Total Records
    try:
        total_records = pd.read_sql("SELECT COUNT(*) FROM clients_processed", conn).iloc[0, 0]
    except:
        total_records = 0
        
    print(f"Total Records: {total_records:,}")
    
    # 2. Reduction Ratio
    # Total possible pairs = N * (N-1) / 2
    # This metric tells us how much work we avoided compared to a full comparison.
    # Target: > 99.9% for large datasets.
    total_possible = (total_records * (total_records - 1)) / 2
    reduction_ratio = 1 - (pairs_count / total_possible) if total_possible > 0 else 0
    
    print(f"Candidate Pairs: {pairs_count:,}")
    print(f"Reduction Ratio: {reduction_ratio:.8%}")
    print(f"Pairs per Record: {pairs_count / total_records:.2f}" if total_records > 0 else "Pairs per Record: N/A")
    
    # 3. Block Size Analysis (The "Heavy Hitters")
    # Large blocks (e.g., "Smith") cause quadratic explosions (O(N^2)).
    # We monitor the Top 3 largest blocks to identify "Stop Words" or generic values.
    keys = ['bk_nid', 'bk_phonetic_year', 'bk_initials', 'bk_phone', 'bk_email']
    
    print("\n[Block Size Analysis - Top 3 Largest Blocks per Key]")
    for key in keys:
        query = f"""
        SELECT {key}, COUNT(*) as cnt 
        FROM clients_processed 
        WHERE {key} IS NOT NULL 
        GROUP BY {key} 
        ORDER BY cnt DESC 
        LIMIT 3
        """
        top_blocks = pd.read_sql(query, conn)
        print(f"Key: {key}")
        if top_blocks.empty:
            print("  (No keys found)")
        else:
            for _, row in top_blocks.iterrows():
                # Alert if block size is dangerous (> 50 is arbitrary for this small dataset, 
                # but for 50M records, > 1000 is usually the danger zone)
                alert = " [WARNING: Large Block]" if row['cnt'] > 50 else ""
                print(f"  '{row[key]}': {row['cnt']} records{alert}")
                
    print("-" * 30 + "\n")

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
        (t1.bk_phone = t2.bk_phone AND t1.bk_phone IS NOT NULL) OR
        (t1.bk_email = t2.bk_email AND t1.bk_email IS NOT NULL)
    WHERE t1.record_id < t2.record_id
    """
    
    pairs = pd.read_sql(query, conn)
    print(f"Found {len(pairs)} candidate pairs.")
    
    # Run Health Check
    analyze_blocking_stats(conn, len(pairs))
    
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

def decide_match_status(row):
    """
    Determines the match status (match, review, no_match) for a single pair row.
    Expects row to have feature columns and 'ml_prob'.
    """
    # Calculate Name Similarity Average
    name_avg = (row['first_name_score'] + row['last_name_score']) / 2
    
    # --- RULE BASED LOGIC (High Precision) ---
    
    # RULE 1: Strong National ID Match
    if row['nid_score'] >= 0.85 and name_avg > 0.5:
        return 'match'
        
    # RULE 2: Email/Phone Match (Corroborated)
    if (row['email_score'] > 0.95 or row['phone_match'] == 1) and name_avg > 0.8:
        return 'match'
        
    # RULE 3: Strong Name + Date Match
    if name_avg > 0.85 and row['dob_match'] == 1:
        return 'match'
        
    # RULE 4: Strong Name + Address Match
    if name_avg > 0.85 and row['addr_score'] > 0.7:
        return 'match'
        
    # RULE 5: Very Strong Name Match (Rare Name assumption)
    if name_avg > 0.93 and row['year_match'] == 1:
        return 'match'
        
    # --- ML MODEL RESCUE (High Recall) ---
    if row['ml_prob'] > 0.5:
            return 'match'
            
    # --- MANUAL REVIEW (Gray Area) ---
    # 1. ML Uncertainty: Model thinks there's a chance (20-50%)
    if row['ml_prob'] > 0.2:
        return 'review'
        
    # 2. Strong ID but weak name (Possible data entry error or fraud)
    if row['nid_score'] >= 0.85:
        return 'review'
        
    # 3. Very Strong Name but no other corroboration (Possible missing data)
    if name_avg > 0.9:
        return 'review'
        
    # 4. Exact Email Match (New Rule)
    # Emails are unique. If they match, it's highly suspicious even if names differ.
    if row['email_score'] == 1.0:
        return 'review'

    return 'no_match'

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
            'dob_match', 'year_match'
        ]
        # Ensure columns exist and are in correct order
        # Fill NaNs just in case, though calculate_features handles most
        X = features_df[feature_cols].fillna(0)
        # Get probabilities (class 1)
        features_df['ml_prob'] = model.predict_proba(X)[:, 1]
    else:
        features_df['ml_prob'] = 0.0
    
    features_df['match_type'] = features_df.apply(decide_match_status, axis=1)
    features_df['is_match'] = (features_df['match_type'] == 'match').astype(int)
    
    counts = features_df['match_type'].value_counts()
    print(f"Classification Results:\n{counts}")
    
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

def evaluate_results(predictions_df, classified_df, conn):
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
    
    # Identify Review Pairs
    # We need to ensure order doesn't matter for set intersection
    review_mask = classified_df['match_type'] == 'review'
    review_pairs = set()
    for _, row in classified_df[review_mask].iterrows():
        # Store both directions to be safe, or sort them
        p1 = (row['id_a'], row['id_b'])
        p2 = (row['id_b'], row['id_a'])
        review_pairs.add(p1)
        review_pairs.add(p2)
    
    tp = len(true_pairs.intersection(pred_pairs))
    fp_pairs = pred_pairs - true_pairs
    fn_pairs = true_pairs - pred_pairs
    
    # Filter FN: Remove pairs that are in the Review bucket
    # fn_pairs are (id_a, id_b) tuples. We check if they exist in review_pairs
    caught_in_review = len(fn_pairs.intersection(review_pairs))
    truly_missed_pairs = fn_pairs - review_pairs
    
    fp = len(fp_pairs)
    fn = len(fn_pairs)
    
    # Export Truly Missed Matches (FN - Review)
    if len(truly_missed_pairs) > 0:
        missed_df = pd.DataFrame(list(truly_missed_pairs), columns=['id_a', 'id_b'])
        missed_path = "data/missed_matches.csv"
        missed_df.to_csv(missed_path, index=False)
        print(f"Exported {len(missed_df)} truly missed matches to {missed_path}")

    # Export False Matches (FP)
    if fp > 0:
        fp_df = pd.DataFrame(list(fp_pairs), columns=['id_a', 'id_b'])
        fp_path = "data/false_matches.csv"
        fp_df.to_csv(fp_path, index=False)
        print(f"Exported {len(fp_df)} false matches to {fp_path}")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("-" * 30)
    print(f"True Pairs: {len(true_pairs)}")
    print(f"Predicted Pairs: {len(pred_pairs)}")
    print(f"Correct Matches (TP): {tp}")
    print(f"False Matches (FP): {fp}")
    print(f"Missed Matches (FN): {fn}")
    print(f"  - Caught in Review: {caught_in_review} (These are NOT auto-linked, but flagged for human)")
    print(f"  - Truly Missed:     {len(truly_missed_pairs)} (Neither auto-linked nor flagged)")
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
    evaluate_results(predictions_df, classified_df, conn)
    
    # 7. Export Manual Review Cases
    review_cases = classified_df[classified_df['match_type'] == 'review']
    if not review_cases.empty:
        review_path = "data/manual_review_cases.csv"
        review_cases.to_csv(review_path, index=False)
        print(f"Exported {len(review_cases)} cases for manual review to {review_path}")
    
    conn.close()
    return predictions_df

if __name__ == "__main__":
    run_matching()
    

