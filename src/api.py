import os
import sqlite3
import joblib
import pandas as pd
import logging
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List
from prometheus_client import make_asgi_app, Counter, Histogram
import time
import json
import numpy as np
from datasketch import MinHash, MinHashLSH
from pathlib import Path
import pickle

# Import our logic
from src import preprocessing
from src import matching
from src.settings import settings

# --- CONFIGURATION ---
DB_PATH = settings.db_path
MODEL_PATH = settings.model_path
LSH_INDEX_PATH = settings.lsh_index_path
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("api")

# --- METRICS ---
# Prometheus metrics
REQUEST_COUNT = Counter("er_requests_total", "Total entity resolution requests")
MATCH_FOUND_COUNT = Counter("er_matches_found_total", "Total matches found", ["status"])
LATENCY = Histogram("er_request_latency_seconds", "Request latency in seconds")

# --- DATA MODELS ---
class ClientRequest(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    dob: Optional[str] = Field(None, description="Date of Birth in YYYY-MM-DD format")
    email: Optional[str] = None
    phone_number: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    national_id: Optional[str] = None

class MatchScores(BaseModel):
    name: float
    national_id: float
    email: float
    phone: float
    address: float

class MatchResult(BaseModel):
    candidate_id: str
    match_type: str = Field(..., description="match, review, or no_match")
    confidence_score: float
    ml_probability: float
    scores: MatchScores
    explanation: List[str] = []

class ResolutionResponse(BaseModel):
    status: str
    matches: List[MatchResult] = []
    best_match: Optional[MatchResult] = None
    candidates_checked: int
    processing_time_ms: float

# --- APP INITIALIZATION ---
app = FastAPI(
    title="Entity Resolution API",
    description="API for matching client records against a watchlist.",
    version="1.0.0"
)

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Global Resources
model = None
lsh_index = None

@app.on_event("startup")
def load_resources():
    global model, lsh_index
    logger.info("Loading resources...")
    
    # Load ML Model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            logger.info(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    else:
        logger.warning("Model file not found. Running in Rule-Only mode.")

    # Load LSH Index (default) or rebuild (opt-in)
    # NOTE: Building LSH on startup can be expensive and slows rollouts/tests.
    try:
        if os.path.exists(LSH_INDEX_PATH):
            logger.info(f"Loading LSH index from {LSH_INDEX_PATH}...")
            with open(LSH_INDEX_PATH, "rb") as f:
                lsh_index = pickle.load(f)
            logger.info(
                f"LSH index loaded (threshold={settings.lsh_threshold}, num_perm={settings.lsh_num_perm})."
            )
        elif settings.rebuild_lsh_on_startup:
            logger.info(
                "LSH index not found; rebuilding on startup because ER_REBUILD_LSH_ON_STARTUP=true"
            )
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='clients_processed'"
                )
                if cursor.fetchone():
                    df_lsh = pd.read_sql(
                        "SELECT record_id, bk_minhash FROM clients_processed WHERE bk_minhash IS NOT NULL",
                        conn,
                    )
                    lsh = MinHashLSH(
                        threshold=settings.lsh_threshold, num_perm=settings.lsh_num_perm
                    )
                    count = 0
                    with lsh.insertion_session() as session:
                        for _, row in df_lsh.iterrows():
                            mh = MinHash(num_perm=settings.lsh_num_perm)
                            mh.hashvalues = np.array(
                                json.loads(row["bk_minhash"]), dtype="uint64"
                            )
                            session.insert(row["record_id"], mh)
                            count += 1
                    lsh_index = lsh

                    os.makedirs(os.path.dirname(LSH_INDEX_PATH) or ".", exist_ok=True)
                    with open(LSH_INDEX_PATH, "wb") as f:
                        pickle.dump(lsh_index, f)
                    logger.info(f"LSH index rebuilt with {count} records and saved.")
                else:
                    logger.warning("clients_processed table not found. Skipping LSH build.")
        else:
            logger.info(
                "LSH index not found; skipping build on startup (set ER_REBUILD_LSH_ON_STARTUP=true to rebuild)."
            )
    except Exception as e:
        logger.error(f"Failed to load/build LSH index: {e}")

# --- HELPER FUNCTIONS ---

def get_db_connection():
    # In production, use a connection pool
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def find_candidates(conn, record: pd.Series, limit=50):
    """
    Finds candidates in the DB using LSH (fuzzy name) and Exact Keys.
    """
    candidates_ids = set()
    
    # 1. LSH Lookup (Fuzzy Name)
    # Use the global lsh_index built at startup
    if lsh_index and record.get('bk_minhash'):
        try:
            mh = MinHash(num_perm=settings.lsh_num_perm)
            mh.hashvalues = np.array(json.loads(record['bk_minhash']), dtype='uint64')
            result = lsh_index.query(mh)
            candidates_ids.update(result)
        except Exception as e:
            logger.error(f"LSH Query failed: {e}")
            
    # 2. Exact Keys Lookup
    keys = {
        'bk_nid': record.get('bk_nid'),
        'bk_phone': record.get('bk_phone'),
        'bk_email': record.get('bk_email'),
        'bk_initial_dob': record.get('bk_initial_dob')
    }
    
    conditions = []
    params = []
    
    for key, value in keys.items():
        if value is not None:
            conditions.append(f"({key} = ?)")
            params.append(value)
            
    if conditions:
        where_clause = " OR ".join(conditions)
        # Fetch IDs first to merge with LSH results
        query = f"SELECT record_id FROM clients_processed WHERE {where_clause} LIMIT ?"
        exact_matches = pd.read_sql(query, conn, params=params + [limit])
        candidates_ids.update(exact_matches['record_id'].tolist())
        
    if not candidates_ids:
        return pd.DataFrame()
        
    # Fetch full records for all candidate IDs
    ids_list = list(candidates_ids)[:limit] # Apply limit to total
    if not ids_list:
        return pd.DataFrame()
        
    placeholders = ",".join(["?"] * len(ids_list))
    query = f"SELECT * FROM clients_processed WHERE record_id IN ({placeholders})"
    candidates = pd.read_sql(query, conn, params=ids_list)
    
    return candidates

# --- ENDPOINTS ---

@app.post("/resolve", response_model=ResolutionResponse)
async def resolve_entity(client: ClientRequest):
    start_time = time.time()
    REQUEST_COUNT.inc()
    
    try:
        # 1. Preprocess Input
        # Convert to DataFrame for compatibility with existing functions
        input_data = client.model_dump()
        df = pd.DataFrame([input_data])
        
        # Apply preprocessing (normalization + blocking key generation)
        # We reuse the logic from preprocessing.py
        df = preprocessing.create_blocking_keys(df)
        record = df.iloc[0]
        
        # 2. Find Candidates
        # We need a DB connection here. 
        # Since we are inside an async function, we should be careful with blocking calls.
        # We use run_in_executor to run the synchronous DB call in a separate thread.
        import asyncio
        loop = asyncio.get_event_loop()
        
        def query_db():
            with sqlite3.connect(DB_PATH) as conn:
                return find_candidates(conn, record)
                
        candidates = await loop.run_in_executor(None, query_db)
            
        if candidates.empty:
            duration = (time.time() - start_time) * 1000
            MATCH_FOUND_COUNT.labels(status="no_candidates").inc()
            return ResolutionResponse(
                status="no_match",
                candidates_checked=0,
                processing_time_ms=duration
            )
            
        # 3. Calculate Features
        # Prepare pairs DataFrame
        pairs_data = []
        df_dict = {0: record} # 0 is the input ID
        
        for idx, cand in candidates.iterrows():
            cand_id = cand['record_id']
            df_dict[cand_id] = cand
            pairs_data.append({'id_a': 0, 'id_b': cand_id})
            
        pairs_df = pd.DataFrame(pairs_data)
        
        # Calculate features (reuse matching.py logic)
        features_df = matching.calculate_features(pairs_df, df_dict)

        # Merge National ID for deduplication
        # We want to treat records with the same National ID as the same entity.
        features_df = features_df.merge(
            candidates[['record_id', 'national_id']], 
            left_on='id_b', 
            right_on='record_id', 
            how='left'
        )
        
        # 4. Predict
        # ML Prediction
        if model:
            feature_cols = [
                'nid_score', 'email_score', 'phone_match', 
                'first_name_score', 'last_name_score', 
                'dob_match', 'year_match'
            ]
            X = features_df[feature_cols].fillna(0)
            features_df['ml_prob'] = model.predict_proba(X)[:, 1]
        else:
            features_df['ml_prob'] = 0.0
            
        # Apply Rules
        features_df[['match_type', 'confidence_score', 'explanation']] = features_df.apply(matching.decide_match_status, axis=1)
        
        # 5. Select Best Match
        # Priority: Match > Review > No Match
        # Tie-breaker: confidence_score
        
        # Filter for matches or reviews
        potential_matches = features_df[features_df['match_type'].isin(['match', 'review'])]
        
        matches_list = []
        best_result = None
        status = "no_match"
        
        if not potential_matches.empty:
            # Sort by priority (Match > Review) and then score
            potential_matches['priority'] = potential_matches['match_type'].map({'match': 2, 'review': 1})
            potential_matches = potential_matches.sort_values(by=['priority', 'confidence_score'], ascending=False)
            
            seen_nids = set()

            # Convert all potential matches to MatchResult objects
            for _, row in potential_matches.iterrows():
                # Deduplication: If we've seen this National ID before, skip it.
                # This collapses multiple records of the same person into the single best match.
                nid = row.get('national_id')
                if nid and isinstance(nid, str) and nid.strip():
                    if nid in seen_nids:
                        continue
                    seen_nids.add(nid)

                matches_list.append(MatchResult(
                    candidate_id=str(row['id_b']),
                    match_type=row['match_type'],
                    confidence_score=float(row['confidence_score']),
                    ml_probability=float(row['ml_prob']),
                    scores=MatchScores(
                        name=(float(row.get('first_name_score', 0)) + float(row.get('last_name_score', 0))) / 2,
                        national_id=float(row.get('nid_score', 0)),
                        email=float(row.get('email_score', 0)),
                        phone=float(row.get('phone_match', 0)),
                        address=float(row.get('addr_score', 0))
                    ),
                    explanation=row['explanation'].split("; ")
                ))
            
            # Best match is the top one
            best_result = matches_list[0]
            status = best_result.match_type
            
            # Check for Identity Conflict (Contradictory Signals)
            # Conflict exists if:
            # 1. Candidate A matches via National ID.
            # 2. Candidate B matches via Name/Bio (and NOT National ID).
            # 3. Candidate A != Candidate B.
            
            id_match_candidates = set()
            name_match_candidates = set()
            
            for m in matches_list:
                # Check for ID signals (looking for "ID" in explanation strings)
                is_id_driven = any("ID" in exp for exp in m.explanation)
                
                # Check for Name/Bio signals (excluding ID ones to isolate pure name matches)
                # We treat "High ML Probability" as a Name/Bio signal if it's NOT accompanied by an ID signal.
                has_name_text = any("Name" in exp for exp in m.explanation)
                has_ml_text = any("ML Probability" in exp for exp in m.explanation)
                
                is_name_driven = (has_name_text or has_ml_text) and not is_id_driven
                
                if is_id_driven:
                    id_match_candidates.add(m.candidate_id)
                if is_name_driven:
                    name_match_candidates.add(m.candidate_id)
            
            # If we have ID matches and Name matches, and they are disjoint sets (different people)
            if id_match_candidates and name_match_candidates:
                if id_match_candidates.isdisjoint(name_match_candidates):
                    status = "review" # Force review on conflict
                    best_result.match_type = "review"
                    best_result.explanation.append("WARNING: Identity Conflict Detected (ID Match vs Name Match)")
            
        MATCH_FOUND_COUNT.labels(status=status).inc()
        duration = (time.time() - start_time) * 1000
        LATENCY.observe(time.time() - start_time)
        
        return ResolutionResponse(
            status=status,
            matches=matches_list,
            best_match=best_result,
            candidates_checked=len(candidates),
            processing_time_ms=duration
        )

    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}
