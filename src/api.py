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

# Import our logic
from src import preprocessing
from src import matching

# --- CONFIGURATION ---
DB_PATH = "data/clients.db"
MODEL_PATH = "models/entity_resolution_model.pkl"
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

# Global Model
model = None

@app.on_event("startup")
def load_resources():
    global model
    logger.info("Loading resources...")
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            logger.info(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    else:
        logger.warning("Model file not found. Running in Rule-Only mode.")

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
    Finds candidates in the DB using blocking keys from the input record.
    """
    # Extract keys from the single record
    keys = {
        'bk_nid': record['bk_nid'],
        'bk_phonetic_year': record['bk_phonetic_year'],
        'bk_initials': record['bk_initials'],
        'bk_phone': record['bk_phone'],
        'bk_email': record['bk_email']
    }
    
    # Build dynamic query
    conditions = []
    params = []
    
    for key, value in keys.items():
        if value is not None:
            conditions.append(f"({key} = ?)")
            params.append(value)
            
    if not conditions:
        return pd.DataFrame()
        
    where_clause = " OR ".join(conditions)
    query = f"SELECT * FROM clients_processed WHERE {where_clause} LIMIT ?"
    params.append(limit)
    
    candidates = pd.read_sql(query, conn, params=params)
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
        # For SQLite, it's fast enough for this demo.
        with sqlite3.connect(DB_PATH) as conn:
            candidates = find_candidates(conn, record)
            
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
        df_dict = {0: record} # 0 is our input ID
        
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
