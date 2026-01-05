# Entity Resolution Service Architecture

## 1. Overview
This component is designed as a stateless microservice responsible for resolving client identities against a watchlist (Sanctions/PEP).

## 2. API Contract (OpenAPI)

### Endpoint: `POST /resolve`

**Input:**
```json
{
  "first_name": "Jan",
  "last_name": "Kowalski",
  "dob": "1980-01-01",
  "national_id": "PL80010112345",
  "email": "jan.kowalski@example.com"
}
```

**Output:**
```json
{
  "status": "match",
  "matches": [
    {
      "candidate_id": "fab58e7f-1685-4873-a346-4ead834b11c2",
      "match_type": "match",
      "confidence_score": 0.9992,
      "ml_probability": 0.9992,
      "scores": {
        "name": 1,
        "national_id": 1,
        "email": 1,
        "phone": 1,
        "address": 1
      },
      "explanation": [
        "Rule 1: Strong National ID & Name Match",
        "Rule 2: Strong Contact Info & Name Match (Verified)",
        "Rule 5: High ML Probability (1.00)",
        "Reason: Exact Email Match",
        "Reason: Exact Phone Match",
        "Reason: Strong National ID Match"
      ]
    }
  ],
  "best_match": {
    "candidate_id": "fab58e7f-1685-4873-a346-4ead834b11c2",
    "match_type": "match",
    "confidence_score": 0.9992,
    "ml_probability": 0.9992,
    "scores": {
      "name": 1,
      "national_id": 1,
      "email": 1,
      "phone": 1,
      "address": 1
    },
    "explanation": [
      "Rule 1: Strong National ID & Name Match",
      "Rule 2: Strong Contact Info & Name Match (Verified)",
      "Rule 5: High ML Probability (1.00)",
      "Reason: Exact Email Match"
    ]
  },
  "candidates_checked": 7,
  "processing_time_ms": 54.01
}
```

## 3. Scalability Strategy

### Parallel Processing
- **Stateless Design:** The API holds no state between requests. This allows horizontal scaling using Kubernetes (HPA) based on CPU/Memory usage.
- **Async I/O:** FastAPI uses `asyncio` to handle concurrent requests efficiently, preventing I/O blocking during database queries.

### Database Bottlenecks
- **Blocking Strategy:** We use a "Blocking" technique (Indexing) to reduce the search space from $O(N)$ to $O(1)$ (approx).
- **Current PoC Optimizations:**
    - **LSH (Locality Sensitive Hashing):** Reduces fuzzy search space drastically.
    - **Optimized SQL:** Uses `UNION` instead of `OR` to enforce index usage.
    - **Parallelism:** Parallel feature calculation in batch mode; Async thread-pool execution for DB queries in API mode.
- **Future Optimization:** For high load (>1000 RPS), the SQLite database should be replaced with:
    - **PostgreSQL** with Read Replicas.
    - **Elasticsearch** or **Milvus** for scalable vector search.
    - **Redis** for caching frequent queries.

## 4. Quality Monitoring & Observability

### Metrics (Prometheus)
The service exposes a `/metrics` endpoint scraping:
- `er_requests_total`: Traffic volume.
- `er_matches_found_total{status="match|review|no_match"}`: Match rate distribution.
- `er_request_latency_seconds`: P95/P99 latency tracking.

### Logging
- **Standard Logging:** Timestamped text logs tracking service lifecycle and errors.
- **PII Protection:** Request payloads containing sensitive data (Names, IDs) are not logged.

## 5. Model Versioning & Rollback
- **Design Requirement:** In a production 50M-client environment, strict model governance is required.
    - **Artifact Management:** Models should be versioned (e.g., `model_v1.pkl`, `model_v2.pkl`) in an artifact registry (Artifactory/S3).
    - **Configuration:** The active model path is configurable via Environment Variables (`MODEL_PATH`).
    - **Rollback:** Reverting to a previous model involves rolling back the deployment to the previous Docker image/config.
- **Current PoC:** The system loads a single model file at startup. 

## 6. Security & Audit
- **Audit Trail:** Every decision includes an `explanation` field detailing *why* a match was found (e.g., "Exact Email Match").
- **Encryption:** All traffic should be encrypted via TLS (handled by the Ingress Controller/Load Balancer).

