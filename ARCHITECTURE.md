# Entity Resolution Service Architecture

## 1. Overview
This component is designed as a stateless microservice responsible for resolving client identities against a watchlist (Sanctions/PEP). It operates in an on-premise, air-gapped environment.

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
  "best_match": {
    "candidate_id": 1042,
    "match_type": "match",
    "confidence_score": 0.98,
    "explanation": ["Strong National ID Match", "High ML Confidence"]
  },
  "candidates_checked": 12,
  "processing_time_ms": 45.2
}
```

## 3. Scalability Strategy

### Parallel Processing
- **Stateless Design:** The API holds no state between requests. This allows horizontal scaling using Kubernetes (HPA) based on CPU/Memory usage.
- **Async I/O:** FastAPI uses `asyncio` to handle concurrent requests efficiently, preventing I/O blocking during database queries.

### Database Bottlenecks
- **Blocking Strategy:** We use a "Blocking" technique (Indexing) to reduce the search space from $O(N)$ to $O(1)$ (approx).
- **Future Optimization:** For high load (>1000 RPS), the SQLite database should be replaced with:
    - **PostgreSQL** with Read Replicas.
    - **Elasticsearch** for fuzzy blocking queries.
    - **Redis** for caching frequent queries.

## 4. Quality Monitoring & Observability

### Metrics (Prometheus)
The service exposes a `/metrics` endpoint scraping:
- `er_requests_total`: Traffic volume.
- `er_matches_found_total{status="match|review|no_match"}`: Match rate distribution.
- `er_request_latency_seconds`: P95/P99 latency tracking.

### Logging
- **Structured Logging:** JSON-formatted logs for easy ingestion (ELK Stack).
- **PII Protection:** No sensitive data (Names, IDs) is ever written to logs. Only Request IDs and non-sensitive metadata.

## 5. Model Versioning & Rollback
- **Artifact Management:** Models are versioned (e.g., `model_v1.pkl`, `model_v2.pkl`).
- **Configuration:** The active model path is configurable via Environment Variables (`MODEL_PATH`).
- **Rollback:** Reverting to a previous model version is as simple as changing the env var and restarting the container (or using a Canary Deployment strategy).

## 6. Security & Audit
- **Audit Trail:** Every decision includes an `explanation` field detailing *why* a match was found (e.g., "Exact Email Match").
- **Encryption:** All traffic should be encrypted via TLS (handled by the Ingress Controller/Load Balancer).
- **Data Minimization:** The API only returns the ID and Score, not the full sensitive profile of the matched candidate, unless authorized.
