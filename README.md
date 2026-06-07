# MACI API v5.3

## Deploy in 3 minutes (Railway — free tier)

1. Push maci_api/ to a GitHub repo
2. Go to railway.app → New Project → Deploy from GitHub
3. Set environment variable: HF_REPO=MerridaDataScientist72/maci-shield
4. Deploy → get public URL in 2 minutes

## Endpoints

GET  /          → version info
GET  /health    → model load status
POST /api/v1/classify       → single text
POST /api/v1/classify/batch → up to 50 texts

## Example request

```bash
curl -X POST https://your-url/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "guaranteed 8% monthly returns on investment"}'
```

## Example response

```json
{
  "schema_version": "MACI-0.1",
  "packet_id": "uuid",
  "language": "EN",
  "maci_evaluation": {
    "result": "FLAGGED",
    "violation_class": "Riba (Usury/Interest)",
    "confidence_score": 0.87,
    "severity": "CRITICAL"
  },
  "authority_required": "shariah_scholar",
  "evidence_pointer": "...",
  "proposed_movement": "quarantine_output",
  "protected_effect_risk": "Customer may rely on interest-bearing guidance",
  "refusal_condition": "output contains interest-bearing structure",
  "boundary_behavior": {
    "recommendation": "QUARANTINE",
    "safe_next_step": "quarantine_output",
    "review_required": true
  }
}
```
