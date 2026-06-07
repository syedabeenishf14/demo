# MACI API v5.3

# MACI API v5.3 — Muslim AI Content Intelligence

**Shariah compliance signal layer for governed AI pipelines.**

MACI classifies AI-generated outputs against Islamic finance and values-sensitive criteria, producing a structured decision packet designed for handoff to a consequence-boundary runtime (Elyria Systems). It is not a governance runtime. It is a domain-intelligence signal surface.

---

## What MACI detects

| Class | Description |
|---|---|
| Authentic | Shariah-compliant content — pass |
| Riba | Interest-bearing financial guidance |
| Gharar | Excessive uncertainty / void contract language |
| Maysir | Gambling or prohibited speculation |
| Fabricated Fatwa | Unverified or hallucinated religious ruling |
| Quran/Hadith Fabrication | Falsified scripture or hadith citations |
| MLM / Pyramid Scheme | Prohibited recruitment-based income structures |
| Scholar Misquotation | Misattributed scholarly positions |

---

## Language coverage

| Language | Status |
|---|---|
| Arabic (MSA + Gulf dialect) | Full |
| English | Full |
| Arabizi (Arabic in Latin script) | Full |
| Urdu | Full |
| Farsi | Partial — fiqh layer |
| Tajik | Partial — detection + flag |

---

## Models

- **Primary:** XLM-RoBERTa fine-tuned on 320 balanced samples — test F1 **0.9497**
- **Fallback:** TF-IDF ensemble (word + char + char_wb) — F1 **0.9026**
- **Ensemble logic:** XLM-R is used when confidence ≥ 0.50. Below that, a 60/40 blend of ML + XLM-R is applied.
- **Evidence retrieval:** FAISS index of 145 fiqh rulings (dim=768) — nearest ruling returned as evidence pointer
- **Madhab coverage:** Hanafi, Maliki, Shafi'i, Hanbali, Ja'fari

---

## API

### Base URL
```
https://merridadatascientist72-maci-api.hf.space
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | API info and version |
| GET | `/health` | Model load status |
| POST | `/api/v1/classify` | Classify a single text |
| POST | `/api/v1/classify/batch` | Classify up to 50 texts |

---

### POST /api/v1/classify

**Request**
```json
{
  "text": "ana biddi akhadh qard bil faida 24%",
  "context": "islamic_finance",
  "user_role": "customer",
  "deployment_context": "customer_facing_tool"
}
```

**Response — full signal packet (Terry schema v0.1)**
```json
{
  "schema_version": "MACI-0.1",
  "packet_id": "3f9d2a8e-8e2e-4f80-b2d1-000000000001",
  "created_at_utc": "2026-05-28T10:44:00Z",
  "language": "AR",
  "maci_evaluation": {
    "result": "FLAGGED",
    "violation_class": "Riba (Usury/Interest)",
    "confidence_score": 0.9312,
    "confidence_band": "HIGH",
    "severity": "CRITICAL",
    "model_used": "XLM-R"
  },
  "authority_required": "shariah_scholar",
  "evidence_pointer": "Riba al-fadl and riba al-nasiah are both prohibited...",
  "proposed_movement": "quarantine_output",
  "protected_effect_risk": "Customer may rely on interest-bearing guidance as compliant",
  "refusal_condition": "output contains interest-bearing structure",
  "boundary_behavior": {
    "recommendation": "QUARANTINE",
    "safe_next_step": "quarantine_output",
    "review_required": false,
    "review_reason": null
  },
  "payload_hash": "sha256:a3f9..."
}
```

---

### POST /api/v1/classify/batch

**Request**
```json
["القرض بفائده 24%", "mudaraba is a halal partnership contract", "join our halal MLM network"]
```

**Response:** Array of signal packets (max 50 per request)

---

## Signal packet fields

| Field | Description |
|---|---|
| `schema_version` | MACI-0.1 — maps to Elyria Signal Packet v0.1 |
| `packet_id` | UUID per evaluation — unique receipt reference |
| `language` | Detected language: AR / EN / UR / FA / TJK / Arabizi |
| `result` | PASS / FLAGGED / UNCERTAIN |
| `violation_class` | 8-class taxonomy |
| `confidence_band` | HIGH (>0.75) / MEDIUM (>0.45) / LOW |
| `severity` | NONE / LOW / MEDIUM / HIGH / CRITICAL |
| `authority_required` | none / compliance_officer / shariah_scholar / shariah_board |
| `evidence_pointer` | Nearest fiqh ruling from FAISS index (120 chars) |
| `proposed_movement` | deliver / escalate / quarantine |
| `protected_effect_risk` | Plain-language description of harm if output is delivered |
| `refusal_condition` | Specific condition that triggered refusal |
| `boundary_behavior.recommendation` | ALLOW / ESCALATE / QUARANTINE / REVIEW |
| `payload_hash` | SHA-256 of packet payload — audit and replay integrity |

---

## Architecture

```
User / Enterprise Application
        │
        ▼
MACI — Output Admissibility Signal Layer
  XLM-RoBERTa (primary, F1 0.9497)
  TF-IDF ensemble (fallback, F1 0.9026)
  FAISS fiqh ruling index (145 vectors)
  Language detection (AR/EN/UR/FA/TJK/Arabizi)
        │
        ▼  Full signal packet
Elyria Consequence-Boundary Runtime
  Standing resolution
  Admissibility verdict (EXECUTE / REFUSE / ESCALATE)
  Receipt + Replay
        │
        ▼
Institutional Decision Environment
```

MACI is the signal layer. It does not hold execution custody. Execution custody remains with the consequence-boundary runtime. MACI produces the values-sensitive semantic signal the runtime cannot generate internally for Arabic-language and Islamic finance contexts.

---

## Deployment

### HuggingFace Spaces (Docker)

**Environment variable required:**
```
HF_REPO = MerridaDataScientist72/maci-shield
```

Files needed in Space root:
- `main.py`
- `requirements.txt`
- `Dockerfile`

### Requirements
```
fastapi
uvicorn
transformers
torch
faiss-cpu
huggingface_hub
scipy
numpy
pydantic
```

---

## HuggingFace model repo

```
MerridaDataScientist72/maci-shield
├── ml_model.pkl
├── v5/
│   ├── xlmr/
│   │   ├── model.safetensors
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   ├── fiqh_rulings.json
│   ├── fiqh_index_xlmr.faiss
│   └── metadata_v53.json
└── api/
    ├── main.py
    ├── requirements.txt
    └── Dockerfile
```

---

## Status

| Component | Status |
|---|---|
| XLM-RoBERTa classifier | ✅ F1 0.9497 |
| TF-IDF ensemble | ✅ F1 0.9026 |
| FAISS fiqh index | ✅ 145 rulings |
| Terry schema v0.1 | ✅ All fields |
| Elyria seam | ✅ Ready |
| FastAPI endpoint | ✅ Live |
| Farsi/Tajik | ⚠️ Partial — fiqh layer only |
| XLM-R GPU fine-tuning expansion | 🔄 Next phase |
| Institutional signing key | 🔄 Pending pilot |

---

## Pending (post-pilot)

- XLM-RoBERTa dataset expansion (target 500+ samples per class)
- Urdu/Farsi script disambiguation fix (2 edge cases)
- OpenITI ruling expansion
- Institutional signing key for `signature_status` field
- CamelBERT-CA integration for morphological Arabic

---

*MaqasidAI · Syeda Beenish Fatima · maqasidai.org*
*Signal layer for governed AI execution in Islamic finance and MENA institutional contexts*
```
