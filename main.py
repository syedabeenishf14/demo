
import os, json, pickle, re, uuid, hashlib, numpy as np
from datetime import datetime, timezone
from pathlib import Path
from scipy.sparse import hstack
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import torch, faiss
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

# ── App ────────────────────────────────────────────────────────
app = FastAPI(
    title       = "MACI — Muslim AI Content Intelligence",
    description = "Shariah compliance classifier with full decision packet",
    version     = "5.3",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Config ─────────────────────────────────────────────────────
HF_REPO    = os.getenv("HF_REPO", "MerridaDataScientist72/maci-shield")
MODEL_DIR  = Path("maci_model")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_NAMES = {
    0:"Authentic",               1:"Riba (Usury/Interest)",
    2:"Gharar (Excessive Uncertainty)", 3:"Maysir (Gambling/Speculation)",
    4:"Fabricated/Unauthorized Fatwa",  5:"Quran/Hadith Fabrication",
    6:"MLM / Pyramid Scheme",           7:"Scholar Misquotation",
}
AUTH  = {0:"none",1:"shariah_scholar",2:"shariah_board",3:"shariah_scholar",
         4:"shariah_board",5:"shariah_board",6:"compliance_officer",7:"shariah_scholar"}
RISK  = {0:"None — content is Shariah-compliant",
         1:"Customer may rely on interest-bearing guidance as compliant",
         2:"Customer may enter contract with impermissible uncertainty",
         3:"Customer may engage in prohibited speculative activity",
         4:"False religious authority may be established",
         5:"Corrupted sacred text may be presented as authentic",
         6:"Customer may join prohibited pyramid scheme",
         7:"False scholarly position may mislead customer"}
REFU  = {0:None,
         1:"output contains interest-bearing structure",
         2:"output describes ambiguous or void contract",
         3:"output facilitates gambling or speculation",
         4:"output attributes ruling without verified source",
         5:"output presents fabricated text as authentic",
         6:"output promotes recruitment-based income",
         7:"output misattributes position to named scholar"}
MOVE  = {0:"deliver_output_to_user",1:"escalate_to_human",
         2:"escalate_to_human",3:"escalate_to_human",
         4:"quarantine_output",5:"quarantine_output",
         6:"quarantine_output",7:"quarantine_output"}
SEV   = {(0,"HIGH"):"NONE",(0,"MEDIUM"):"NONE",(0,"LOW"):"LOW",
         (1,"HIGH"):"CRITICAL",(1,"MEDIUM"):"HIGH",(1,"LOW"):"MEDIUM",
         (2,"HIGH"):"HIGH",(2,"MEDIUM"):"MEDIUM",(2,"LOW"):"LOW",
         (3,"HIGH"):"HIGH",(3,"MEDIUM"):"MEDIUM",(3,"LOW"):"LOW",
         (4,"HIGH"):"CRITICAL",(4,"MEDIUM"):"HIGH",(4,"LOW"):"MEDIUM",
         (5,"HIGH"):"CRITICAL",(5,"MEDIUM"):"CRITICAL",(5,"LOW"):"HIGH",
         (6,"HIGH"):"HIGH",(6,"MEDIUM"):"MEDIUM",(6,"LOW"):"LOW",
         (7,"HIGH"):"HIGH",(7,"MEDIUM"):"MEDIUM",(7,"LOW"):"LOW"}

URDU_RE  = re.compile(r"[\u0679\u0688\u0691\u06BA\u06BE\u06C1\u06C3\u06D2\u06D3]")
FARSI_RE = re.compile(r"[\u067E\u0686\u06AF\u06A9\u06CC\u06F0-\u06F9]")
TAJIK_KW = ["қуръон","фоиз","ҳалол","ҳаром","шариат","тиҷорат","закот"]

# ── Global state ───────────────────────────────────────────────
STATE = {}

def detect_lang(text):
    has_ar  = bool(re.search(r"[\u0600-\u06FF]", text))
    has_la  = bool(re.search(r"[a-zA-Z]", text))
    has_cy  = bool(re.search(r"[\u0400-\u04FF]", text))
    has_ur  = bool(URDU_RE.search(text))
    has_fa  = bool(FARSI_RE.search(text))
    has_tjk = has_cy and any(s in text.lower() for s in TAJIK_KW)
    if has_tjk: return "TJK"
    if has_ur:  return "UR"
    if has_fa:  return "FA"
    if has_ar and not has_la: return "AR"
    if has_ar:  return "AR"
    return "EN"

@app.on_event("startup")
async def load_models():
    MODEL_DIR.mkdir(exist_ok=True)
    print("Loading models from HuggingFace...")

    for repo_path, local_name in [
        ("ml_model.pkl",             "ml_model.pkl"),
        ("v5/fiqh_rulings.json",     "fiqh_rulings.json"),
        ("v5/xlmr/config.json",      "xlmr/config.json"),
        ("v5/xlmr/model.safetensors","xlmr/model.safetensors"),
        ("v5/xlmr/tokenizer.json",   "xlmr/tokenizer.json"),
        ("v5/xlmr/tokenizer_config.json","xlmr/tokenizer_config.json"),
        ("v5/fiqh_index_xlmr.faiss", "fiqh_index_xlmr.faiss"),
    ]:
        dst = MODEL_DIR / local_name
        dst.parent.mkdir(exist_ok=True)
        if not dst.exists():
            hf_hub_download(repo_id=HF_REPO, filename=repo_path,
                            repo_type="model", local_dir=str(MODEL_DIR))

    with open(MODEL_DIR/"ml_model.pkl","rb") as f:
        STATE["ml"] = pickle.load(f)
    with open(MODEL_DIR/"fiqh_rulings.json",encoding="utf-8") as f:
        STATE["rulings"] = json.load(f)

    STATE["tokenizer"] = AutoTokenizer.from_pretrained(
        str(MODEL_DIR/"xlmr"))
    STATE["xlmr"] = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR/"xlmr")).to(DEVICE)
    STATE["xlmr"].eval()

    STATE["faiss"] = faiss.read_index(
        str(MODEL_DIR/"fiqh_index_xlmr.faiss"))
    STATE["base_enc"] = (STATE["xlmr"].roberta
                         if hasattr(STATE["xlmr"],"roberta")
                         else STATE["xlmr"].base_model)

    print(f"✅ Models loaded | device={DEVICE} | "
          f"rulings={len(STATE['rulings'])}")

# ── Request / Response schemas ─────────────────────────────────
class ClassifyRequest(BaseModel):
    text               : str   = Field(..., min_length=3, max_length=2000)
    context            : Optional[str] = "general"
    user_role          : Optional[str] = "unknown"
    deployment_context : Optional[str] = "general"

class MACIEvaluation(BaseModel):
    result           : str
    violation_class  : str
    confidence_score : float
    confidence_band  : str
    severity         : str
    model_used       : str

class BoundaryBehavior(BaseModel):
    recommendation : str
    safe_next_step : str
    review_required: bool
    review_reason  : Optional[str]

class ClassifyResponse(BaseModel):
    schema_version          : str
    packet_id               : str
    created_at_utc          : str
    language                : str
    maci_evaluation         : MACIEvaluation
    authority_required      : str
    evidence_pointer        : str
    proposed_movement       : str
    protected_effect_risk   : str
    refusal_condition       : Optional[str]
    boundary_behavior       : BoundaryBehavior
    payload_hash            : str

# ── Core classify function ─────────────────────────────────────
def classify(text: str, context: str = "general") -> dict:
    ml      = STATE["ml"]
    xlmr    = STATE["xlmr"]
    tok     = STATE["tokenizer"]
    idx     = STATE["faiss"]
    rulings = STATE["rulings"]
    base_e  = STATE["base_enc"]
    lang    = detect_lang(text)
    partial = lang in ("FA","TJK")

    # XLM-R
    enc = tok(text, max_length=128, padding="max_length",
              truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        xlmr_probs = torch.softmax(
            xlmr(**enc).logits, dim=-1).cpu().numpy()[0]
    xlmr_conf = float(xlmr_probs.max())

    # ML model
    X = hstack([ml["vw"].transform([text]),
                ml["vc"].transform([text]),
                ml["vs"].transform([text])])
    ml_probs = ml["clf"].predict_proba(X)[0]

    # Ensemble
    if xlmr_conf >= 0.50:
        probs, model_used = xlmr_probs, "XLM-R"
    else:
        probs      = 0.60*ml_probs + 0.40*xlmr_probs
        model_used = "ML+XLM-R blend"

    # FAISS evidence
    with torch.no_grad():
        q_out = base_e(**enc)
    q_mask = enc["attention_mask"].unsqueeze(-1).float()
    q_emb  = (q_out.last_hidden_state * q_mask).sum(1) / q_mask.sum(1)
    q_emb  = torch.nn.functional.normalize(q_emb, dim=-1)
    q_emb  = q_emb.cpu().numpy().astype(np.float32)
    thresh = 0.38 if partial else 0.42
    scores, idxs = idx.search(q_emb, 3)
    evidence = "no_fiqh_match"
    for sc, ri in zip(scores[0], idxs[0]):
        if ri != -1 and float(sc) >= thresh:
            evidence = rulings[ri].get("text","")[:120]
            break

    # Decision
    pred  = int(np.argmax(probs))
    conf  = float(probs[pred])
    label = LABEL_NAMES[pred]
    tier  = "HIGH" if conf>0.75 else "MEDIUM" if conf>0.45 else "LOW"
    verdict = ("VIOLATION"      if pred!=0 and conf>0.45 else
               "AUTHENTIC"      if pred==0 and conf>0.45 else
               "LOW_CONFIDENCE")
    terry   = ("FLAGGED" if verdict=="VIOLATION" else
               "PASS"    if verdict=="AUTHENTIC"  else "UNCERTAIN")
    severity= SEV.get((pred,tier),"MEDIUM")
    mv      = MOVE[pred]
    if verdict=="AUTHENTIC": mv = "deliver_output_to_user"
    if severity=="CRITICAL" and verdict=="VIOLATION": mv = "quarantine_output"

    needs_review  = tier=="LOW" or partial
    review_reason = ("low_confidence" if tier=="LOW" else
                     "farsi_tajik_partial" if partial else None)
    boundary = {
        "recommendation": (
            "ALLOW"     if verdict=="AUTHENTIC" and not needs_review else
            "QUARANTINE"if verdict=="VIOLATION" and severity=="CRITICAL" else
            "ESCALATE"  if verdict=="VIOLATION" else "REVIEW"),
        "safe_next_step": mv,
        "review_required": needs_review,
        "review_reason"  : review_reason,
    }

    pid     = str(uuid.uuid4())
    ts      = datetime.now(timezone.utc).isoformat()
    payload = {"text":text,"verdict":verdict,"label":label,
               "conf":round(conf,6),"ts":ts}
    phash   = hashlib.sha256(
        json.dumps(payload,sort_keys=True).encode()).hexdigest()

    return {
        "schema_version"       : "MACI-0.1",
        "packet_id"            : pid,
        "created_at_utc"       : ts,
        "language"             : lang,
        "maci_evaluation"      : {
            "result"           : terry,
            "violation_class"  : label,
            "confidence_score" : round(conf,4),
            "confidence_band"  : tier,
            "severity"         : severity,
            "model_used"       : model_used,
        },
        "authority_required"   : AUTH[pred],
        "evidence_pointer"     : evidence,
        "proposed_movement"    : mv,
        "protected_effect_risk": RISK[pred],
        "refusal_condition"    : REFU[pred],
        "boundary_behavior"    : boundary,
        "payload_hash"         : f"sha256:{phash}",
    }

# ── Routes ─────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"name":"MACI API","version":"5.3",
            "status":"online","docs":"/docs"}

@app.get("/health")
def health():
    loaded = "xlmr" in STATE and "ml" in STATE
    return {"status":"ok" if loaded else "loading",
            "models_loaded": loaded,
            "device": DEVICE}

@app.post("/api/v1/classify", response_model=ClassifyResponse)
def classify_endpoint(req: ClassifyRequest):
    if "xlmr" not in STATE:
        raise HTTPException(503, "Models still loading")
    try:
        return classify(req.text, req.context)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/v1/classify/batch")
def classify_batch(texts: list[str]):
    if len(texts) > 50:
        raise HTTPException(400, "Max 50 texts per batch")
    return [classify(t) for t in texts]
