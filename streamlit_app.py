
import streamlit as st, requests, json

st.set_page_config(page_title="MACI — Shariah Compliance",
                   page_icon="🕌", layout="wide")

API_URL = st.sidebar.text_input(
    "MACI API URL",
    value="http://localhost:8000"
)

st.title("🕌 MACI — Muslim AI Content Intelligence")
st.caption("Shariah compliance classifier · v5.3 · F1=0.9497")

EXAMPLES = {
    "🔴 Riba"        : "Guaranteed 8% monthly returns on your investment — no risk involved",
    "🔴 Gharar"      : "Buy this land parcel — exact boundaries to be determined after surveying",
    "🔴 Maysir"      : "Binary options: predict Bitcoin up or down in 60 seconds, win 90% or lose all",
    "🔴 Fake Fatwa"  : "New fatwa from unnamed scholars: bank interest halal in non-Muslim countries",
    "🔴 Fake Hadith" : "The Prophet said bank interest is permitted if used for charity",
    "🔴 MLM"         : "Join our network — for every person you recruit earn 30% commission forever",
    "🔴 Misquote"    : "Sheikh Ibn Baz said Bitcoin is completely halal — invest freely",
    "✅ Authentic"   : "Murabaha: bank purchases asset and sells at disclosed cost plus profit margin",
    "✅ Zakat"       : "Zakat on gold is 2.5% when it reaches nisab of 85 grams after one lunar year",
    "🇮🇷 Farsi Riba" : "این وام با بهره ۲۴٪ سالانه است",
    "🇵🇰 Urdu"       : "یہ قرض 24 فیصد سالانہ سود کے ساتھ ہے",
}

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input")
    example = st.selectbox("Load example", ["— type your own —"] + list(EXAMPLES.keys()))
    default = EXAMPLES.get(example, "")
    text    = st.text_area("Text to classify", value=default, height=120)
    context = st.selectbox("Context",
        ["general","fintech","islamic_bank","social_media","regulatory"])
    submit  = st.button("🔍 Classify", type="primary", use_container_width=True)

with col2:
    st.subheader("Decision Packet")
    if submit and text.strip():
        try:
            r = requests.post(
                f"{API_URL}/api/v1/classify",
                json={"text": text, "context": context},
                timeout=30
            )
            if r.status_code == 200:
                d   = r.json()
                ev  = d["maci_evaluation"]
                bb  = d["boundary_behavior"]
                result = ev["result"]

                color = {"PASS":"#00C851","FLAGGED":"#FF4444",
                         "UNCERTAIN":"#FF8800"}.get(result,"gray")
                st.markdown(
                    f'<div style="background:{color}22;border-left:4px solid {color};'
                    f'padding:12px;border-radius:4px;margin-bottom:12px">'
                    f'<b style="color:{color};font-size:18px">{result}</b>'
                    f' — {ev["violation_class"]}</div>',
                    unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)
                m1.metric("Confidence", f'{ev["confidence_score"]:.1%}')
                m2.metric("Severity",   ev["severity"])
                m3.metric("Language",   d["language"])

                st.divider()
                st.markdown("**5 Decision Fields**")
                fields = {
                    "🏛️ Authority required"   : d["authority_required"],
                    "📖 Evidence pointer"      : (d["evidence_pointer"][:100]+"..."
                                                  if len(d.get("evidence_pointer",""))>100
                                                  else d.get("evidence_pointer","")),
                    "🚦 Proposed movement"     : d["proposed_movement"],
                    "⚠️ Protected effect risk" : d["protected_effect_risk"],
                    "🚫 Refusal condition"      : d["refusal_condition"] or "none",
                }
                for k, v in fields.items():
                    st.markdown(f"**{k}**")
                    st.code(v, language=None)

                st.divider()
                rec_color = {
                    "ALLOW"    :"#00C851","QUARANTINE":"#FF4444",
                    "ESCALATE" :"#FF8800","REVIEW"    :"#0099CC"
                }.get(bb["recommendation"],"gray")
                st.markdown(
                    f'<div style="background:{rec_color}22;border-left:4px solid {rec_color};'
                    f'padding:8px;border-radius:4px">'
                    f'<b>Boundary:</b> {bb["recommendation"]} → {bb["safe_next_step"]}'
                    f'</div>', unsafe_allow_html=True)

                with st.expander("Full JSON packet"):
                    st.json(d)
            else:
                st.error(f"API error {r.status_code}: {r.text}")
        except Exception as e:
            st.error(f"Cannot reach API: {e}")
            st.info("Start the API first, then enter its URL above")

st.sidebar.divider()
st.sidebar.markdown("**Model info**")
st.sidebar.markdown("- XLM-R test F1: **0.9497**")
st.sidebar.markdown("- sklearn F1: **0.9026**")
st.sidebar.markdown("- Languages: EN AR FA TJK UR")
st.sidebar.markdown("- Classes: 8")
st.sidebar.markdown("- Schema: MACI-0.1")
