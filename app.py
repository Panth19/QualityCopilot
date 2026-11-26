from __future__ import annotations
import json
from pathlib import Path
import streamlit as st
from qualitycopilot.graph import run_analysis
from qualitycopilot.storage import save_review, history
from qualitycopilot.retrieval import retrieve

st.set_page_config(page_title="QualityCopilot",page_icon="🔎",layout="wide")
st.title("🔎 QualityCopilot")
st.caption("Demo with synthetic data. Not a real quality system.")
DATA=Path("data/reports.json")
if not DATA.exists():
    from scripts.generate_data import write
    write()
if not Path("data/index").exists():
    from qualitycopilot.retrieval import ingest
    ingest(json.loads(DATA.read_text()))
if "analysis_count" not in st.session_state: st.session_state.analysis_count=0
questions=["What causes brake pad glazing?","Why do battery reports describe power loss?","What is linked to infotainment resets?","What patterns appear in water ingress reports?"]
tab, hist=st.tabs(["Analyze","History"])
with tab:
    example=st.selectbox("Demo questions",["Choose an example…"]+questions)
    question=st.text_input("Ask about the synthetic reports",value="" if example=="Choose an example…" else example,max_chars=500)
    if st.button("Analyze",type="primary"):
        if st.session_state.analysis_count>=15: st.warning("Session limit reached (15 analyses). Refresh later to continue.")
        elif not question.strip(): st.info("Enter a question to begin.")
        else:
            st.session_state.analysis_count+=1
            try:
                result=run_analysis(question)
                st.session_state.last_result=result
            except Exception:
                st.error("The analysis service is temporarily unavailable. Please try again.")
    result=st.session_state.get("last_result")
    if result:
        st.subheader("Result")
        if result["status"]=="no_evidence": st.info("No evidence found in the reports. I can't answer from memory.")
        else:
            st.write(result.get("analysis",{}).get("summary",""))
            if result.get("status")=="needs_review": st.warning("One or more claims could not be validated. Review carefully.")
            chunks={d["doc_id"]:d for d in result.get("retrieved",[])}
            for i,h in enumerate(result.get("analysis",{}).get("hypotheses",[])):
                st.markdown(f"**Hypothesis {i+1}:** {h['claim']}")
                for cid in h.get("citations",[]):
                    if cid in chunks:
                        with st.expander(f"Source: {cid}"): st.write(chunks[cid]["text"])
                    else: st.error(f"Invalid citation: {cid}")
                if result.get("status")=="needs_review": st.warning("Unvalidated claim")
                with st.expander("Human review"):
                    comment=st.text_input("Comment",key=f"comment-{i}")
                    edit=st.text_input("Edited claim (for Edit)",value=h["claim"],key=f"edit-{i}")
                    cols=st.columns(3)
                    for col,decision in zip(cols,["approve","edit","reject"]):
                        with col:
                            if st.button(decision.title(),key=f"{decision}-{i}"):
                                doc=h["citations"][0] if h.get("citations") else "unknown"
                                save_review(doc,decision,h["claim"],edit if decision=="edit" else None,comment)
                                st.success(f"{decision.title()} saved.")
with hist:
    st.subheader("Review history")
    rows=history()
    if rows:
        for row in rows: st.write({"doc_id":row[1],"decision":row[2],"claim":row[3],"edited_claim":row[4],"comment":row[5],"created_at":row[6]})
    else: st.caption("No review decisions yet.")
