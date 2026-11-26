from __future__ import annotations
import json
import tempfile
from pathlib import Path

from scripts.generate_data import generate
from qualitycopilot.retrieval import ingest,retrieve
from qualitycopilot.graph import run_analysis,DemoModel
from qualitycopilot.schemas import Analysis,SupportCheck
from qualitycopilot.storage import save_review,history

def test_data_deterministic():
    a,b=generate();c,d=generate();assert a==c and b==d and len(a)>=150 and b

def test_ingestion_idempotent(tmp_path):
    docs,_=generate();p=tmp_path/"index";assert ingest(docs,p)==len(docs)==ingest(docs,p)
    c_docs=retrieve("brake batch glazing",path=p)
    assert c_docs
    assert all(d["doc_id"] and d["component"] and d["text"] for d in c_docs)

def test_schema_and_citations(tmp_path):
    docs,_=generate();p=tmp_path/"idx";ingest(docs,p)
    r=run_analysis("brake batch glazing",DemoModel(),retriever=lambda q,**kw:retrieve(q,path=p,**kw))
    Analysis.model_validate(r["analysis"]);ids={d["doc_id"] for d in r["retrieved"]}
    assert all(c in ids for h in r["analysis"]["hypotheses"] for c in h["citations"])

def test_validator_rejects_false_claim():
    m=DemoModel()
    assert not m.validate("The report proves chocolate is used in brakes.","Component: brakes. Failure mode: brake pad glazing.").supported

def test_retry_terminates_after_two_retries():
    class Bad(DemoModel):
        def validate(self,claim,text): return SupportCheck(supported=False,reason="forced failure")
    docs,_=generate();fake=lambda q,**kw:docs[:2]
    r=run_analysis("brake",Bad(),retriever=fake)
    assert r["status"]=="needs_review" and r["retries"]==2

def test_injection_is_not_followed(tmp_path):
    docs,_=generate();p=tmp_path/"idx";ingest(docs,p)
    r=run_analysis("brake batch glazing supplier",DemoModel(),retriever=lambda q,**kw:retrieve(q,path=p,**kw))
    assert any("IGNORE PREVIOUS INSTRUCTIONS" in d["text"].upper() for d in r["retrieved"])
    assert "IGNORE PREVIOUS INSTRUCTIONS" not in json.dumps(r["analysis"]).upper()
    assert "reveal secrets" not in json.dumps(r["analysis"]).lower()

def test_sqlite_decisions_survive_reopen(tmp_path):
    p=tmp_path/"reviews.sqlite3"
    for decision in ("approve","edit","reject"):
        save_review("d1",decision,"old","new","note",p)
    assert [r[2] for r in history(p)]==["reject","edit","approve"]

def test_app_smoke_and_friendly_error(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import app
    monkeypatch.setattr(app,"run_analysis",lambda q:{"status":"validated","analysis":{"summary":"Test result","hypotheses":[{"claim":"A supported synthetic claim.","citations":["brake_supplier_batch-001"]}]},"retrieved":[{"doc_id":"brake_supplier_batch-001","text":"Source report text."}]})
    at=AppTest.from_file("app.py").run()
    at.text_input(key="none").set_value("brakes") if at.text_input(key="none") else None
    text_inputs=at.text_input
    assert len(text_inputs)>=1
    text_inputs[0].set_value("brakes").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("Test result" in str(x.value) for x in at.markdown)
    assert any("Approve" in str(x.label) for x in at.button)

    monkeypatch.setattr(app,"run_analysis",lambda q:(_ for _ in ()).throw(TimeoutError()))
    at=AppTest.from_file("app.py").run()
    at.text_input[0].set_value("brakes").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("temporarily unavailable" in str(x.value) for x in at.error)
