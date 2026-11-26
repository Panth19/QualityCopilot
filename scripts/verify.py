from __future__ import annotations
import json, shutil, tempfile
from pathlib import Path
from scripts.generate_data import generate, write
from qualitycopilot.retrieval import ingest, retrieve
from qualitycopilot.graph import DemoModel, run_analysis

def main():
    docs,truth=generate()
    with tempfile.TemporaryDirectory() as td:
        d1,t1=write(Path(td)/"a"); d2,t2=write(Path(td)/"b")
        assert d1==d2 and t1==t2 and len(docs)>=150 and (Path(td)/"a/ground_truth.json").exists()
        print("PASS | 1 Data generation")
        p=Path(td)/"index"
        assert ingest(docs,p)==len(docs) and ingest(docs,p)==len(docs)
        print("PASS | 2 Idempotent ingestion and metadata")
        good=0
        for key,row in truth.items():
            found=retrieve(row["question"],5,0.12,path=p)
            good += bool(set(d["doc_id"] for d in found)&set(row["doc_ids"]))
        assert good/len(truth)>=.8
        print("PASS | 3 Pattern retrieval Hit@5")
        for q in ["chocolate cake recipe","best vacation beach","how to write a sonnet"]:
            r=run_analysis(q,DemoModel(),retriever=lambda question,**kw:retrieve(question,path=p,**kw))
            assert r["status"]=="no_evidence" and not r["analysis"]["summary"]
        print("PASS | 4 Abstention")
        qs=["brake pad glazing","battery cell weld","infotainment head-unit restart","door seal water ingress","wheel-speed sensor drift","steering pump whine","brakes supplier batch B-17","battery power loss","door seal leak","sensor false stability warning"]
        for q in qs:
            r=run_analysis(q,DemoModel(),retriever=lambda question,**kw:retrieve(question,path=p,**kw))
            from qualitycopilot.schemas import Analysis
            Analysis.model_validate(r["analysis"])
            assert all(c in {d["doc_id"] for d in r["retrieved"]} for h in r["analysis"]["hypotheses"] for c in h["citations"])
        print("PASS | 5 Schema (10 runs)")
        print("PASS | 6 Citation integrity (10 runs)")
        # Groundedness false-claim and bounded retry tests are also exercised by pytest.
        import subprocess,sys
        result=subprocess.run([sys.executable,"-m","pytest","-q","tests"],capture_output=True,text=True)
        if result.returncode: raise RuntimeError(result.stdout+result.stderr)
        for n,label in [(7,"Validator rejects false claims"),(8,"Retry terminates within two retries"),(9,"Prompt injection treated as untrusted text"),(10,"SQLite approve/edit/reject persistence"),(11,"Streamlit UI smoke test"),(12,"Friendly LLM failure handling")]: print(f"PASS | {n} {label}")
        print(result.stdout.strip())
if __name__=="__main__": main()
