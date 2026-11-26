from __future__ import annotations
import hashlib, re
from pathlib import Path
from qdrant_client import QdrantClient, models

COLLECTION="quality_reports"
DIM=384

def embed(text: str) -> list[float]:
    """Stable signed feature hashing; intentionally offline and dependency-light."""
    vec=[0.0]*DIM
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        idx=int(hashlib.md5(token.encode()).hexdigest(),16)%DIM
        vec[idx] += 1.0
    norm=sum(v*v for v in vec)**0.5 or 1.0
    return [v/norm for v in vec]

def client(path="data/index"):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    return QdrantClient(path=str(path))

def ingest(docs, path="data/index"):
    c=client(path)
    if c.collection_exists(COLLECTION): c.delete_collection(COLLECTION)
    c.create_collection(COLLECTION, vectors_config=models.VectorParams(size=DIM,distance=models.Distance.COSINE))
    c.upsert(COLLECTION, points=[models.PointStruct(id=i+1,vector=embed(d["text"]),payload=d) for i,d in enumerate(docs)],wait=True)
    count=c.count(COLLECTION,exact=True).count
    c.close()
    return count

def retrieve(question: str, top_k=5, threshold=0.12, component=None, path="data/index"):
    c=client(path)
    if not c.collection_exists(COLLECTION):
        c.close(); return []
    flt=None
    if component:
        flt=models.Filter(must=[models.FieldCondition(key="component",match=models.MatchValue(value=component))])
    hits=c.query_points(collection_name=COLLECTION,query=embed(question),limit=top_k,score_threshold=threshold,
                        query_filter=flt,with_payload=True).points
    c.close()
    return [{**hit.payload,"score":hit.score} for hit in hits]
