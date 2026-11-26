from __future__ import annotations
import sqlite3
from pathlib import Path

def connect(path="data/reviews.sqlite3"):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    c=sqlite3.connect(path);c.execute("CREATE TABLE IF NOT EXISTS reviews (id INTEGER PRIMARY KEY, doc_id TEXT NOT NULL, decision TEXT NOT NULL CHECK(decision IN ('approve','edit','reject')), original_claim TEXT NOT NULL, edited_claim TEXT, comment TEXT, created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')))" );c.commit();return c

def save_review(doc_id,decision,original_claim,edited_claim=None,comment="",path="data/reviews.sqlite3"):
    with connect(path) as c:
        cur=c.execute("INSERT INTO reviews(doc_id,decision,original_claim,edited_claim,comment) VALUES(?,?,?,?,?)",(doc_id,decision,original_claim,edited_claim,comment))
        return cur.lastrowid

def history(path="data/reviews.sqlite3"):
    with connect(path) as c: return c.execute("SELECT id,doc_id,decision,original_claim,edited_claim,comment,created_at FROM reviews ORDER BY id DESC").fetchall()
