# ---------- app.py (updated) ----------
import os
import csv
import io
import json
import base64
import smtplib
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta, date
from email.message import EmailMessage
from urllib.parse import urlencode, quote_plus

import httpx
from fastapi import FastAPI, Depends, HTTPException, Header, APIRouter, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, conint

from sqlalchemy import (
    create_engine, Column, Integer, String, JSON, Boolean,
    DateTime, Date, UniqueConstraint, ForeignKey, func
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session, relationship

# =========================
# ENV / CONFIG
# =========================

def _env(name: str, default: str = "") -> str:
    v = os.getenv(name, default)
    if v is None:
        return default
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        v = v[1:-1].strip()
    return v

DATABASE_URL = _env("DATABASE_URL", "sqlite:///./schedule.db")
API_KEY      = _env("API_KEY", "changeme")
CHECK_TOKEN  = _env("CHECK_TOKEN", "RMETVM")

TIMEZONE             = _env("TIMEZONE", "Asia/Kolkata")
DEFAULT_PLAN_ID      = _env("DEFAULT_PLAN_ID", "default")
PROJECT_START_DATE   = _env("PROJECT_START_DATE")
FRONTEND_PUBLIC_BASE = _env("FRONTEND_PUBLIC_BASE", "")
BACKEND_PUBLIC_BASE  = _env("BACKEND_PUBLIC_BASE", "")

SMTP_HOST = _env("SMTP_HOST")
SMTP_PORT = int(_env("SMTP_PORT", "587") or "587")
SMTP_USER = _env("SMTP_USER")
SMTP_PASS = _env("SMTP_PASS")
SMTP_FROM = _env("SMTP_FROM", SMTP_USER or "")

DAILY_TO  = [e.strip() for e in _env("DAILY_TO", "").replace(";", ",").split(",") if e.strip()]

RESEND_API_KEY = _env("RESEND_API_KEY")
RESEND_FROM    = _env("RESEND_FROM", SMTP_FROM or (SMTP_USER or ""))
RESEND_TO      = _env("RESEND_TO", ",".join(DAILY_TO))

# =========================
# DB setup
# =========================
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

class Plan(Base):
    __tablename__ = "plans"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    total_days = Column(Integer, default=60)
    areas = Column(JSON, default=[])
    allow_multiple = Column(Boolean, default=False)
    start_date = Column(Date, nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    cells = relationship("Cell", cascade="all, delete-orphan", back_populates="plan")

class Cell(Base):
    __tablename__ = "cells"
    id = Column(Integer, primary_key=True)
    plan_id = Column(String, ForeignKey("plans.id", ondelete="CASCADE"), index=True)
    area = Column(String, index=True)
    day = Column(Integer, index=True)
    activities = Column(JSON, default=[])
    updated_by = Column(String, nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    plan = relationship("Plan", back_populates="cells")
    __table_args__ = (UniqueConstraint("plan_id", "area", "day", name="uniq_cell"),)

class Snapshot(Base):
    __tablename__ = "snapshots"
    plan_id = Column(String, primary_key=True)
    taken_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    grid = Column(JSON)

class RolloverLog(Base):
    __tablename__ = "rollover_log"
    id = Column(Integer, primary_key=True, autoincrement=True)
    plan_id = Column(String, index=True)
    from_day = Column(Integer)
    to_day = Column(Integer)
    moved = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# ---- Items & Costs persisted per plan+day ----
class Extras(Base):
    __tablename__ = "extras"
    id = Column(Integer, primary_key=True)
    plan_id = Column(String, index=True, nullable=False)
    day = Column(Integer, nullable=False)
    items = Column(JSON, nullable=False, default=[])
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    __table_args__ = (UniqueConstraint("plan_id", "day", name="extras_plan_day_uniq"),)

Base.metadata.create_all(engine)

# =========================
# FastAPI app
# =========================
app = FastAPI(title="Remodel Planner API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def require_key(x_api_key: Optional[str] = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Invalid API key")

# write-permission guard (admin key OR editor token)
def require_write(
    x_api_key: Optional[str] = Header(None),
    x_edit_token: Optional[str] = Header(None),
    t: Optional[str] = Query(None),
):
    if API_KEY and x_api_key == API_KEY:
        return
    token = x_edit_token or t
    if CHECK_TOKEN and token == CHECK_TOKEN:
        return
    raise HTTPException(401, "Need API key or editor token")

ops = APIRouter(prefix="/ops", tags=["ops"])

# =========================
# Schemas
# =========================
class PlanUpsert(BaseModel):
    id: str = "default"
    name: str = "Default Plan"
    total_days: conint(ge=1) = 60
    areas: List[str]
    allow_multiple: bool = False
    start_date: Optional[date] = None

class CellIn(BaseModel):
    area: str
    day: conint(ge=1)
    activities: List[str]

class GridIn(BaseModel):
    cells: List[CellIn]
    allow_multiple: Optional[bool] = None

class ChecklistCellUpdate(BaseModel):
    area: str
    day: conint(ge=1)
    done: List[int] = []
    undone: List[int] = []

class ChecklistUpdateIn(BaseModel):
    plan_id: str = DEFAULT_PLAN_ID
    updates: List[ChecklistCellUpdate]

# Items & Costs schemas
class ExtraItem(BaseModel):
    item: str
    qty: float = 0
    rate: float = 0
    amount: Optional[float] = None  # computed server-side

class ExtrasUpsert(BaseModel):
    items: List[ExtraItem] = []

# =========================
# Utilities
# =========================
def _today_local() -> date:
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(TIMEZONE)
        return datetime.now(tz).date()
    except Exception:
        return datetime.now().date()

def _now_local_dt() -> datetime:
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(TIMEZONE)
        return datetime.now(tz)
    except Exception:
        return datetime.now()

def _get_start_date_for_plan(db: Session, plan_id: str) -> date:
    p = db.get(Plan, plan_id)
    if p and p.start_date:
        return p.start_date
    if PROJECT_START_DATE:
        return datetime.strptime(PROJECT_START_DATE, "%Y-%m-%d").date()
    return _today_local()

# --- OFF-DAYS (Sundays + Dec 24/25/26 + Dec 31 + Jan 1, any year) ---
OFF_MD = {(12,24), (12,25), (12,26), (12,31), (1,1)}

def _is_off(d: date) -> bool:
    return d.weekday() == 6 or (d.month, d.day) in OFF_MD

def _date_for_day(start: date, day: int) -> date:
    dt = start
    steps = max(0, day - 1)
    while steps > 0:
        dt = dt + timedelta(days=1)
        if not _is_off(dt):
            steps -= 1
    return dt

def _workday_index(start: date, d: date) -> int:
    if d <= start:
        return 1
    idx, cur = 1, start
    while cur < d:
        cur = cur + timedelta(days=1)
        if not _is_off(cur):
            idx += 1
    return idx

def _fetch_cells_range(db: Session, plan_id: str, from_day: int, to_day: int):
    rows = (
        db.query(Cell.area, Cell.day, Cell.activities)
          .filter(Cell.plan_id == plan_id, Cell.day >= from_day, Cell.day <= to_day)
          .order_by(Cell.day.asc(), Cell.area.asc())
          .all()
    )
    return [{"area": r[0], "day": int(r[1]), "activities": r[2] or []} for r in rows]

def _upsert_cell(db: Session, plan_id: str, area: str, day: int, acts: List[str]):
    row = db.query(Cell).filter_by(plan_id=plan_id, area=area, day=day).one_or_none()
    if row:
        row.activities = acts
    else:
        db.add(Cell(plan_id=plan_id, area=area, day=day, activities=acts))

def _append_to_cell(db: Session, plan_id: str, area: str, day: int, items: List[str]):
    if not items:
        return
    dest = db.query(Cell).filter_by(plan_id=plan_id, area=area, day=day).one_or_none()
    if dest:
        lst = list(dest.activities or [])
        lst.extend(items)
        dest.activities = lst
    else:
        db.add(Cell(plan_id=plan_id, area=area, day=day, activities=list(items)))

# ---- task normalization + flag/slider sync ----
def _normalize_task(obj_or_json):
    try:
        t = json.loads(obj_or_json) if isinstance(obj_or_json, str) else dict(obj_or_json or {})
    except Exception:
        return {"name": str(obj_or_json), "role": "", "workers": 0, "hours": 0.0, "done": False, "progress": 0}
    name    = t.get("name") or t.get("task") or t.get("n") or ""
    role    = t.get("role") or t.get("r") or ""
    workers = t.get("workers", t.get("w", 0)) or 0
    hours   = t.get("hours",   t.get("h", 0)) or 0
    done    = bool(t.get("done") or t.get("d") or t.get("x") or t.get("dd"))
    progress = max(0, min(100, int(t.get("progress", t.get("p", 0)) or 0)))
    return {"name": name, "role": role, "workers": int(workers), "hours": float(hours), "done": done, "progress": progress}

def _set_done_flag_value(raw, value: bool):
    try:
        t = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except Exception:
        return raw
    compact = any(k in t for k in ("n","r","w","h")) and ("name" not in t)
    if compact:
        if value:
            t["x"] = True
            t["p"] = 100
        else:
            for k in ("x","d","dd"):
                if k in t:
                    t[k] = False
            if t.get("p", 0) >= 100:
                t["p"] = 0
    else:
        t["done"] = bool(value)
        if value:
            t["progress"] = 100
        else:
            if int(t.get("progress", 0)) >= 100:
                t["progress"] = 0
    return json.dumps(t, ensure_ascii=False)

def _set_progress_value(raw, value: int):
    try:
        t = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except Exception:
        return raw
    p = max(0, min(100, int(value)))
    compact = any(k in t for k in ("n","r","w","h")) and ("name" not in t)
    if compact:
        t["p"] = p
        if p >= 100:
            t["x"] = True
        else:
            for k in ("x","d","dd"):
                if k in t:
                    t[k] = False
    else:
        t["progress"] = p
        t["done"] = bool(p >= 100)
    return json.dumps(t, ensure_ascii=False)

def _clear_done_and_progress(raw):
    s = _set_progress_value(raw, 0)
    return _set_done_flag_value(s, False)

def _clear_done_keep_progress(raw):
    try:
        t = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except Exception:
        return raw
    compact = any(k in t for k in ("n","r","w","h")) and ("name" not in t)
    if compact:
        for k in ("x","d","dd"):
            if k in t:
                t[k] = False
    else:
        t["done"] = False
    return json.dumps(t, ensure_ascii=False)

def _serialize_grid(db: Session, plan_id: str):
    rows = db.query(Cell).filter(Cell.plan_id == plan_id).all()
    return [{"area": r.area, "day": int(r.day), "activities": list(r.activities or [])} for r in rows]

def _overwrite_grid(db: Session, plan_id: str, cells: List[dict]):
    db.query(Cell).filter(Cell.plan_id == plan_id).delete()
    for c in cells:
        db.add(Cell(plan_id=plan_id, area=c["area"], day=int(c["day"]), activities=list(c.get("activities") or [])))
    db.commit()

def _canon(x):
    if isinstance(x, str):
        try:
            obj = json.loads(x)
            return json.dumps(obj, sort_keys=True, ensure_ascii=False)
        except Exception:
            return x
    try:
        return json.dumps(x, sort_keys=True, ensure_ascii=False)
    except Exception:
        return str(x)

# ---------- NEW: derive "today" from latest rollover if recent ----------
def _latest_rollover_day(db: Session, plan_id: str) -> Optional[Tuple[int, datetime]]:
    log = (
        db.query(RolloverLog)
          .filter(RolloverLog.plan_id == plan_id)
          .order_by(RolloverLog.id.desc())
          .first()
    )
    if not log:
        return None
    return (int(log.to_day or 0), log.created_at)

def _today_workday_for_email(db: Session, plan_id: str, override_start_day: Optional[int] = None) -> int:
    """
    Priority:
      1) explicit ?start_day= from UI (if provided)
      2) if there's a recent rollover log (<=36h old), use its 'to_day'
      3) fall back to workday index computed from plan.start_date and today's local date
    """
    if override_start_day and override_start_day >= 1:
        return int(override_start_day)

    # try recent rollover
    lr = _latest_rollover_day(db, plan_id)
    if lr:
        to_day, created_at = lr
        try:
            # treat as recent if within 36 hours from now (local TZ)
            now_local = _now_local_dt()
            ca = created_at
            if ca.tzinfo is None:
                # created_at should be timezone-aware from DB; if not, compare naive times
                delta = now_local.replace(tzinfo=None) - ca
            else:
                delta = now_local.astimezone(ca.tzinfo) - ca
            if delta <= timedelta(hours=36) and to_day >= 1:
                return to_day
        except Exception:
            # if anything odd, still accept the to_day
            if to_day >= 1:
                return to_day

    # fallback: compute by start_date
    start_date = _get_start_date_for_plan(db, plan_id)
    return max(1, _workday_index(start_date, _today_local()))

# =========================
# Rollover (append-only)
# =========================
def _rollover_incomplete(db: Session, plan_id: str, from_day: int) -> int:
    to_day = from_day + 1
    src_rows = db.query(Cell).filter_by(plan_id=plan_id, day=from_day).all()
    moved = 0
    for row in src_rows:
        acts = list(row.activities or [])
        keep, carry = [], []
        for s in acts:
            nt = _normalize_task(s)
            if nt["done"] or nt["progress"] >= 100:
                keep.append(s)
            else:
                carry.append(s)
        row.activities = keep
        if carry:
            carry = [_clear_done_keep_progress(s) for s in carry]
            _append_to_cell(db, plan_id, row.area, to_day, carry)
            moved += len(carry)
    db.commit()
    return moved

def _rollover_incomplete_with_detail(db: Session, plan_id: str, from_day: int):
    to_day = from_day + 1
    src_rows = db.query(Cell).filter_by(plan_id=plan_id, day=from_day).all()
    moved = 0
    detail = []
    for row in src_rows:
        acts = list(row.activities or [])
        keep, carry = [], []
        for s in acts:
            nt = _normalize_task(s)
            if nt["done"] or nt["progress"] >= 100:
                keep.append(s)
            else:
                carry.append(s)
        row.activities = keep
        if carry:
            cleared = [_clear_done_keep_progress(s) for s in carry]
            _append_to_cell(db, plan_id, row.area, to_day, cleared)
            detail.append({"area": row.area, "items": cleared})
            moved += len(carry)
    db.commit()
    return moved, detail

# =========================
# Checklist write endpoints
# =========================
class ChecklistWH(BaseModel):
    area: str
    day: conint(ge=1)
    index: conint(ge=0)
    workers: Optional[int] = None
    hours: Optional[float] = None
    progress: Optional[int] = None

class WHUpdateIn(BaseModel):
    plan_id: str = DEFAULT_PLAN_ID
    items: List[ChecklistWH]

class ProgressItem(BaseModel):
    area: str
    day: conint(ge=1)
    index: conint(ge=0)
    progress: conint(ge=0, le=100)

class ProgressUpdateIn(BaseModel):
    plan_id: str = DEFAULT_PLAN_ID
    items: List[ProgressItem]

@ops.post("/checklist_mark")
def checklist_mark(payload: ChecklistUpdateIn, token: Optional[str] = None, db: Session = Depends(get_db)):
    if not CHECK_TOKEN or token != CHECK_TOKEN:
        raise HTTPException(401, "Invalid token")

    plan = db.get(Plan, payload.plan_id)
    if not plan:
        raise HTTPException(404, "Plan not found")

    for upd in payload.updates:
        row = db.query(Cell).filter_by(plan_id=payload.plan_id, area=upd.area, day=int(upd.day)).one_or_none()
        if not row:
            continue
        acts = list(row.activities or [])
        done_idx   = sorted(set(upd.done or []))
        undone_idx = sorted(set(upd.undone or []))
        for i in done_idx:
            if 0 <= i < len(acts):
                acts[i] = _set_done_flag_value(acts[i], True)
        for i in undone_idx:
            if 0 <= i < len(acts):
                acts[i] = _set_done_flag_value(acts[i], False)
        row.activities = acts
    db.commit()
    return {"ok": True, "moved": 0}

@ops.post("/checklist_update_fields")
def checklist_update_fields(payload: WHUpdateIn, token: Optional[str] = None, db: Session = Depends(get_db)):
    if not CHECK_TOKEN or token != CHECK_TOKEN:
        raise HTTPException(401, "Invalid token")

    plan = db.get(Plan, payload.plan_id)
    if not plan:
        raise HTTPException(404, "Plan not found")

    updated = 0
    for it in payload.items or []:
        row = db.query(Cell).filter_by(plan_id=payload.plan_id, area=it.area, day=int(it.day)).one_or_none()
        if not row:
            continue
        acts = list(row.activities or [])
        if it.index < 0 or it.index >= len(acts):
            continue
        try:
            t = json.loads(acts[it.index]) if isinstance(acts[it.index], str) else dict(acts[it.index])
        except Exception:
            t = {"name": str(acts[it.index]), "role": "", "workers": 0, "hours": 0, "done": False, "progress": 0}

        if it.workers is not None:
            t["workers"] = int(it.workers)
        if it.hours is not None:
            t["hours"] = float(it.hours)
        if it.progress is not None:
            p = max(0, min(100, int(it.progress)))
            t["progress"] = p
            t["done"] = bool(p >= 100)

        acts[it.index] = json.dumps(t, ensure_ascii=False)
        row.activities = acts
        updated += 1

    db.commit()
    return {"ok": True, "updated": updated}

@ops.post("/checklist_update_progress")
def checklist_update_progress(payload: ProgressUpdateIn, token: Optional[str] = None, db: Session = Depends(get_db)):
    if not CHECK_TOKEN or token != CHECK_TOKEN:
        raise HTTPException(401, "Invalid token")

    plan = db.get(Plan, payload.plan_id)
    if not plan:
        raise HTTPException(404, "Plan not found")

    updated = 0
    for it in (payload.items or []):
        row = db.query(Cell).filter_by(
            plan_id=payload.plan_id, area=it.area, day=int(it.day)
        ).one_or_none()
        if not row:
            continue
        acts = list(row.activities or [])
        if it.index < 0 or it.index >= len(acts):
            continue
        acts[it.index] = _set_progress_value(acts[it.index], int(it.progress))
        row.activities = acts
        updated += 1
    db.commit()
    return {"ok": True, "updated": updated}

# --- dedicated progress endpoint used by live checklist ---
class ChecklistProgressItem(BaseModel):
    area: str
    day: conint(ge=1)
    index: conint(ge=0)
    progress: conint(ge=0, le=100)

class ChecklistProgressUpdateIn(BaseModel):
    plan_id: str = DEFAULT_PLAN_ID
    items: List[ChecklistProgressItem]

@ops.post("/checklist_progress")
def checklist_progress(payload: ChecklistProgressUpdateIn, token: Optional[str] = None, db: Session = Depends(get_db)):
    if not CHECK_TOKEN or token != CHECK_TOKEN:
        raise HTTPException(401, "Invalid token")
    plan = db.get(Plan, payload.plan_id)
    if not plan:
        raise HTTPException(404, "Plan not found")
    updated = 0
    for it in payload.items or []:
        row = db.query(Cell).filter_by(plan_id=payload.plan_id, area=it.area, day=int(it.day)).one_or_none()
        if not row:
            continue
        acts = list(row.activities or [])
        if 0 <= it.index < len(acts):
            acts[it.index] = _set_progress_value(acts[it.index], int(it.progress))
            row.activities = acts
            updated += 1
    db.commit()
    return {"ok": True, "updated": updated}

# =========================
# Email builders + senders
# =========================

def _norm_role(raw: str) -> str:
    s = (raw or "").strip().lower()
    MAP = {
        "demolition":"Demolition",
        "civil work":"Civil",
        "civil":"Civil",
        "plumbing":"Plumbing",
        "electrical":"Electrical",
        "carpentry":"Carpentry",
        "tiling":"Tiling",
        "painting":"Painting",
        "cleaning":"Cleaning",
        "other":"Other",
        # common aliases/typos
        "plumbing work":"Plumbing",
        "electric work":"Electrical",
        "electical":"Electrical",
        "glass work":"Other",
        "metal work":"Other",
        "metal and roofing work":"Other",
    }
    return MAP.get(s, "Other")

def _iter_tasks_upto(db: Session, plan_id: str, thru_day: int):
    q = (
        db.query(Cell.area, Cell.day, Cell.activities)
          .filter(Cell.plan_id == plan_id, Cell.day <= int(thru_day))
          .order_by(Cell.day.asc(), Cell.area.asc())
    )
    for area, day, activities in q:
        for s in (activities or []):
            t = _normalize_task(s)
            yield area, int(day), t

def _simple_bar(percent: float, w: int = 80, h: int = 8, color: str = "#10b981") -> str:
    pct = max(0.0, min(100.0, float(percent)))
    return (
        f"<div style='width:{w}px;height:{h}px;background:#e5e7eb;border-radius:999px;overflow:hidden;display:inline-block;vertical-align:middle'>"
        f"<div style='width:{pct}%;height:{h}px;background:{color}'></div></div>"
        f"<span style='display:inline-block;margin-left:6px;color:#374151;font-size:12px'>{pct:.1f}%</span>"
    )

def _build_cumulative_summary(db: Session, plan_id: str, cutoff_day: int) -> Tuple[str, bytes]:
    # tallies
    per_area: Dict[str, Dict[str,int]] = {}
    per_role: Dict[str, Dict[str,int]] = {}
    pendings: Dict[str, List[dict]] = {}

    def bump(bucket: Dict[str, Dict[str,int]], key: str, state: str):
        m = bucket.setdefault(key, {"done":0,"inprog":0,"pending":0,"total":0})
        m[state] += 1; m["total"] += 1

    total_done = total_all = 0

    for area, day, t in _iter_tasks_upto(db, plan_id, cutoff_day):
        done = bool(t["done"] or (t["progress"] >= 100))
        inprog = (not done) and (int(t["progress"]) > 0)
        state = "done" if done else ("inprog" if inprog else "pending")

        bump(per_area, area, state)
        bump(per_role, _norm_role(t["role"]), state)

        total_all += 1
        if done: total_done += 1

        if not done:
            pendings.setdefault(area, []).append({
                "day": day, "task": t["name"], "role": t["role"], "progress": int(t["progress"])
            })

    # build HTML
    def pct(a,b): return (100.0*a/b) if b>0 else 0.0

   def tabulate(mapping: Dict[str, Dict[str,int]], title: str) -> str:
        rows = []
        rows.append(f"<h3>{title}</h3>")
        rows.append("<table border='0' cellpadding='6' cellspacing='0' "
                    "style='border-collapse:collapse;font-family:Arial;font-size:14px;width:100%'>")
        rows.append("<tr style='background:#f3f4f6'>"
                    "<th align='left'>Name</th><th>Done</th><th>In-prog</th><th>Pending</th>"
                    "<th>Total</th><th>% Done</th><th align='left'>Mix</th></tr>")
        for name in sorted(mapping.keys()):
            m = mapping[name]
            total = max(1, m['total'])
            pct_done   = round(100.0 * m['done']   / total, 1)
            pct_inprog = round(100.0 * m['inprog'] / total, 1)
            pct_pend   = round(100.0 * m['pending']/ total, 1)
            bar = (
              "<div style='width:160px;background:#e5e7eb;border-radius:8px;overflow:hidden;height:8px;display:inline-block'>"
              f"<div style='width:{pct_done}%;height:8px;background:#10b981;display:inline-block'></div>"
              f"<div style='width:{pct_inprog}%;height:8px;background:#f59e0b;display:inline-block'></div>"
              f"<div style='width:{pct_pend}%;height:8px;background:#9ca3af;display:inline-block'></div>"
              "</div>"
            )
            rows.append(
                f"<tr>"
                f"<td>{name}</td>"
                f"<td align='center'>{m['done']}</td>"
                f"<td align='center'>{m['inprog']}</td>"
                f"<td align='center'>{m['pending']}</td>"
                f"<td align='center'>{m['total']}</td>"
                f"<td align='center'>{pct_done}%</td>"
                f"<td>{bar}</td>"
                f"</tr>"
            )
        if not mapping:
            rows.append("<tr><td colspan='7' style='color:#6b7280'>No tasks scheduled yet.</td></tr>")
        rows.append("</table>")
        return "\n".join(rows)


    # pending lists (top 5 per area, oldest first)
    pend_html = ["<h3 style='margin:12px 0 6px 0'>Key pending items by area (oldest first)</h3>"]
    if not pendings:
        pend_html.append("<p style='color:#6b7280'>None.</p>")
    else:
        start_date = _get_start_date_for_plan(db, plan_id)
        pend_html.append("<div>")
        for area in sorted(pendings.keys()):
            items = sorted(pendings[area], key=lambda x:(x["day"], x["task"]))[:5]
            lis = []
            for it in items:
                dt = _date_for_day(start_date, it["day"]).strftime("%d %b")
                lis.append(f"<li>Day {it['day']} ({dt}) — {it['task']} <span style='color:#6b7280'>( {it['role'] or '—'}, {it['progress']}% )</span></li>")
            pend_html.append(f"<p style='margin:10px 0 4px 0'><b>{area}</b></p><ul>{''.join(lis) or '<li>—</li>'}</ul>")
        pend_html.append("</div>")

    overall_pct = pct(total_done, total_all)
    overall_bar = _simple_bar(overall_pct, w=160, h=10)

    html = f"""
    <div style="font-family:Arial,Helvetica,sans-serif">
      <h2 style="margin:0 0 8px 0">Daily Summary (cumulative up to yesterday)</h2>
      <div style="margin:4px 0 12px 0"><b>Overall progress:</b> {overall_bar}</div>
      {tabulate(per_area, "Per-area cumulative")}
      <div style="height:12px"></div>
      {tabulate(per_role, "Per-trade cumulative")}
      <div style="height:12px"></div>
      {''.join(pend_html)}
    </div>
    """.strip()

    # CSV (single file with "section" discriminator)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["section","name","done","in_progress","pending","total","pct_done"])
    for name, m in sorted(per_area.items()):
        w.writerow(["area", name, m["done"], m["inprog"], m["pending"], m["total"],
                    (round(100.0*m["done"]/m["total"],1) if m["total"] else "")])
    for name, m in sorted(per_role.items()):
        w.writerow(["role", name, m["done"], m["inprog"], m["pending"], m["total"],
                    (round(100.0*m["done"]/m["total"],1) if m["total"] else "")])
    # pending rows
    w.writerow([])
    w.writerow(["section","area","day","task","role","progress"])
    for area, items in sorted(pendings.items()):
        items = sorted(items, key=lambda x:(x["day"], x["task"]))[:5]
        for it in items:
            w.writerow(["pending", area, it["day"], it["task"], it["role"], it["progress"]])
    csv_bytes = buf.getvalue().encode("utf-8")
    return html, csv_bytes

def _build_carryover_block(db: Session, plan_id: str) -> str:
    log = (
        db.query(RolloverLog)
          .filter(RolloverLog.plan_id == plan_id)
          .order_by(RolloverLog.id.desc())
          .first()
    )
    if not log or not (log.moved or []):
        return ""
    # hide if older than ~36h
    try:
        if log.created_at and (datetime.utcnow() - log.created_at.replace(tzinfo=None)) > timedelta(hours=36):
            return ""
    except Exception:
        pass

    # summarize by area
    by_area: Dict[str, int] = {}
    for e in (log.moved or []):
        by_area[e.get("area","")] = by_area.get(e.get("area",""), 0) + len(e.get("items") or [])

    rows = []
    rows.append("<h3>Carry-overs from last night’s rollover</h3>")
    rows.append(f"<p style='margin:0 0 8px 0;color:#374151'>Moved {sum(by_area.values())} item(s) from Day {log.from_day} → {log.to_day}.</p>")
    rows.append("<table border='0' cellpadding='6' cellspacing='0' style='border-collapse:collapse;font-family:Arial;font-size:14px;width:100%'>")
    rows.append("<tr style='background:#f3f4f6'><th align='left'>Area</th><th align='left'>Items</th></tr>")
    for area in sorted(by_area.keys()):
        rows.append(f"<tr><td>{area or '—'}</td><td>{by_area[area]}</td></tr>")
    rows.append("</table>")
    return "\n".join(rows)

def _build_three_day_report(db: Session, plan_id: str, start_day: int, span: int = 3):
    start_day = max(1, int(start_day))
    start_date = _get_start_date_for_plan(db, plan_id)
    cells = _fetch_cells_range(db, plan_id, start_day, start_day + span - 1)

    rows = []
    html_rows = []  # (day, area, task, role, w, h, progress, index)

    for c in cells:
        for idx, s in enumerate(c["activities"] or []):
            nt = _normalize_task(s)
            rows.append({
                "date": _date_for_day(start_date, c["day"]).isoformat(),
                "day": c["day"],
                "area": c["area"],
                "task": nt["name"],
                "role": nt["role"],
                "workers": nt["workers"],
                "hours": nt["hours"],
                "progress": nt["progress"],
                "done": "yes" if nt["done"] else ""
            })
            html_rows.append({
                "day": c["day"],
                "area": c["area"],
                "task": nt["name"],
                "role": nt["role"],
                "workers": nt["workers"],
                "hours": nt["hours"],
                "progress": nt["progress"],
                "index": idx
            })

    # CSV
    csv_buf = io.StringIO()
    writer = csv.DictWriter(csv_buf, fieldnames=["date", "day", "area", "task", "role", "workers", "hours", "progress", "done"])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    # HTML
    by_day = {}
    for r in html_rows:
        by_day.setdefault(int(r["day"]), []).append(r)

    def h(s):
        return (str(s) if s is not None else "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def quick_link(day, area, index, pct, label):
        if not BACKEND_PUBLIC_BASE:
            return f"<span class='muted'>{label}</span>"
        q = {
            "plan_id": plan_id, "day": day, "area": area,
            "index": index, "progress": pct, "token": CHECK_TOKEN
        }
        return f"<a href='{BACKEND_PUBLIC_BASE}/ops/progress_link?{urlencode(q, quote_via=quote_plus)}' " \
               f"style='display:inline-block;margin:2px 6px;padding:6px 10px;border:1px solid #d1d5db;border-radius:8px;text-decoration:none'>{h(label)}</a>"

    sections = []
    for d in range(start_day, start_day + span):
        the_date = _date_for_day(start_date, d).strftime("%d %b %Y")
        section = [
            f"<h3>Day {d} — {the_date}</h3>",
            "<table border='0' cellpadding='6' cellspacing='0' style='border-collapse:collapse;font-family:Arial;font-size:14px;width:100%'>",
            "<tr style='background:#f3f4f6'><th align='left'>Area</th><th align='left'>Task</th><th align='left'>Role</th>"
            "<th align='left'>Workers</th><th align='left'>Hours</th><th align='left'>Progress</th></tr>"
        ]
        for r in by_day.get(d, []):
            btns = " ".join([
                quick_link(d, r["area"], r["index"], 0, "0%"),
                quick_link(d, r["area"], r["index"], 25, "25%"),
                quick_link(d, r["area"], r["index"], 50, "50%"),
                quick_link(d, r["area"], r["index"], 75, "75%"),
                quick_link(d, r["area"], r["index"], 100, "100%"),
            ])
            left = f"{max(0, 100 - int(r['progress']))}% left"
            section.append(
                f"<tr><td>{h(r['area'])}</td><td>{h(r['task'])}</td><td>{h(r['role'])}</td>"
                f"<td>{r['workers']}</td><td>{r['hours']}</td>"
                f"<td>{btns}<div class='muted' style='margin-top:4px'>{left}</div></td></tr>"
            )
        if not by_day.get(d):
            section.append("<tr><td colspan='6' style='color:#6b7280'>No tasks</td></tr>")
        section.append("</table>")
        sections.append("\n".join(section))

    checklist_link = (
        FRONTEND_PUBLIC_BASE and
        f"{FRONTEND_PUBLIC_BASE}/checklist.html?plan={plan_id}&startDay={start_day}&days={span}"
        + (f"&t={CHECK_TOKEN}" if CHECK_TOKEN else "")
        + (f"&api={BACKEND_PUBLIC_BASE}" if BACKEND_PUBLIC_BASE else "")
    )

    html = f"""
    <div style="font-family:Arial,Helvetica,sans-serif">
      <p>Good morning! Here is the {span}-day plan.</p>
      {('<p><a href="'+h(checklist_link)+'" style="display:inline-block;background:#2563eb;color:#fff;padding:10px 14px;border-radius:6px;text-decoration:none">Open live checklist</a></p>' if checklist_link else '')}
      {''.join(sections)}
    </div>
    """.strip()

    return html, csv_bytes

def _send_via_smtp(subject: str, html: str, attachments: List[Tuple[str, bytes]]):
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and (DAILY_TO or RESEND_TO)):
        raise RuntimeError("SMTP not configured (SMTP_HOST/USER/PASS and recipients)")
    msg = EmailMessage()
    msg["From"] = SMTP_FROM or SMTP_USER
    msg["To"]   = ", ".join(DAILY_TO) if DAILY_TO else RESEND_TO
    msg["Subject"] = subject
    msg.set_content("HTML + attachments.")
    msg.add_alternative(html, subtype="html")
    for fname, content in (attachments or []):
        msg.add_attachment(content, maintype="text", subtype="csv", filename=fname)
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)

def _send_via_resend(subject: str, html: str, attachments: List[Tuple[str, bytes]]):
    if not RESEND_API_KEY:
        raise RuntimeError("RESEND_API_KEY missing")
    to_source = RESEND_TO or ",".join(DAILY_TO)
    to_list = [x.strip() for x in to_source.replace(";", ",").split(",") if x.strip()]
    if not to_list:
        raise RuntimeError("RESEND_TO (or DAILY_TO) not set")
    payload = {
        "from": RESEND_FROM or "Remodel Planner <onboarding@resend.dev>",
        "to": to_list,
        "subject": subject,
        "html": html,
        "attachments": [
            {"filename": fname, "content": base64.b64encode(content).decode("ascii")}
            for (fname, content) in (attachments or [])
        ],
    }
    headers = {
        "Authorization": f"Bearer {RESEND_API_KEY}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=30) as client:
        r = client.post("https://api.resend.com/emails", json=payload, headers=headers)
        r.raise_for_status()
        return r.json()

def _send_email(subject: str, html: str, attachments: List[Tuple[str, bytes]]):
    if RESEND_API_KEY:
        return _send_via_resend(subject, html, attachments or [])
    return _send_via_smtp(subject, html, attachments or [])

# =========================
# Routes (core)
# =========================
@app.get("/")
def root():
    return {
        "ok": True,
        "service": "remodel-backend",
        "docs": "/docs",
        "now": datetime.utcnow().isoformat(),
        "email_mode": "resend" if RESEND_API_KEY else ("smtp" if SMTP_HOST else "none"),
    }

@app.post("/plans", dependencies=[Depends(require_key)], response_model=PlanUpsert)
def create_or_update_plan(payload: PlanUpsert, db: Session = Depends(get_db)):
    p = db.get(Plan, payload.id)
    if p:
        p.name = payload.name
        p.total_days = payload.total_days
        p.areas = payload.areas
        p.allow_multiple = payload.allow_multiple
        p.start_date = payload.start_date
    else:
        p = Plan(**payload.dict())
        db.add(p)
    db.commit()
    return payload

@app.get("/plans/{plan_id}", response_model=PlanUpsert)
def get_plan(plan_id: str, db: Session = Depends(get_db)):
    p = db.get(Plan, plan_id)
    if not p:
        raise HTTPException(404, "Plan not found")
    return PlanUpsert(
        id=p.id, name=p.name, total_days=p.total_days, areas=p.areas,
        allow_multiple=p.allow_multiple, start_date=p.start_date
    )

@app.get("/plans/{plan_id}/grid")
def get_grid(plan_id: str, db: Session = Depends(get_db)):
    p = db.get(Plan, plan_id)
    if not p:
        raise HTTPException(404, "Plan not found")
    rows = db.query(Cell).filter(Cell.plan_id == plan_id).all()
    return {
        "allowMultiple": p.allow_multiple,
        "cells": [{"area": c.area, "day": c.day, "activities": c.activities} for c in rows],
        "start_date": (p.start_date.isoformat() if p.start_date else None),
    }

@app.put("/plans/{plan_id}/grid", dependencies=[Depends(require_write)])
def upsert_grid(plan_id: str, payload: GridIn, db: Session = Depends(get_db)):
    p = db.get(Plan, plan_id)
    if not p:
        raise HTTPException(404, "Plan not found")
    if payload.allow_multiple is not None:
        p.allow_multiple = payload.allow_multiple
    for c in payload.cells:
        row = db.query(Cell).filter_by(plan_id=plan_id, area=c.area, day=c.day).one_or_none()
        if row:
            row.activities = c.activities
        else:
            db.add(Cell(plan_id=plan_id, area=c.area, day=c.day, activities=c.activities))
    db.commit()
    return {"ok": True}

# ---- Items & Costs endpoints ----
@app.get("/plans/{plan_id}/extras")
def get_extras(plan_id: str, day: conint(ge=1) = Query(...), db: Session = Depends(get_db)):
    row = db.query(Extras).filter_by(plan_id=plan_id, day=int(day)).one_or_none()
    return row.items if row else []

@app.put("/plans/{plan_id}/extras", dependencies=[Depends(require_write)])
def put_extras(plan_id: str, payload: ExtrasUpsert, day: conint(ge=1) = Query(...), db: Session = Depends(get_db)):
    row = db.query(Extras).filter_by(plan_id=plan_id, day=int(day)).one_or_none()
    data = []
    for it in (payload.items or []):
        q = float(it.qty or 0)
        r = float(it.rate or 0)
        amt = round(q * r, 2)
        data.append({"item": it.item, "qty": q, "rate": r, "amount": amt})
    if row:
        row.items = data
    else:
        db.add(Extras(plan_id=plan_id, day=int(day), items=data))
    db.commit()
    total = round(sum(x.get("amount", 0) for x in data), 2)
    return {"ok": True, "count": len(data), "total": total}

# =========================
# Ops: status + email + rollover
# =========================
@ops.get("/ping")
def ops_ping():
    return {
        "ok": True,
        "ts": datetime.utcnow().isoformat(),
        "has_email_route": True,
        "email_mode": "resend" if RESEND_API_KEY else ("smtp" if SMTP_HOST else "none"),
        "smtp_configured": bool(SMTP_HOST and SMTP_USER and SMTP_PASS and (DAILY_TO or RESEND_TO)),
        "resend_configured": bool(RESEND_API_KEY),
    }

@ops.get("/email_ping", dependencies=[Depends(require_key)])
def email_ping():
    try:
        _send_email(
            subject="Email ping",
            html="<b>Hello</b> from Remodel Planner.",
            attachments=[("ping.csv", b"date,ok\n2025-09-13,yes\n")],
        )
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@ops.get("/send_daily_email", dependencies=[Depends(require_key)])
def send_daily_email(
    plan_id: str = DEFAULT_PLAN_ID,
    span: int = 3,
    start_day: Optional[int] = Query(None)  # optional override from UI
):
    """
    Builds 'Summary + (Today+next span-1) checklist' and emails it.
    We compute the authoritative 'today_idx' using server workday logic.
    If the UI sends a start_day that's close (±1) we allow it; otherwise we ignore it.
    """
    try:
        with SessionLocal() as db:
            start_date = _get_start_date_for_plan(db, plan_id)

            # server-truth for today's workday index (Sundays/holidays skipped)
            server_today = max(1, _workday_index(start_date, _today_local()))

            # if UI sent something wildly off (like raw calendar diff), ignore it
            requested = int(start_day) if (start_day and start_day >= 1) else None
            if requested is not None and abs(requested - server_today) <= 1:
                today_idx = requested
            else:
                today_idx = server_today

            cutoff = max(0, today_idx - 1)  # cumulative up to yesterday

            # 1) Summary (cumulative up to 'cutoff')
            summary_html, summary_csv = _build_cumulative_summary(db, plan_id, cutoff)

            # 2) Carry-overs (last midnight)
            carry_html = _build_carryover_block(db, plan_id)

            # 3) 3-day checklist starting today
            checklist_html, checklist_csv = _build_three_day_report(db, plan_id, today_idx, span=span)

            # Combine HTML
            today_date = _date_for_day(start_date, today_idx).strftime("%d %b %Y")
            html = f"""
              <div style="font-family:Arial,Helvetica,sans-serif">
                <h2 style="margin:0 0 12px 0">Daily Report — Day {today_idx} ({today_date})</h2>
                {summary_html}
                <div style="height:16px"></div>
                {carry_html}
                <div style="height:16px"></div>
                <h2 style="margin:12px 0 8px 0">Checklist (Today + next {span-1} day[s])</h2>
                {checklist_html}
              </div>
            """.strip()

            _send_email(
                subject=f"[{plan_id}] Daily report (Summary + {span}-day checklist) — Day {today_idx}",
                html=html,
                attachments=[
                    (f"checklist_day{today_idx}.csv", checklist_csv),
                    (f"summary_upto_day{cutoff}.csv", summary_csv),
                ],
            )
            return {"ok": True, "day": today_idx}
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@ops.post("/rollover", dependencies=[Depends(require_key)])
def rollover(plan_id: str = DEFAULT_PLAN_ID):
    try:
        with SessionLocal() as db:
            start_date = _get_start_date_for_plan(db, plan_id)
            from_day = max(1, _workday_index(start_date, _today_local()))
            moved = _rollover_incomplete(db, plan_id, from_day)
            return {"ok": True, "moved": moved, "from_day": from_day, "to_day": from_day + 1}
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@ops.get("/progress_link", response_class=HTMLResponse)
def progress_link(
    plan_id: str,
    area: str,
    day: conint(ge=1),
    index: conint(ge=0),
    progress: conint(ge=0, le=100),
    token: str,
    db: Session = Depends(get_db)
):
    if not CHECK_TOKEN or token != CHECK_TOKEN:
        raise HTTPException(401, "Invalid token")
    row = db.query(Cell).filter_by(plan_id=plan_id, area=area, day=int(day)).one_or_none()
    if not row:
        raise HTTPException(404, "Cell not found")
    acts = list(row.activities or [])
    if index < 0 or index >= len(acts):
        raise HTTPException(400, "Bad index")
    acts[index] = _set_progress_value(acts[index], int(progress))
    row.activities = acts
    db.commit()
    msg = f"Updated: {area}, Day {day}, item #{index} → {progress}%"
    link = ""
    if FRONTEND_PUBLIC_BASE and BACKEND_PUBLIC_BASE:
        link = f"<p><a href='{FRONTEND_PUBLIC_BASE}/checklist.html?plan={plan_id}&startDay={day}&days=3&t={CHECK_TOKEN}&api={BACKEND_PUBLIC_BASE}'>Open live checklist</a></p>"
    return f"<div style='font-family:Arial;padding:16px'><p>{msg}</p>{link}<p>You can close this tab.</p></div>"

# =========================
# Snapshot/reset/undo endpoints
# =========================
@ops.post("/snapshot_save", dependencies=[Depends(require_key)])
def snapshot_save(plan_id: str = DEFAULT_PLAN_ID):
    with SessionLocal() as db:
        cells = _serialize_grid(db, plan_id)
        snap = db.get(Snapshot, plan_id)
        if snap:
            snap.grid = cells
        else:
            db.add(Snapshot(plan_id=plan_id, grid=cells))
        db.commit()
        total = sum(len(c.get("activities") or []) for c in cells)
        return {"ok": True, "plan_id": plan_id, "cells": len(cells), "tasks": total}

@ops.post("/snapshot_reset", dependencies=[Depends(require_key)])
def snapshot_reset(plan_id: str = DEFAULT_PLAN_ID, reset_done: bool = True):
    with SessionLocal() as db:
        snap = db.get(Snapshot, plan_id)
        if not snap:
            raise HTTPException(404, "No snapshot for this plan")
        cells = snap.grid or []
        if reset_done:
            def _clear_flags(acts):
                return [_clear_done_and_progress(s) for s in (acts or [])]
            cells = [{"area": c["area"], "day": c["day"], "activities": _clear_flags(c.get("activities"))} for c in cells]
        _overwrite_grid(db, plan_id, cells)
        db.query(RolloverLog).filter(RolloverLog.plan_id == plan_id).delete()
        db.commit()
        return {"ok": True, "plan_id": plan_id, "restored_cells": len(cells)}

@ops.post("/checklist_reset_all", dependencies=[Depends(require_key)])
def checklist_reset_all(plan_id: str = DEFAULT_PLAN_ID):
    with SessionLocal() as db:
        rows = db.query(Cell).filter_by(plan_id=plan_id).all()
        for row in rows:
            row.activities = [_clear_done_and_progress(s) for s in (row.activities or [])]
        db.commit()
    return {"ok": True, "plan_id": plan_id}

@ops.post("/rollover_logged", dependencies=[Depends(require_key)])
def rollover_logged(plan_id: str = DEFAULT_PLAN_ID):
    with SessionLocal() as db:
        start = _get_start_date_for_plan(db, plan_id)
        from_day = max(1, _workday_index(start, _today_local()))
        moved, detail = _rollover_incomplete_with_detail(db, plan_id, from_day)
        log = RolloverLog(plan_id=plan_id, from_day=from_day, to_day=from_day+1, moved=detail)
        db.add(log)
        db.commit()
        return {"ok": True, "log_id": log.id, "moved": moved, "from_day": from_day, "to_day": from_day + 1}

@ops.post("/unrollover_last", dependencies=[Depends(require_key)])
def unrollover_last(plan_id: str = DEFAULT_PLAN_ID):
    with SessionLocal() as db:
        log = db.query(RolloverLog).filter(RolloverLog.plan_id == plan_id)\
                                   .order_by(RolloverLog.id.desc()).first()
        if not log:
            raise HTTPException(404, "No rollover to undo")

        from_day, to_day = log.from_day, log.to_day
        detail = log.moved or []

        for entry in detail:
            area = entry["area"]
            items = entry.get("items") or []

            dest = db.query(Cell).filter_by(plan_id=plan_id, area=area, day=to_day).one_or_none()
            dest_list = list(dest.activities or []) if dest else []
            dest_can = [_canon(s) for s in dest_list]

            for moved_item in items:
                mc = _canon(moved_item)
                try:
                    idx = dest_can.index(mc)
                    dest_can.pop(idx)
                    dest_list.pop(idx)
                except ValueError:
                    pass

            if dest:
                dest.activities = dest_list

            src = db.query(Cell).filter_by(plan_id=plan_id, area=area, day=from_day).one_or_none()
            if src:
                src.activities = list(src.activities or []) + items
            else:
                db.add(Cell(plan_id=plan_id, area=area, day=from_day, activities=list(items)))

        db.delete(log)
        db.commit()

        return {
            "ok": True,
            "undone_log_id": log.id,
            "from_day": from_day,
            "to_day": to_day,
            "moved_back": sum(len(e.get("items") or []) for e in detail)
        }

@ops.post("/set_start_date", dependencies=[Depends(require_key)])
def set_start_date(plan_id: str = DEFAULT_PLAN_ID, start: str = ""):
    try:
        d = datetime.strptime(start, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(400, "start must be YYYY-MM-DD")
    with SessionLocal() as db:
        p = db.get(Plan, plan_id)
        if not p:
            raise HTTPException(404, "Plan not found")
        p.start_date = d
        db.commit()
        return {"ok": True, "plan_id": plan_id, "start_date": str(d)}

# ---- scheduler: 07:00 email & 00:00 rollover in configured TIMEZONE ----
app.include_router(ops)

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    try:
        from zoneinfo import ZoneInfo
        TZ = ZoneInfo(TIMEZONE)
    except Exception:
        TZ = None
    HAVE_SCHEDULER = True
except Exception:
    HAVE_SCHEDULER = False

if HAVE_SCHEDULER:
    try:
        scheduler = BackgroundScheduler()

        def job_email():
            try:
                with SessionLocal() as db:
                    # NEW: use same “today” logic as manual route
                    today_day = _today_workday_for_email(db, DEFAULT_PLAN_ID, None)
                    start_date = _get_start_date_for_plan(db, DEFAULT_PLAN_ID)
                    cutoff = max(0, today_day - 1)
                    summary_html, summary_csv = _build_cumulative_summary(db, DEFAULT_PLAN_ID, cutoff)
                    carry_html = _build_carryover_block(db, DEFAULT_PLAN_ID)
                    checklist_html, checklist_csv = _build_three_day_report(db, DEFAULT_PLAN_ID, today_day, span=3)

                    today_date = _date_for_day(start_date, today_day).strftime("%d %b %Y")
                    html = f"""
                      <div style="font-family:Arial,Helvetica,sans-serif">
                        <h2 style="margin:0 0 12px 0">Daily Report — Day {today_day} ({today_date})</h2>
                        {summary_html}
                        <div style="height:16px"></div>
                        {carry_html}
                        <div style="height:16px"></div>
                        <h2 style="margin:12px 0 8px 0">Checklist (Today + next 2 days)</h2>
                        {checklist_html}
                      </div>
                    """.strip()

                    _send_email(
                        subject=f"[{DEFAULT_PLAN_ID}] Daily report (Summary + 3-day checklist) — Day {today_day}",
                        html=html,
                        attachments=[
                            (f"checklist_day{today_day}.csv", checklist_csv),
                            (f"summary_upto_day{cutoff}.csv", summary_csv),
                        ],
                    )
            except Exception as e:
                print("Morning email failed:", e)

        def job_rollover():
            try:
                with SessionLocal() as db:
                    start_date = _get_start_date_for_plan(db, DEFAULT_PLAN_ID)
                    today = _today_local()
                    yday = today - timedelta(days=1)
                    if _is_off(yday):
                        print(f"Rollover: yesterday {yday} was OFF – skipping.")
                        return
                    from_day = _workday_index(start_date, yday)
                    moved = _rollover_incomplete(db, DEFAULT_PLAN_ID, from_day)
                    # log it
                    log = RolloverLog(plan_id=DEFAULT_PLAN_ID, from_day=from_day, to_day=from_day+1, moved=[])
                    db.add(log); db.commit()
                    print(f"Rollover moved {moved} item(s) from Day {from_day} → {from_day + 1}.")
            except Exception as e:
                print("Evening rollover failed:", e)

        scheduler.add_job(job_email,    CronTrigger(hour=7,  minute=0, timezone=TZ))
        scheduler.add_job(job_rollover, CronTrigger(hour=0,  minute=0, timezone=TZ))
        scheduler.start()
    except Exception as e:
        print("Scheduler not started:", e)
# ---------- end app.py ----------
