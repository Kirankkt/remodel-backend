F
import httpx
from fastapi import FastAPI, Depends, HTTPException, Header, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, conint

from sqlalchemy import (
    create_engine, Column, Integer, String, JSON, Boolean,
    DateTime, UniqueConstraint, ForeignKey, func
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session, relationship

# =========================
# ENV / CONFIG
# =========================
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./schedule.db")
API_KEY      = os.getenv("API_KEY", "changeme")  # must match X-API-Key header
CHECK_TOKEN = os.getenv("CHECK_TOKEN", "RMETVM")  # shared-secret for checklist writes


# Email + report options
TIMEZONE            = os.getenv("TIMEZONE", "Asia/Kolkata")
DEFAULT_PLAN_ID     = os.getenv("DEFAULT_PLAN_ID", "default")
PROJECT_START_DATE  = os.getenv("PROJECT_START_DATE")  # yyyy-mm-dd, optional
FRONTEND_PUBLIC_BASE= os.getenv("FRONTEND_PUBLIC_BASE", "")

# SMTP (fallback)
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER or "")
DAILY_TO  = [e.strip() for e in os.getenv("DAILY_TO", "").split(",") if e.strip()]

# Resend (preferred on Railway)
RESEND_API_KEY = os.getenv("RESEND_API_KEY")         # e.g. re_XXXXXXXXXXXXXXXX
RESEND_FROM    = os.getenv("RESEND_FROM", SMTP_FROM or (SMTP_USER or ""))
RESEND_TO      = os.getenv("RESEND_TO", ",".join(DAILY_TO))

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
    id = Column(String, primary_key=True)  # e.g., "default"
    name = Column(String, nullable=False)
    total_days = Column(Integer, default=60)
    areas = Column(JSON, default=[])
    allow_multiple = Column(Boolean, default=False)
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


Base.metadata.create_all(engine)

# =========================
# FastAPI app
# =========================
app = FastAPI(title="Remodel Planner API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

def require_key(x_api_key: Optional[str] = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Invalid API key")

# =========================
# Schemas
# =========================
class PlanUpsert(BaseModel):
    id: str = "default"
    name: str = "Default Plan"
    total_days: conint(ge=1) = 60
    areas: List[str]
    allow_multiple: bool = False

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
    done: List[int] = []      # task indexes to mark done
    undone: List[int] = []    # task indexes to mark NOT done

class ChecklistUpdateIn(BaseModel):
    plan_id: str = DEFAULT_PLAN_ID
    updates: List[ChecklistCellUpdate]

@ops.post("/checklist_mark")
def checklist_mark(payload: ChecklistUpdateIn, token: Optional[str] = None, db: Session = Depends(get_db)):
    if not CHECK_TOKEN or token != CHECK_TOKEN:
        raise HTTPException(401, "Invalid token")

    for upd in payload.updates:
        row = db.query(Cell).filter_by(plan_id=payload.plan_id, area=upd.area, day=upd.day).one_or_none()
        if not row:
            continue
        acts = list(row.activities or [])

        for idx in set(upd.done or []):
            if 0 <= idx < len(acts):
                raw = acts[idx]
                # preserve style if possible (compact vs verbose)
                try:
                    d = json.loads(raw) if isinstance(raw, str) else dict(raw)
                except Exception:
                    d = {"name": str(raw)}
                # decide output keys
                style_compact = any(k in d for k in ("n","r","w","h")) and not ("name" in d)
                if style_compact:
                    d["x"] = True
                else:
                    d["done"] = True
                acts[idx] = json.dumps(d, ensure_ascii=False)

        for idx in set(upd.undone or []):
            if 0 <= idx < len(acts):
                raw = acts[idx]
                try:
                    d = json.loads(raw) if isinstance(raw, str) else dict(raw)
                except Exception:
                    d = {"name": str(raw)}
                style_compact = any(k in d for k in ("n","r","w","h")) and not ("name" in d)
                if style_compact:
                    d["x"] = False
                    d["d"] = False
                    d["dd"] = False
                else:
                    d["done"] = False
                acts[idx] = json.dumps(d, ensure_ascii=False)

        row.activities = acts

    db.commit()
    return {"ok": True}


# =========================
# Utilities: dates, grid helpers
# =========================
def _today_local() -> date:
    # simple local date; OK for IST
    return datetime.now().date()

def _get_start_date_for_plan(db: Session, plan_id: str) -> date:
    if PROJECT_START_DATE:
        return datetime.strptime(PROJECT_START_DATE, "%Y-%m-%d").date()
    return _today_local()

def _day_for_date(start: date, d: date) -> int:
    return (d - start).days + 1

def _date_for_day(start: date, day: int) -> date:
    return start + timedelta(days=day - 1)

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

# =========================
# Task normalization (NEW)
# =========================
def _normalize_task(obj_or_json):
    """
    Accepts an activity either as a JSON string or a dict.
    Returns a dict with unified keys: name, role, workers, hours, done.
    Supports both verbose keys (name/role/workers/hours/done)
    and compact keys (n/r/w/h/d or x or dd).
    """
    try:
        t = json.loads(obj_or_json) if isinstance(obj_or_json, str) else dict(obj_or_json or {})
    except Exception:
        return {"name": str(obj_or_json), "role": "", "workers": 0, "hours": 0, "done": False}

    name = t.get("name") or t.get("task") or t.get("n") or ""
    role = t.get("role") or t.get("r") or ""
    workers = t.get("workers", t.get("w", 0)) or 0
    hours = t.get("hours", t.get("h", 0)) or 0

    # Any of these flags means done
    done = bool(t.get("done") or t.get("d") or t.get("x") or t.get("dd"))

    return {
        "name": name,
        "role": role,
        "workers": int(workers),
        "hours": int(hours),
        "done": done,
    }

# =========================
# Email builders + senders
# =========================
def _build_three_day_report(db: Session, plan_id: str, start_day: int, span: int = 3):
    start_date = _get_start_date_for_plan(db, plan_id)
    cells = _fetch_cells_range(db, plan_id, start_day, start_day + span - 1)

    rows = []
    for c in cells:
        for s in (c["activities"] or []):
            nt = _normalize_task(s)
            rows.append({
                "date": _date_for_day(start_date, c["day"]).isoformat(),
                "day": c["day"],
                "area": c["area"],
                "task": nt["name"],
                "role": nt["role"],
                "workers": nt["workers"],
                "hours": nt["hours"],
                "done": "yes" if nt["done"] else ""
            })

    # CSV
    csv_buf = io.StringIO()
    writer = csv.DictWriter(csv_buf, fieldnames=["date","day","area","task","role","workers","hours","done"])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    # HTML
    by_day = {}
    for r in rows:
        by_day.setdefault(int(r["day"]), []).append(r)

    def h(s): return (s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    sections = []
    for d in range(start_day, start_day + span):
        the_date = _date_for_day(start_date, d).strftime("%d %b %Y")
        section = [f"<h3>Day {d} — {the_date}</h3>",
                   "<table border='1' cellpadding='6' cellspacing='0' style='border-collapse:collapse;font-family:Arial;font-size:14px'>",
                   "<tr style='background:#f3f4f6'><th>Area</th><th>Task</th><th>Role</th><th>Workers</th><th>Hours</th></tr>"]
        for r in by_day.get(d, []):
            section.append(f"<tr><td>{h(r['area'])}</td><td>{h(r['task'])}</td><td>{h(r['role'])}</td><td>{r['workers']}</td><td>{r['hours']}</td></tr>")
        if not by_day.get(d):
            section.append("<tr><td colspan='5' style='color:#6b7280'>No tasks</td></tr>")
        section.append("</table>")
        sections.append("\n".join(section))

checklist_link = (
    FRONTEND_PUBLIC_BASE and
    f"{FRONTEND_PUBLIC_BASE}/checklist.html?plan={plan_id}&startDay={start_day}&days={span}" +
    (f"&t={CHECK_TOKEN}" if CHECK_TOKEN else "")
)

    html = f"""
    <div style="font-family:Arial,Helvetica,sans-serif">
      <p>Good morning! Here is the {span}-day plan.</p>
      {('<p><a href="'+h(checklist_link)+'" style="display:inline-block;background:#2563eb;color:#fff;padding:10px 14px;border-radius:6px;text-decoration:none">Open live checklist</a></p>' if checklist_link else '')}
      {''.join(sections)}
    </div>
    """.strip()

    return html, csv_bytes

def _send_via_smtp(subject: str, html: str, csv_bytes: bytes, csv_name="tasks.csv"):
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and (DAILY_TO or RESEND_TO)):
        raise RuntimeError("SMTP not configured (SMTP_HOST/USER/PASS and recipients)")
    msg = EmailMessage()
    msg["From"] = SMTP_FROM or SMTP_USER
    msg["To"]   = ", ".join(DAILY_TO) if DAILY_TO else RESEND_TO
    msg["Subject"] = subject
    msg.set_content("HTML + CSV attached.")
    msg.add_alternative(html, subtype="html")
    msg.add_attachment(csv_bytes, maintype="text", subtype="csv", filename=csv_name)
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)

def _send_via_resend(subject: str, html: str, csv_bytes: bytes, csv_name="tasks.csv"):
    if not RESEND_API_KEY:
        raise RuntimeError("RESEND_API_KEY missing")
    to_list = [x.strip() for x in (RESEND_TO.split(",") if RESEND_TO else [])]
    if not to_list:
        raise RuntimeError("RESEND_TO (or DAILY_TO) not set")
    payload = {
        "from": RESEND_FROM or "Remodel Planner <onboarding@resend.dev>",
        "to": to_list,
        "subject": subject,
        "html": html,
        "attachments": [{
            "filename": csv_name,
            "content": base64.b64encode(csv_bytes).decode("ascii"),
        }],
    }
    headers = {"Authorization": f"Bearer {RESEND_API_KEY}"}
    with httpx.Client(timeout=30) as client:
        r = client.post("https://api.resend.com/emails", json=payload, headers=headers)
        r.raise_for_status()
        return r.json()

def _send_email(subject: str, html: str, csv_bytes: bytes, csv_name="tasks.csv"):
    """Preferred: Resend; Fallback: SMTP."""
    if RESEND_API_KEY:
        return _send_via_resend(subject, html, csv_bytes, csv_name)
    return _send_via_smtp(subject, html, csv_bytes, csv_name)

def _rollover_incomplete(db: Session, plan_id: str, from_day: int) -> int:
    """Move NOT-done tasks from `from_day` -> `from_day + 1` (same area)."""
    to_day = from_day + 1
    src_rows = db.query(Cell).filter_by(plan_id=plan_id, day=from_day).all()

    dest_rows = db.query(Cell).filter_by(plan_id=plan_id, day=to_day).all()
    dest_map = {r.area: (r.activities or []) for r in dest_rows}

    moved = 0
    for row in src_rows:
        keep, carry = [], []
        for s in (row.activities or []):
            nt = _normalize_task(s)  # just to check done in any format
            # Keep the ORIGINAL string/object 's' to avoid reformatting storage
            if nt["done"]:
                keep.append(s)
            else:
                carry.append(s)
                moved += 1
        row.activities = keep
        new_list = dest_map.get(row.area, [])
        if carry:
            new_list.extend(carry)
            _upsert_cell(db, plan_id, row.area, to_day, new_list)

    db.commit()
    return moved

# =========================
# Routes
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
    return PlanUpsert(id=p.id, name=p.name, total_days=p.total_days, areas=p.areas, allow_multiple=p.allow_multiple)

@app.get("/plans/{plan_id}/grid")
def get_grid(plan_id: str, db: Session = Depends(get_db)):
    p = db.get(Plan, plan_id)
    if not p:
        raise HTTPException(404, "Plan not found")
    rows = db.query(Cell).filter(Cell.plan_id == plan_id).all()
    return {
        "allowMultiple": p.allow_multiple,
        "cells": [{"area": c.area, "day": c.day, "activities": c.activities} for c in rows]
    }

@app.put("/plans/{plan_id}/grid", dependencies=[Depends(require_key)])
def upsert_grid(plan_id: str, payload: GridIn, db: Session = Depends(get_db)):
    p = db.get(Plan, plan_id)
    if not p:
        raise HTTPException(404, "Plan not found")
    if payload.allow_multiple is not None:
        p.allow_multiple = payload.allow_multiple

    for c in payload.cells:
        row = db.query(Cell).filter_by(plan_id=plan_id, area=c.area, day=c.day).one_or_none()
        if row: row.activities = c.activities
        else:   db.add(Cell(plan_id=plan_id, area=c.area, day=c.day, activities=c.activities))

    db.commit()
    return {"ok": True}

# ---- Ops router (email / rollover) ----
ops = APIRouter(prefix="/ops", tags=["ops"])

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
            csv_bytes=b"date,ok\n2025-09-13,yes\n",
            csv_name="ping.csv",
        )
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@ops.get("/send_daily_email", dependencies=[Depends(require_key)])
def send_daily_email(plan_id: str = DEFAULT_PLAN_ID, span: int = 3):
    try:
        with SessionLocal() as db:
            start_date = _get_start_date_for_plan(db, plan_id)
            today_day = _day_for_date(start_date, _today_local())
            html, csv_bytes = _build_three_day_report(db, plan_id, today_day, span=span)
            _send_email(
                subject=f"[{plan_id}] 3-day checklist (Day {today_day}–{today_day+span-1})",
                html=html, csv_bytes=csv_bytes, csv_name=f"checklist_day{today_day}.csv",
            )
            return {"ok": True, "day": today_day}
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@ops.post("/rollover", dependencies=[Depends(require_key)])
def rollover(plan_id: str = DEFAULT_PLAN_ID):
    try:
        with SessionLocal() as db:
            start_date = _get_start_date_for_plan(db, plan_id)
            from_day = _day_for_date(start_date, _today_local())
            moved = _rollover_incomplete(db, plan_id, from_day)
            return {"ok": True, "moved": moved, "from_day": from_day, "to_day": from_day + 1}
    except Exception as e:
        raise HTTPException(500, detail=str(e))

app.include_router(ops)

# =========================
# Optional in-process scheduler (07:00 email, 19:00 rollover)
# =========================
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    HAVE_SCHEDULER = True
except Exception:
    HAVE_SCHEDULER = False

if HAVE_SCHEDULER:
    try:
        scheduler = BackgroundScheduler()
        def job_email():
            try:
                with SessionLocal() as db:
                    start_date = _get_start_date_for_plan(db, DEFAULT_PLAN_ID)
                    today_day = _day_for_date(start_date, _today_local())
                    html, csv_bytes = _build_three_day_report(db, DEFAULT_PLAN_ID, today_day, span=3)
                    _send_email(
                        subject=f"[{DEFAULT_PLAN_ID}] 3-day checklist (Day {today_day}–{today_day+2})",
                        html=html, csv_bytes=csv_bytes, csv_name=f"checklist_day{today_day}.csv")
            except Exception as e:
                print("Morning email failed:", e)

        def job_rollover():
            try:
                with SessionLocal() as db:
                    start_date = _get_start_date_for_plan(db, DEFAULT_PLAN_ID)
                    from_day = _day_for_date(start_date, _today_local())
                    moved = _rollover_incomplete(db, DEFAULT_PLAN_ID, from_day)
                    print("Rollover moved:", moved)
            except Exception as e:
                print("Evening rollover failed:", e)

        scheduler.add_job(job_email,    CronTrigger(hour=7,  minute=0))
        scheduler.add_job(job_rollover, CronTrigger(hour=19, minute=0))
        scheduler.start()
    except Exception as e:
        print("Scheduler not started:", e)

