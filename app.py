# app.py
import os
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, conint
from sqlalchemy import (create_engine, Column, Integer, String, JSON, Boolean,
                        DateTime, UniqueConstraint, ForeignKey, func)
from sqlalchemy.orm import sessionmaker, declarative_base, Session, relationship

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./schedule.db")
API_KEY = os.getenv("API_KEY", "changeme")  # set on server

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

class Plan(Base):
    __tablename__ = "plans"
    id = Column(String, primary_key=True)         # e.g., "default"
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

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
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

# ---- Schemas ----
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

# ---- Endpoints ----
@app.post("/plans", dependencies=[Depends(require_key)], response_model=PlanUpsert)
def create_or_update_plan(payload: PlanUpsert, db: Session = Depends(get_db)):
    p = db.get(Plan, payload.id)
    if p:
        p.name, p.total_days, p.areas, p.allow_multiple = (
            payload.name, payload.total_days, payload.areas, payload.allow_multiple
        )
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
    if not p: raise HTTPException(404)
    rows = db.query(Cell).filter(Cell.plan_id == plan_id).all()
    return {"allowMultiple": p.allow_multiple,
            "cells": [{"area": c.area, "day": c.day, "activities": c.activities} for c in rows]}

@app.put("/plans/{plan_id}/grid", dependencies=[Depends(require_key)])
def upsert_grid(plan_id: str, payload: GridIn, db: Session = Depends(get_db)):
    p = db.get(Plan, plan_id)
    if not p: raise HTTPException(404, "Plan not found")
    if payload.allow_multiple is not None:
        p.allow_multiple = payload.allow_multiple
    for c in payload.cells:
        row = db.query(Cell).filter_by(plan_id=plan_id, area=c.area, day=c.day).one_or_none()
        if row: row.activities = c.activities
        else:   db.add(Cell(plan_id=plan_id, area=c.area, day=c.day, activities=c.activities))
    db.commit()
    return {"ok": True}
