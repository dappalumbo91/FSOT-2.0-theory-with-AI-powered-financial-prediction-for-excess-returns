"""FSOT Market Monitor — FastAPI entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import backtest, health, history, market, news, paper, predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("fsot-market")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("FSOT Market Monitor v%s starting", __version__)
    yield
    log.info("shutdown")


app = FastAPI(
    title="FSOT Market Monitor",
    description="Financial monitoring & prediction powered by Fluid Spacetime Omni-Theory (FSOT 2.1)",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(market.router)
app.include_router(predict.router)
app.include_router(backtest.router)
app.include_router(paper.router)
app.include_router(news.router)
app.include_router(history.router)


@app.get("/")
def root():
    return {
        "app": "FSOT Market Monitor",
        "version": __version__,
        "docs": "/docs",
        "health": "/api/health",
    }
