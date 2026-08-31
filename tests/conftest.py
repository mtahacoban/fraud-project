from __future__ import annotations

import os
import tempfile

_TEST_DB_FD, TEST_DB_PATH = tempfile.mkstemp(suffix=".db", prefix="fraud_test_")
os.close(_TEST_DB_FD)
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH}"
os.environ["LLM_PROVIDER"] = ""
os.environ["LLM_API_KEY"] = ""
os.environ["LLM_MODEL"] = ""

import atexit

import pytest
from fastapi.testclient import TestClient

from backend.database import Base, SessionLocal, engine
from backend.main import app


@atexit.register
def _cleanup_test_db():
    engine.dispose()
    try:
        os.remove(TEST_DB_PATH)
    except OSError:
        pass


@pytest.fixture(scope="session", autouse=True)
def _create_test_schema():
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(autouse=True)
def block_real_groq_calls(monkeypatch):
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("A test attempted to call the real Groq API - this must never happen.")
    monkeypatch.setattr("backend.llm_service._generate_groq", _fail_if_called)
