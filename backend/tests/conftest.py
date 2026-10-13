import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base, get_db
from app.deps import get_judge_factory, get_session_factory
from app.main import app
from tests.fakes import FakeJudge


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def judge():
    return FakeJudge()


@pytest.fixture
def client(session_factory, judge):
    def override_db():
        with session_factory() as db:
            yield db

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_session_factory] = lambda: session_factory
    app.dependency_overrides[get_judge_factory] = lambda: lambda model: judge
    yield TestClient(app)
    app.dependency_overrides.clear()
