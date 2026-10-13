from alembic import context
from app import models  # noqa: F401  (register tables)
from app.config import settings
from app.db import Base, _make_engine

target_metadata = Base.metadata


def run_migrations_online() -> None:
    engine = _make_engine(settings.database_url)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
