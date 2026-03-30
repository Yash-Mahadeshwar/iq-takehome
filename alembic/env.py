"""
Alembic environment configuration for AgentAssist.

Supports:
  - Offline mode  (--sql / generate SQL scripts without a live DB connection)
  - Online mode   (async SQLAlchemy engine for direct database migration)

The database URL is read exclusively from ``app.config.get_settings()`` so
that it stays in sync with the application and is never hard-coded here.

All ORM models are imported via ``from app.models import *`` to ensure that
``Base.metadata`` is fully populated before autogenerate / diff runs.
"""
from __future__ import annotations

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# ── Make all ORM models visible to Alembic autogenerate ──────────────────────
# This populates Base.metadata with all table definitions.
from app.models import *  # noqa: F401, F403, E402
from app.database import Base  # noqa: E402
from app.config import get_settings  # noqa: E402

# ── Alembic Config object (wraps alembic.ini) ─────────────────────────────────
config = context.config

# ── Logging setup from alembic.ini [loggers] section ─────────────────────────
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ── Target metadata for --autogenerate support ────────────────────────────────
target_metadata = Base.metadata

# ── Override sqlalchemy.url from application settings ────────────────────────
# This ensures migrations always use the same URL as the running application.
_settings = get_settings()
config.set_main_option("sqlalchemy.url", _settings.db_url)


# ─────────────────────────────────────────────────────────────────────────────
# OFFLINE MODE
# Generates SQL DDL scripts without requiring a live database connection.
# Usage: alembic upgrade head --sql
# ─────────────────────────────────────────────────────────────────────────────
def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine.
    Calls to ``context.execute()`` emit the given string to the script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# ─────────────────────────────────────────────────────────────────────────────
# ONLINE MODE (async)
# Connects directly to the database and runs migrations.
# ─────────────────────────────────────────────────────────────────────────────
def do_run_migrations(connection: Connection) -> None:
    """Execute migrations against an active (sync) connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations inside a sync runner."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # avoid holding connections across migration runs
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode using an async engine."""
    asyncio.run(run_async_migrations())


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
