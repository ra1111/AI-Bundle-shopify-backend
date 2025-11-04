from logging.config import fileConfig
import os

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy import text

from dotenv import load_dotenv

import re

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

load_dotenv()

database_url = os.getenv("DATABASE_URL")
if database_url:
    sync_database_url = database_url.replace("postgresql+asyncpg://", "postgresql://")
    config.set_main_option("sqlalchemy.url", sync_database_url)

try:
    from sqlalchemy.dialects.postgresql.base import PGDialect

    _original_get_server_version_info = PGDialect._get_server_version_info

    def _cockroach_safe_server_version(self, connection):
        version_str = connection.scalar(text("SELECT version()"))
        if isinstance(version_str, bytes):
            version_str = version_str.decode("utf-8", errors="ignore")

        if version_str and "CockroachDB" in version_str:
            match = re.search(r"v(\d+)\.(\d+)\.(\d+)", version_str)
            if match:
                return (13, 0)

        return _original_get_server_version_info(self, connection)

    PGDialect._get_server_version_info = _cockroach_safe_server_version
except Exception:
    # Best-effort patch; fall back to SQLAlchemy defaults if anything fails
    pass

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = None

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
