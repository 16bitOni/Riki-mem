"""Vector search with pgvector.

The ``embedding`` columns are native pgvector ``vector`` columns, searched in
the database with HNSW indexes (cosine distance).

One table can hold vectors from several embedding models at once (a model
change is re-embedded gradually, see ``reembed_stale``), and models differ in
dimension. So the column is an untyped ``vector`` and each (table, model) pair
gets its own partial HNSW index over ``embedding::vector(dim)``, built the
first time that model is seen. A search only ever compares vectors from the
current model.

Every search is scoped to one user. When a user owns a small share of the
table, Postgres picks the ``user_id`` index and compares that user's vectors
exactly (perfect recall); for a user with a large share it can use HNSW, and
the iterative index scan keeps the user filter from starving the results.
Vectors travel in binary (:class:`Embedding`).
"""

import hashlib
import logging

from sqlalchemy import Float, bindparam, cast, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql.base import ischema_names
from sqlalchemy.types import UserDefinedType

log = logging.getLogger(__name__)


class Embedding(UserDefinedType):
    """A pgvector ``vector`` column whose values travel in binary.

    pgvector's own SQLAlchemy type sends vectors as text, and Postgres then
    parses 17-digit decimals on every use: a 1536-dim query vector made an
    80-row search take 45 ms instead of 2 ms. Here lists pass straight to the
    binary asyncpg codec that :func:`register_codec` installs on each connection.
    """

    cache_ok = True

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim

    def get_col_spec(self, **kw) -> str:
        return "VECTOR" if self.dim is None else f"VECTOR({int(self.dim)})"

    def bind_processor(self, dialect):
        return None  # the asyncpg codec encodes lists

    def result_processor(self, dialect, coltype):
        def process(value):
            return None if value is None else [float(x) for x in value.to_list()]

        return process


# So reflection (migrations' model check) reads a `vector` column back as this type.
ischema_names["vector"] = Embedding


async def register_codec(conn) -> None:
    """Set up one new asyncpg connection: binary ``vector`` codec, and pgvector's
    iterative index scan (>= 0.8), so an HNSW scan keeps going until enough
    rows pass the per-user filter. Set once here instead of before every search.

    On a brand-new database the type doesn't exist until ``init()`` creates the
    extension; that connection is then discarded (see ``Database.init``).
    """
    from pgvector.asyncpg import register_vector

    try:
        await register_vector(conn)
    except ValueError as exc:
        if "unknown type" not in str(exc):
            raise
        return
    try:
        await conn.execute("SET hnsw.iterative_scan = relaxed_order")
    except Exception:
        # pgvector < 0.8 has no iterative scan; searches still work, the filter
        # just can't extend an HNSW scan (exact per-user search is unaffected).
        pass


def index_name(table_name: str, model: str, dim: int) -> str:
    """Stable, identifier-safe name for one model's HNSW index on one table."""
    digest = hashlib.sha1(model.encode("utf-8")).hexdigest()[:10]
    return f"{table_name}_hnsw_{digest}_{dim}"


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


VECTOR_TABLES = ("memories", "episodes")


class PgVectorIndex:
    """Builds the per-model HNSW indexes and runs nearest-neighbour searches."""

    def __init__(self) -> None:
        self._indexed: set[str] = set()

    async def prepare(self, conn) -> None:
        from .migrate import LOCK_ID

        # Two processes starting at once would both try to create it, and one fails:
        # serialise on the same lock the migrations use (released at commit).
        await conn.execute(text("SELECT pg_advisory_xact_lock(:id)"), {"id": LOCK_ID})
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    async def init(self, db) -> None:
        async with db.engine.begin() as conn:
            await self._migrate_text_columns(conn)
            combos = []
            for table_name in VECTOR_TABLES:
                rows = await conn.execute(text(
                    f"SELECT DISTINCT embedding_model, vector_dims(embedding) FROM {table_name} "
                    "WHERE embedding IS NOT NULL AND embedding_model IS NOT NULL"
                ))
                combos += [(table_name, model, dim) for model, dim in rows]
        for table_name, model, dim in combos:
            await self.ensure_index(db.engine, table_name, model, dim)

    @staticmethod
    async def _migrate_text_columns(conn) -> None:
        """Databases made before pgvector stored vectors as JSON text; convert them in place.

        ``'[0.1, 0.2]'`` is already pgvector's text format, so a cast is enough.
        """
        rows = await conn.execute(text(
            "SELECT table_name FROM information_schema.columns WHERE table_schema = current_schema() "
            "AND column_name = 'embedding' AND table_name IN ('memories', 'episodes') AND udt_name <> 'vector'"
        ))
        for (table_name,) in rows.all():
            log.warning("converting %s.embedding from text to pgvector", table_name)
            await conn.execute(text(
                f"ALTER TABLE {table_name} ALTER COLUMN embedding TYPE vector USING embedding::vector"
            ))

    async def ensure_index(self, engine, table_name: str, model: str, dim: int) -> None:
        """Build the HNSW index for this model if it doesn't exist (or a past build failed).

        ``CONCURRENTLY`` so a live table keeps taking writes while it builds; that
        needs its own autocommit connection, outside any transaction.
        """
        name = index_name(table_name, model, dim)
        if name in self._indexed:
            return
        try:
            async with engine.connect() as conn:
                conn = await conn.execution_options(isolation_level="AUTOCOMMIT")
                valid = await conn.scalar(text(
                    "SELECT i.indisvalid FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
                    "WHERE c.relname = :name AND c.relnamespace = current_schema()::regnamespace"
                ), {"name": name})
                if valid is False:  # an interrupted concurrent build leaves an invalid index
                    await conn.execute(text(f"DROP INDEX CONCURRENTLY IF EXISTS {name}"))
                if valid is not True:
                    await conn.execute(text(
                        f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {name} ON {table_name} "
                        f"USING hnsw ((embedding::vector({int(dim)})) vector_cosine_ops) "
                        f"WHERE embedding_model = {_quote(model)}"
                    ))
            self._indexed.add(name)
        except Exception:
            # Another process may be building it right now; searches still work without it.
            log.exception("could not build vector index %s", name)

    def _search_stmt(self, table, clauses, vector, model, limit):
        """Factored out so the generated SQL can be compile-checked without a live DB."""
        dim = len(vector)
        typed = Embedding(dim)
        distance = cast(table.embedding, typed).op("<=>", return_type=Float)(
            bindparam("query_vector", vector, type_=typed)
        )
        return (
            select(table.id, distance.label("distance"))
            .where(
                *clauses,
                table.embedding.is_not(None),
                # Inlined, not a parameter: the planner only uses a partial index
                # whose predicate it can see in the query text.
                table.embedding_model == bindparam("model", model, literal_execute=True),
            )
            .order_by(distance)
            .limit(limit)
        )

    async def nearest(
        self,
        session: AsyncSession,
        table,
        clauses: list,
        vector: list[float],
        model: str,
        limit: int,
        min_similarity: float,
        build_index: bool = True,
    ) -> list[tuple[str, float]]:
        """``(id, cosine similarity)`` of the closest rows of ``table`` matching
        ``clauses``, best first, at most ``limit``, none below ``min_similarity``.

        ``build_index=False`` inside a write transaction: a concurrent index build
        waits for every open transaction, including the caller's own."""
        if build_index:
            await self.ensure_index(session.bind, table.__tablename__, model, len(vector))
        rows = (await session.execute(self._search_stmt(table, clauses, vector, model, limit))).all()
        scored = [(rid, 1.0 - float(dist)) for rid, dist in rows]
        scored = [pair for pair in scored if pair[1] >= min_similarity]
        # relaxed_order can return rows slightly out of order
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored
