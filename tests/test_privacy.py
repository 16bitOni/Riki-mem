"""Secrets are scrubbed before an exchange is stored in the turn log or sent to a model."""

import pytest

from riki_memory import MemoryConfig, RikiMemory
from riki_memory.models import Turn
from riki_memory.privacy import redact_secrets


@pytest.mark.parametrize("text, secret", [
    ("My wifi keeps dropping. The password is Tiger@4521 if that matters.", "Tiger@4521"),
    ("PIN: 4821", "4821"),
    ("my password: sunshine", "sunshine"),
    ("the otp 552901 expired", "552901"),
    ("card 4111 1111 1111 1111 exp 09/27", "4111 1111 1111 1111"),
    ("use sk-abcdefghijklmnopqrstuv for the API", "sk-abcdefghijklmnopqrstuv"),
    ("aadhaar 1234 5678 9012", "1234 5678 9012"),
])
def test_secrets_are_removed(text, secret):
    assert secret not in redact_secrets(text) and "[redacted]" in redact_secrets(text)


@pytest.mark.parametrize("text", [
    "my pin code is 700091",            # an Indian postal code
    "order 1234567890123 arrived",      # fails the card checksum
    "I use a password manager",
    "The password was changed yesterday",
    "I passed the exam with 92%",
])
def test_ordinary_text_is_left_alone(text):
    assert redact_secrets(text) == text


class Recorder:
    def __init__(self, provider):
        self._provider = provider
        self.embedding_model_id = provider.embedding_model_id
        self.prompts = []

    async def embed(self, text):
        return await self._provider.embed(text)

    async def chat_json(self, system, user, schema, *, fast=True):
        self.prompts.append(user)
        return schema.model_validate({"memories": []})

    async def chat_text(self, system, user, *, fast=True):
        return ""


async def test_the_turn_log_and_the_model_never_see_the_secret(db_url, provider):
    llm = Recorder(provider)
    async with RikiMemory(database_url=db_url, llm=llm, config=MemoryConfig(planner_mode="heuristic")) as mem:
        await mem.remember("u1", "The wifi password is Tiger@4521, it keeps dropping.", "Try restarting it.")
        async with mem.db.session_scope() as session:
            from sqlalchemy import select

            [turn] = (await session.scalars(select(Turn))).all()
    assert "Tiger@4521" not in turn.user_text and "Tiger@4521" not in llm.prompts[0]
