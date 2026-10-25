"""Model replies as real providers send them. Each case here once made extraction
save nothing without an error: DeepSeek and Llama wrap the object under the
schema name, and gpt-oss leaves out importance and semantic."""

import json

import pytest
from pydantic import ValidationError

from riki_memory.llm import parse_json_reply
from riki_memory.schemas import Extraction

REPLY = {"memories": [{"content": "Got the car serviced on 2023-03-15.", "kind": "event"}]}


@pytest.mark.parametrize("content", [
    json.dumps(REPLY),
    json.dumps({"answer": json.dumps(REPLY)}),  # DeepSeek: the object as a string under the schema name
    json.dumps({"answer": REPLY}),  # Llama 4: the object itself under the schema name
    "```json\n" + json.dumps(REPLY) + "\n```",
])
def test_wrapped_replies_are_unwrapped(content):
    result = Extraction.model_validate(parse_json_reply(content, Extraction))
    assert [m.content for m in result.memories] == ["Got the car serviced on 2023-03-15."]


def test_missing_judgement_fields_get_neutral_defaults():
    memory = Extraction.model_validate(REPLY).memories[0]
    assert (memory.importance, memory.semantic) == (3, False)


def test_a_reply_without_memories_fails_loudly():
    with pytest.raises(ValidationError):
        Extraction.model_validate(parse_json_reply('{"facts": [], "notes": "none"}', Extraction))


def test_an_empty_list_is_still_a_valid_answer():
    assert Extraction.model_validate(parse_json_reply('{"memories": []}', Extraction)).memories == []


def test_a_real_field_is_never_mistaken_for_a_wrapper():
    # one key that IS a schema field: kept as is
    assert parse_json_reply(json.dumps(REPLY), Extraction) == REPLY
