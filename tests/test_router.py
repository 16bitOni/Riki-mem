"""The router: one zero-token decision per message, checked against hand-labelled questions."""

import pytest

from riki_memory.router import route_query

LABELLED = [
    # chitchat
    ("hi", "chitchat"),
    ("thanks!", "chitchat"),
    ("good morning", "chitchat"),
    ("ok cool", "chitchat"),
    ("bye", "chitchat"),
    # temporal arithmetic
    ("How many days before my sister's wedding did I buy the dress?", "temporal"),
    ("How many weeks after starting the new job did I get promoted?", "temporal"),
    ("How many days between the dentist appointment and the eye exam?", "temporal"),
    ("How many months since I moved to Pune did I adopt the cat?", "temporal"),
    ("How many days ago did I finish the marathon?", "temporal"),
    # aggregate
    ("How many conferences did I speak at this year?", "aggregate"),
    ("How much did I spend on the car last month?", "aggregate"),
    ("What's the total I donated to charity?", "aggregate"),
    ("How many plants do I currently have?", "aggregate"),
    ("How many hours did I spend volunteering?", "aggregate"),
    ("Count the books I read in 2025.", "aggregate"),
    ("How many cousins' weddings have I been to?", "aggregate"),
    ("How many languages am I learning right now?", "aggregate"),
    # exploratory
    ("What do you know about me?", "exploratory"),
    ("Tell me everything you remember about my work.", "exploratory"),
    ("What have we talked about recently?", "exploratory"),
    ("Tell me about myself", "exploratory"),
    # lookup
    ("What's my sister's name?", "lookup"),
    ("Where did I go on holiday last summer?", "lookup"),
    ("Which company does my brother work at?", "lookup"),
    ("What time do I usually wake up?", "lookup"),
    ("What did the doctor say about my knee?", "lookup"),
    ("Do I prefer tea or coffee?", "lookup"),
    ("hi, what's the name of my dentist?", "lookup"),
    ("When is my next flight?", "lookup"),
]


@pytest.mark.parametrize("question,route", LABELLED)
def test_route(question, route):
    assert route_query(question) == route


def test_pregate_off_never_returns_chitchat():
    assert route_query("hi", pregate=False) == "lookup"
