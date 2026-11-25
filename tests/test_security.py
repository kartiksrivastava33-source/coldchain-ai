import pytest

from src.security import validate_readonly_query


def test_allows_select_from_approved_view_and_adds_limit():
    query = validate_readonly_query(
        "SELECT Record_ID, Risk_Classification FROM VW_ACTIVE_FLEET"
    )
    assert query.endswith("LIMIT 20")


def test_caps_large_limit():
    query = validate_readonly_query(
        "SELECT Record_ID FROM VW_ACTIVE_FLEET LIMIT 500"
    )
    assert query.endswith("LIMIT 20")


@pytest.mark.parametrize("query", [
    "DELETE FROM VW_ACTIVE_FLEET",
    "SELECT * FROM other_table",
    "SELECT * FROM VW_ACTIVE_FLEET; DROP TABLE shipments",
    "SELECT * FROM VW_ACTIVE_FLEET -- comment",
    "SELECT * FROM VW_ACTIVE_FLEET UNION SELECT * FROM users",
    "SELECT SLEEP(10) FROM VW_ACTIVE_FLEET",
    "SELECT * FROM (SELECT * FROM VW_ACTIVE_FLEET) x",
])
def test_blocks_unsafe_queries(query):
    with pytest.raises(ValueError):
        validate_readonly_query(query)


def test_rejects_empty_query():
    with pytest.raises(ValueError):
        validate_readonly_query("   ")
