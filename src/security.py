"""Small, practical security helpers for ColdChain AI."""
import re


_ALLOWED_VIEW = "VW_ACTIVE_FLEET"
_BLOCKED = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|"
    r"GRANT|REVOKE|CALL|EXECUTE|LOAD_FILE|OUTFILE|DUMPFILE|"
    r"UNION|JOIN|SLEEP|BENCHMARK|INTO|INFORMATION_SCHEMA|"
    r"MYSQL\.USER|PERFORMANCE_SCHEMA)\b",
    re.IGNORECASE,
)


def validate_readonly_query(sql_query: str, max_rows: int = 20) -> str:
    """Allow a single, bounded SELECT against VW_ACTIVE_FLEET only.

    This is intentionally restrictive for an LLM-generated SQL tool.
    The MySQL account must still be granted SELECT-only permissions.
    """
    if not isinstance(sql_query, str) or not sql_query.strip():
        raise ValueError("SQL query cannot be empty.")

    query = sql_query.strip()

    if any(token in query for token in (";", "--", "/*", "*/", "#")):
        raise ValueError("Multiple statements and SQL comments are prohibited.")

    if not re.match(r"^SELECT\b", query, re.IGNORECASE):
        raise ValueError("Only SELECT statements are permitted.")

    if _BLOCKED.search(query):
        raise ValueError("Restricted SQL operation or source detected.")

    # Keep the model inside the intended read-only view; no joins/subqueries.
    if not re.search(r"\bFROM\s+`?VW_ACTIVE_FLEET`?\b", query, re.IGNORECASE):
        raise ValueError(f"Queries must read from {_ALLOWED_VIEW} only.")

    if re.search(r"\b(FROM|JOIN)\s*\(", query, re.IGNORECASE):
        raise ValueError("Subqueries are not permitted.")

    # Require one FROM clause, and no additional table sources.
    if len(re.findall(r"\bFROM\b", query, re.IGNORECASE)) != 1:
        raise ValueError("Only one source table is permitted.")

    limit_match = re.search(r"\bLIMIT\s+(\d+)\s*$", query, re.IGNORECASE)
    if limit_match:
        query = re.sub(
            r"\bLIMIT\s+\d+\s*$",
            f"LIMIT {min(int(limit_match.group(1)), max_rows)}",
            query,
            flags=re.IGNORECASE,
        )
    else:
        query += f" LIMIT {max_rows}"

    return query
