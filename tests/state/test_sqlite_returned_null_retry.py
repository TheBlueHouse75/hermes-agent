"""Retry the transient CPython/SQLite ``returned NULL`` SystemError."""

import pytest

from hermes_state import SessionDB


RETURNED_NULL_ERROR = (
    "<hermes_cli.sqlite_safe_read.TrackedConnection object> "
    "returned NULL without setting an exception"
)


@pytest.fixture
def db(tmp_path):
    database = SessionDB(db_path=tmp_path / "state.db")
    yield database
    database.close()


def test_returned_null_system_error_rolls_back_and_retries(db, monkeypatch):
    attempts = 0
    retries = 0

    def allow_retry(_deadline, _patience_s):
        nonlocal retries
        retries += 1
        return True

    monkeypatch.setattr(db, "_sleep_before_write_retry", allow_retry)

    def flaky(conn):
        nonlocal attempts
        attempts += 1
        conn.execute(
            "INSERT INTO state_meta (key, value) VALUES (?, ?)",
            ("returned-null", "ok"),
        )
        if attempts == 1:
            raise SystemError(RETURNED_NULL_ERROR)
        return "done"

    assert db._execute_write(flaky) == "done"
    assert attempts == 2
    assert retries == 1
    assert db.get_meta("returned-null") == "ok"


def test_unrelated_system_error_propagates(db, monkeypatch):
    attempts = 0

    def fail_if_retried(_deadline, _patience_s):
        pytest.fail("unrelated SystemError must not enter the retry path")

    monkeypatch.setattr(db, "_sleep_before_write_retry", fail_if_retried)

    def broken(_conn):
        nonlocal attempts
        attempts += 1
        raise SystemError("unrelated interpreter failure")

    with pytest.raises(SystemError, match="unrelated interpreter failure"):
        db._execute_write(broken)
    assert attempts == 1


def test_returned_null_patience_exhaustion_propagates_original_error(db, monkeypatch):
    original_error = SystemError(RETURNED_NULL_ERROR)
    attempts = 0

    monkeypatch.setattr(
        db, "_sleep_before_write_retry", lambda _deadline, _patience_s: False
    )

    def always_fails(_conn):
        nonlocal attempts
        attempts += 1
        raise original_error

    with pytest.raises(SystemError) as exc_info:
        db._execute_write(always_fails)
    assert exc_info.value is original_error
    assert attempts == 1
