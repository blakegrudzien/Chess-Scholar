from src.ui.resources import (
    KEEP_WARM_INACTIVITY_LIMIT_SECONDS,
    KEEP_WARM_PING_INTERVAL_SECONDS,
    keep_warm_ping_due,
)


def test_first_run_of_a_session_pings_immediately():
    assert keep_warm_ping_due(now=1000.0, last_ping=None, last_interaction=1000.0)


def test_no_ping_before_the_interval_has_passed():
    assert not keep_warm_ping_due(
        now=1000.0 + KEEP_WARM_PING_INTERVAL_SECONDS - 1,
        last_ping=1000.0,
        last_interaction=1000.0,
    )


def test_pings_once_the_interval_has_passed():
    assert keep_warm_ping_due(
        now=1000.0 + KEEP_WARM_PING_INTERVAL_SECONDS,
        last_ping=1000.0,
        last_interaction=1000.0,
    )


def test_stops_pinging_after_the_visitor_goes_inactive():
    now = 1000.0 + KEEP_WARM_INACTIVITY_LIMIT_SECONDS + 1
    assert not keep_warm_ping_due(now=now, last_ping=0.0, last_interaction=1000.0)
