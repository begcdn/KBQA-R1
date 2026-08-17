from kbqa_r1.sparql.odbc_config import ODBCConfig


def test_odbc_retries_transient_failures_by_default(monkeypatch):
    monkeypatch.delenv("FREEBASE_ODBC_MAX_RETRIES", raising=False)
    monkeypatch.delenv("FREEBASE_ODBC_RETRY_DELAY", raising=False)

    config = ODBCConfig.from_env()

    assert config.max_retries == 3
    assert config.retry_delay == 1.0


def test_odbc_retry_policy_can_be_overridden(monkeypatch):
    monkeypatch.setenv("FREEBASE_ODBC_MAX_RETRIES", "5")
    monkeypatch.setenv("FREEBASE_ODBC_RETRY_DELAY", "0.25")

    config = ODBCConfig.from_env()

    assert config.max_retries == 5
    assert config.retry_delay == 0.25
