from datetime import timedelta

from src.operators.distributed import _DEFAULT_DDP_TIMEOUT_MIN, _resolve_timeout


class TestResolveTimeout:
    def test_default_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("DDP_TIMEOUT_MIN", raising=False)
        assert _resolve_timeout() == timedelta(minutes=_DEFAULT_DDP_TIMEOUT_MIN)

    def test_default_above_nccl_library_default(self):
        # NCCL's library default is 10 minutes; ours must exceed it so rank-0-only
        # validation does not trip the collective watchdog.
        assert _DEFAULT_DDP_TIMEOUT_MIN > 10.0

    def test_env_override_is_honored(self, monkeypatch):
        monkeypatch.setenv("DDP_TIMEOUT_MIN", "45")
        assert _resolve_timeout() == timedelta(minutes=45)

    def test_invalid_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("DDP_TIMEOUT_MIN", "not-a-number")
        assert _resolve_timeout() == timedelta(minutes=_DEFAULT_DDP_TIMEOUT_MIN)
