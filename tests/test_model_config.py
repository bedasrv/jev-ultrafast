"""Configuration resolution for the decision and text helpers.

These are pure environment lookups, so they are tested offline with no provider
calls. The cases that matter are the ones where a *wrong* value sends a
credential to a host it does not belong to.
"""

import pytest

from jev_ultrafast.model import PUBLIC_SYSTEMONE, systemone_url

PUBLIC = {"TYPESAFE_ENDPOINT": "", "TYPESAFE_BASE_URL": ""}


def test_unset_falls_back_to_the_public_vendor():
    """An unconfigured install must behave exactly as it did before."""
    with pytest.MonkeyPatch.context() as m:
        m.setenv("TYPESAFE_ENDPOINT", "")
        m.setenv("TYPESAFE_BASE_URL", "")
        assert systemone_url() == PUBLIC_SYSTEMONE


def test_endpoint_is_used_verbatim():
    """TYPESAFE_ENDPOINT is a full URL; nothing is appended to it."""
    url = "http://10.10.100.130:4097/v1/systemone"
    with pytest.MonkeyPatch.context() as m:
        m.delenv("TYPESAFE_BASE_URL", raising=False)
        m.setenv("TYPESAFE_ENDPOINT", url)
        assert systemone_url() == url


def test_endpoint_wins_over_base_url():
    with pytest.MonkeyPatch.context() as m:
        m.setenv("TYPESAFE_ENDPOINT", "http://a.example/v1/systemone")
        m.setenv("TYPESAFE_BASE_URL", "http://b.example")
        assert systemone_url() == "http://a.example/v1/systemone"


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        ("http://host:4097", "http://host:4097/v1/systemone"),
        ("http://host:4097/", "http://host:4097/v1/systemone"),
        ("http://host:4097/v1", "http://host:4097/v1/systemone"),
        ("http://host:4097/v1/", "http://host:4097/v1/systemone"),
    ],
)
def test_base_url_is_joined_with_and_without_v1(base, expected):
    with pytest.MonkeyPatch.context() as m:
        m.setenv("TYPESAFE_ENDPOINT", "")
        m.setenv("TYPESAFE_BASE_URL", base)
        assert systemone_url() == expected


def test_timeout_is_read_from_the_environment():
    from jev_ultrafast.model import _timeout

    with pytest.MonkeyPatch.context() as m:
        m.setenv("TYPESAFE_TIMEOUT_MS", "30000")
        assert _timeout() == 30.0
    with pytest.MonkeyPatch.context() as m:
        m.delenv("TYPESAFE_TIMEOUT_MS", raising=False)
        assert _timeout() == 25.0
    with pytest.MonkeyPatch.context() as m:
        m.setenv("TYPESAFE_TIMEOUT_MS", "not-a-number")
        assert _timeout() == 25.0
    with pytest.MonkeyPatch.context() as m:
        m.setenv("TYPESAFE_TIMEOUT_MS", "10")  # sub-second floor
        assert _timeout() == 1.0


def test_text_model_refuses_to_guess_a_vendor_when_half_configured():
    """A model without an endpoint must fail loudly, not leak the key to DeepSeek."""
    from jev_ultrafast.model import field_text

    with pytest.MonkeyPatch.context() as m:
        m.setenv("TEXT_MODEL_API_KEY", "sk-secret")
        m.setenv("TEXT_MODEL_BASE_URL", "")
        m.setenv("TEXT_MODEL", "some/other-model")
        with pytest.raises(ValueError, match="TEXT_MODEL_BASE_URL"):
            field_text({"goal": "g", "field": {}, "page": {}, "recent_actions": []})


def test_text_model_omit_sends_no_reasoning_key():
    """`omit` must leave the key out entirely, which `none` does not."""
    from jev_ultrafast import model as m

    captured = {}

    def fake_post(url, key, body):
        captured.update(body)
        return {"choices": [{"message": {"content": '{"text": "ok"}'}}]}

    with pytest.MonkeyPatch.context() as mc:
        mc.setenv("TEXT_MODEL_API_KEY", "sk-secret")
        mc.setenv("TEXT_MODEL_BASE_URL", "https://example.test/v1")
        mc.setenv("TEXT_MODEL", "some-model")
        mc.setenv("TEXT_MODEL_REASONING", "omit")
        mc.setattr(m, "post_json", fake_post)
        value, _ = m.field_text({"goal": "g", "field": {}, "page": {}, "recent_actions": []})

    assert value == "ok"
    assert "reasoning" not in captured
    assert "thinking" not in captured


def test_text_model_none_still_sends_the_key():
    """`none` disables reasoning but does send it; `omit` is the distinct mode."""
    from jev_ultrafast import model as m

    captured = {}

    def fake_post(url, key, body):
        captured.update(body)
        return {"choices": [{"message": {"content": '{"text": "ok"}'}}]}

    with pytest.MonkeyPatch.context() as mc:
        mc.setenv("TEXT_MODEL_API_KEY", "sk-secret")
        mc.setenv("TEXT_MODEL_BASE_URL", "https://example.test/v1")
        mc.setenv("TEXT_MODEL", "some-model")
        mc.setenv("TEXT_MODEL_REASONING", "none")
        mc.setattr(m, "post_json", fake_post)
        m.field_text({"goal": "g", "field": {}, "page": {}, "recent_actions": []})

    assert captured["reasoning"] == {"enabled": False}


def test_provider_error_detail_surfaces_the_message():
    from jev_ultrafast.model import provider_error_detail

    class R:
        status_code = 400

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            if isinstance(self._payload, Exception):
                raise self._payload
            return self._payload

    assert provider_error_detail(R({"error": {"message": "Unknown name \"reasoning\""}})) == (
        ': Unknown name "reasoning"'
    )
    # some shims wrap the error in a list
    assert "Extra inputs" in provider_error_detail(R([{"error": {"message": "Extra inputs"}}]))
    assert provider_error_detail(R(ValueError("not json"))) == ""
    assert provider_error_detail(R({"error": {"message": "  "}})) == ""
    assert provider_error_detail(R("plain string")) == ""
