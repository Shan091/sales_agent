"""Startup secret validation (fail-fast in production)."""
import pytest
from config.settings import Settings


def test_production_rejects_placeholder_secrets():
    s = Settings(APP_ENV="production", META_APP_SECRET="your_app_secret", OPENAI_API_KEY="")
    with pytest.raises(RuntimeError):
        s.assert_production_secrets()


def test_production_passes_with_real_secrets():
    s = Settings(
        APP_ENV="production",
        META_APP_SECRET="real", WHATSAPP_API_TOKEN="real",
        WHATSAPP_PHONE_NUMBER_ID="real", WHATSAPP_VERIFY_TOKEN="real",
        OPENAI_API_KEY="real",
    )
    s.assert_production_secrets()  # must not raise


def test_development_ignores_placeholders():
    Settings(APP_ENV="development", META_APP_SECRET="your_app_secret").assert_production_secrets()
