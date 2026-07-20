"""Tests for integration lifecycle handling."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from custom_components.custom_conversation import async_setup_entry, async_unload_entry
from custom_components.custom_conversation.const import (
    CONF_ENABLE_LANGFUSE,
    CONF_LANGFUSE_HOST,
    CONF_LANGFUSE_PUBLIC_KEY,
    CONF_LANGFUSE_SECRET_KEY,
    CONF_LANGFUSE_SECTION,
    CONF_LANGFUSE_TRACING_ENABLED,
    DOMAIN,
)
from custom_components.custom_conversation.prompt_manager import (
    LangfuseClient,
    LangfuseInitError,
)
from homeassistant.exceptions import ConfigEntryError


async def test_setup_failure_releases_langfuse_reference_for_retry(hass):
    """Test platform setup rollback leaves one owner after retry."""
    entry = Mock(
        entry_id="entry-id",
        options={
            CONF_LANGFUSE_SECTION: {
                CONF_ENABLE_LANGFUSE: False,
                CONF_LANGFUSE_HOST: "https://langfuse.example.com",
                CONF_LANGFUSE_PUBLIC_KEY: "public-key",
                CONF_LANGFUSE_SECRET_KEY: "secret-key",
                CONF_LANGFUSE_TRACING_ENABLED: True,
            }
        },
    )
    sdk_client = MagicMock()
    forward = AsyncMock(side_effect=[RuntimeError("platform failed"), None])

    with (
        patch(
            "custom_components.custom_conversation.prompt_manager.Langfuse",
            return_value=sdk_client,
        ) as constructor,
        patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            forward,
        ),
    ):
        with pytest.raises(RuntimeError, match="platform failed"):
            await async_setup_entry(hass, entry)
        assert entry.entry_id not in hass.data[DOMAIN]

        assert await async_setup_entry(hass, entry) is True

    constructor.assert_called_once()
    wrapper = hass.data[DOMAIN][entry.entry_id]["langfuse_client"]
    assert isinstance(wrapper, LangfuseClient)
    await wrapper.cleanup()
    await LangfuseClient.shutdown_all(hass)
    sdk_client.shutdown.assert_called_once_with()


async def test_credential_change_requires_visible_restart(hass):
    """Test a v4 singleton conflict is surfaced instead of silently disabled."""
    first_entry = Mock(
        entry_id="entry-id",
        options={
            CONF_LANGFUSE_SECTION: {
                CONF_ENABLE_LANGFUSE: False,
                CONF_LANGFUSE_HOST: "https://langfuse.example.com",
                CONF_LANGFUSE_PUBLIC_KEY: "public-key",
                CONF_LANGFUSE_SECRET_KEY: "first-secret",
                CONF_LANGFUSE_TRACING_ENABLED: True,
            }
        },
    )
    changed_entry = Mock(
        entry_id="entry-id",
        options={
            CONF_LANGFUSE_SECTION: {
                CONF_ENABLE_LANGFUSE: False,
                CONF_LANGFUSE_HOST: "https://langfuse.example.com",
                CONF_LANGFUSE_PUBLIC_KEY: "public-key",
                CONF_LANGFUSE_SECRET_KEY: "changed-secret",
                CONF_LANGFUSE_TRACING_ENABLED: True,
            }
        },
    )
    sdk_client = MagicMock()

    with (
        patch(
            "custom_components.custom_conversation.prompt_manager.Langfuse",
            return_value=sdk_client,
        ),
        patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            AsyncMock(),
        ),
        patch.object(
            hass.config_entries,
            "async_unload_platforms",
            AsyncMock(return_value=True),
        ),
    ):
        assert await async_setup_entry(hass, first_entry) is True
        assert await async_unload_entry(hass, first_entry) is True

        with pytest.raises(ConfigEntryError, match="restart Home Assistant"):
            await async_setup_entry(hass, changed_entry)

    await LangfuseClient.shutdown_all(hass)


async def test_langfuse_failure_does_not_disable_conversation(hass):
    """Test ordinary telemetry failure preserves conversation setup."""
    entry = Mock(entry_id="entry-id", options={})
    forward = AsyncMock()

    with (
        patch.object(
            LangfuseClient,
            "create",
            AsyncMock(side_effect=LangfuseInitError("temporarily unavailable")),
        ),
        patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            forward,
        ),
    ):
        assert await async_setup_entry(hass, entry) is True

    forward.assert_awaited_once()
    assert hass.data[DOMAIN][entry.entry_id]["langfuse_client"] is None
