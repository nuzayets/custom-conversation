"""Tests for stored default prompts after Home Assistant tool renaming."""

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.custom_conversation import async_migrate_entry
from custom_components.custom_conversation.const import (
    CONF_API_PROMPT_BASE,
    CONF_CUSTOM_PROMPTS_SECTION,
    CONF_LANGFUSE_SECTION,
    CONF_PROMPT_LIVE_CONTEXT,
    CONFIG_VERSION,
    DEFAULT_API_PROMPT_BASE,
    DEFAULT_API_PROMPT_LIVE_CONTEXT,
    DOMAIN,
)


@pytest.mark.parametrize(
    ("saved", "expected"),
    [
        (
            {
                CONF_API_PROMPT_BASE: DEFAULT_API_PROMPT_BASE.replace("intent__", ""),
                CONF_PROMPT_LIVE_CONTEXT: DEFAULT_API_PROMPT_LIVE_CONTEXT.replace(
                    "homeassistant__", ""
                ),
            },
            {
                CONF_API_PROMPT_BASE: DEFAULT_API_PROMPT_BASE,
                CONF_PROMPT_LIVE_CONTEXT: DEFAULT_API_PROMPT_LIVE_CONTEXT,
            },
        ),
        (
            {
                CONF_API_PROMPT_BASE: "My custom HassTurnOn instructions",
                CONF_PROMPT_LIVE_CONTEXT: "My custom GetLiveContext instructions",
            },
            {
                CONF_API_PROMPT_BASE: "My custom HassTurnOn instructions",
                CONF_PROMPT_LIVE_CONTEXT: "My custom GetLiveContext instructions",
            },
        ),
        ({}, {}),
    ],
)
async def test_migrate_only_saved_default_prompts(hass, saved, expected):
    """Update old defaults without changing custom prompts or Langfuse settings."""
    langfuse = {"api_prompt_id": "Jarvis", "langfuse_tracing_enabled": True}
    entry = MockConfigEntry(
        domain=DOMAIN,
        version=CONFIG_VERSION,
        minor_version=1,
        data={"primary_provider": "openai"},
        options={CONF_CUSTOM_PROMPTS_SECTION: saved, CONF_LANGFUSE_SECTION: langfuse},
    )
    entry.add_to_hass(hass)

    assert await async_migrate_entry(hass, entry)
    assert entry.minor_version == 2
    assert entry.options[CONF_CUSTOM_PROMPTS_SECTION] == expected
    assert entry.options[CONF_LANGFUSE_SECTION] == langfuse
    assert entry.data == {"primary_provider": "openai"}
    assert await async_migrate_entry(hass, entry)
    assert entry.options[CONF_CUSTOM_PROMPTS_SECTION] == expected


async def test_migrate_without_saved_prompts(hass):
    """Do not add a prompt section when none was saved."""
    entry = MockConfigEntry(domain=DOMAIN, version=CONFIG_VERSION, minor_version=1)
    entry.add_to_hass(hass)

    assert await async_migrate_entry(hass, entry)
    assert entry.minor_version == 2
    assert not entry.options
