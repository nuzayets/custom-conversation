"""Fixtures for Custom Conversation tests."""

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.custom_conversation.const import (
    CONF_API_PROMPT_BASE,
    CONF_CUSTOM_PROMPTS_SECTION,
    CONF_INSTRUCTIONS_PROMPT,
    CONF_LANGFUSE_SECTION,
    CONF_PRIMARY_API_KEY,
    CONF_PRIMARY_BASE_URL,
    CONF_PRIMARY_CHAT_MODEL,
    CONF_PRIMARY_PROVIDER,
    CONF_PROMPT_BASE,
    CONF_PROMPT_DEVICE_KNOWN_LOCATION,
    CONF_PROMPT_NO_ENABLED_ENTITIES,
    CONFIG_VERSION,
    DOMAIN,
)
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    """Enable custom integrations for testing."""
    yield

@pytest.fixture(autouse=True)
async def mock_default_components(hass: HomeAssistant) -> None:
    """Fixture to setup required default components."""
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "conversation", {})

@pytest.fixture
def hass(hass: HomeAssistant) -> HomeAssistant:
    """Fixture that provides a fully set up Home Assistant instance."""
    hass.data.update({DOMAIN: {}})    
    return hass

@pytest.fixture
def config_entry(hass: HomeAssistant) -> MockConfigEntry:
    """Create a mock config entry."""
    entry = MockConfigEntry(
        title="Test",
        version=CONFIG_VERSION,
        domain=DOMAIN,
        data={ # Use new data structure
            CONF_PRIMARY_PROVIDER: "openai",
            CONF_PRIMARY_API_KEY: "test-api-key",
            CONF_PRIMARY_BASE_URL: "https://api.openai.com/v1",
            CONF_PRIMARY_CHAT_MODEL: "gpt-4o-mini",
        },
        options={
            CONF_CUSTOM_PROMPTS_SECTION: {
                CONF_PROMPT_BASE: "Custom base prompt for {{ ha_name }}",
                CONF_INSTRUCTIONS_PROMPT: "Custom instructions for {{ user_name }}",
                CONF_API_PROMPT_BASE: "Custom API base prompt",
                CONF_PROMPT_DEVICE_KNOWN_LOCATION: "Custom location prompt for {{ location }}",
                CONF_PROMPT_NO_ENABLED_ENTITIES: "Custom no entities prompt",
            },
            CONF_LANGFUSE_SECTION: {}
        }
    )
    entry.add_to_hass(hass)
    hass.config_entries.async_setup(entry.entry_id)
    hass.async_block_till_done()
    return entry

