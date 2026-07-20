"""The Custom Conversation integration."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, EVENT_HOMEASSISTANT_STOP, Platform
from homeassistant.core import Event, HomeAssistant
from homeassistant.exceptions import ConfigEntryError
from homeassistant.helpers import config_validation as cv, llm
from homeassistant.helpers.typing import ConfigType

from .api import CustomLLMAPI
from .const import (
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_LLM_PARAMETERS_SECTION,
    CONF_MAX_TOKENS,
    CONF_PRIMARY_API_KEY,
    CONF_PRIMARY_BASE_URL,
    CONF_PRIMARY_CHAT_MODEL,
    CONF_PRIMARY_PROVIDER,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONFIG_VERSION,
    DEFAULT_PROVIDER,
    DOMAIN,
    LLM_API_ID,
    LOGGER,
)
from .prompt_manager import LangfuseClient, LangfuseError, LangfuseResourceConflictError
from .service import async_setup_services

PLATFORMS = (Platform.CONVERSATION,)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

type CustomConversationConfigEntry = ConfigEntry


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up Custom Conversation."""

    # Make sure the API is registered
    if not any(x.id == LLM_API_ID for x in llm.async_get_apis(hass)):
        llm.async_register_api(hass, CustomLLMAPI(hass))

    await async_setup_services(hass)

    async def shutdown_langfuse_resources(_event: Event) -> None:
        await LangfuseClient.shutdown_all(hass)

    hass.bus.async_listen_once(
        EVENT_HOMEASSISTANT_STOP,
        shutdown_langfuse_resources,
    )

    return True


async def async_setup_entry(
    hass: HomeAssistant, entry: CustomConversationConfigEntry
) -> bool:
    """Set up a  Custom Conversation from a config entry."""

    try:
        langfuse_client = await LangfuseClient.create(hass, entry)
    except LangfuseResourceConflictError as err:
        raise ConfigEntryError(
            "Langfuse credentials changed; restart Home Assistant to reload the Langfuse SDK resource"
        ) from err
    except LangfuseError as err:
        LOGGER.error("Unable to initialize Langfuse: %s", err)
        langfuse_client = None
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = {
        "langfuse_client": langfuse_client,
    }
    try:
        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    except BaseException:
        hass.data.get(DOMAIN, {}).pop(entry.entry_id, None)
        if langfuse_client is not None:
            try:
                await langfuse_client.cleanup()
            except Exception:
                LOGGER.warning(
                    "Error rolling back Langfuse client after setup failure",
                    exc_info=True,
                )
        raise

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Clean up clients."""
    if not await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        return False

    entry_data = hass.data.get(DOMAIN, {}).pop(entry.entry_id, {})
    if langfuse_client := entry_data.get("langfuse_client"):
        try:
            await langfuse_client.cleanup()
        except Exception as err:
            LOGGER.warning("Error cleaning up Langfuse client: %s", err)
    return True


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate old entry."""
    LOGGER.debug(
        "Migrating configuration from version %s.%s",
        config_entry.version,
        config_entry.minor_version,
    )

    if config_entry.version > CONFIG_VERSION:
        # This means the user has downgraded from a future version
        LOGGER.error(
            "Cannot migrate configuration from future version %s.%s",
            config_entry.version,
            config_entry.minor_version,
        )
        return False

    if config_entry.version < CONFIG_VERSION:
        new_data = {**config_entry.data}
        new_options = {**config_entry.options}

        new_data[CONF_PRIMARY_PROVIDER] = DEFAULT_PROVIDER
        new_data[CONF_PRIMARY_API_KEY] = new_data.pop(CONF_API_KEY)

        new_data[CONF_PRIMARY_BASE_URL] = new_data.pop(CONF_BASE_URL)

        # Migrate model from options to data
        llm_params = new_options.get(CONF_LLM_PARAMETERS_SECTION, {})
        new_data[CONF_PRIMARY_CHAT_MODEL] = llm_params.get(CONF_CHAT_MODEL, None)

        # Other LLM parameters have moved up to top level options
        new_options[CONF_TEMPERATURE] = llm_params.get(CONF_TEMPERATURE)
        new_options[CONF_TOP_P] = llm_params.get(CONF_TOP_P)
        new_options[CONF_MAX_TOKENS] = llm_params.get(CONF_MAX_TOKENS)
        hass.config_entries.async_update_entry(
            config_entry,
            data=new_data,
            options=new_options,
            version=CONFIG_VERSION,
        )
        LOGGER.info("Successfully migrated configuration to version %s", CONFIG_VERSION)

    return True
