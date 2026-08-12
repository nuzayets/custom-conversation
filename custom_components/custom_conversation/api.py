"""Custom Version of LLM API for Custom Conversation."""

from __future__ import annotations

from typing import Any

from langfuse.model import Prompt
import slugify as unicode_slug

from homeassistant.components import llm as llm_component
from homeassistant.components.homeassistant.llm import (
    async_get_exposed_entities as _get_exposed_entities,
)
from homeassistant.components.intent import async_device_supports_timers
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    floor_registry as fr,
    llm,
)

from .const import (
    CONF_IGNORED_INTENTS,
    CONF_IGNORED_INTENTS_SECTION,
    DEFAULT_IGNORED_INTENTS,
    LLM_API_ID,
)
from .prompt_manager import PromptContext, PromptManager


class CustomLLMAPI(llm.API):
    """An API for the Custom Conversation integration to use to call Home Assistant services."""

    def __init__(
        self,
        hass: HomeAssistant,
        user_name: str | None = None,
        conversation_config_entry: ConfigEntry | None = None,
    ) -> None:
        """Initialize the API."""
        super().__init__(hass=hass, id=LLM_API_ID, name="Custom Conversation LLM API")
        self._hass = hass
        self._request_user_name = user_name
        self._prompt_manager = PromptManager(hass)
        self.prompt_object = None
        self.conversation_config_entry = conversation_config_entry

    def set_langfuse_client(self, langfuse_client: Any) -> None:
        """Set the Langfuse client."""
        self._prompt_manager.set_langfuse_client(langfuse_client)

    async def async_get_api_instance(
        self, llm_context: llm.LLMContext
    ) -> llm.APIInstance:
        """Return an instance of the Custom Conversation LLM API."""
        if llm_context.assistant:
            exposed_entities: dict | None = _get_exposed_entities(
                self.hass, llm_context.assistant, include_state=False
            )
        else:
            exposed_entities = None

        api_prompt = await self._async_get_api_prompt(llm_context, exposed_entities)
        llm_tools = await self._async_get_tools(llm_context)
        if isinstance(api_prompt, tuple):
            self.prompt_object, api_prompt = api_prompt
        else:
            self.prompt_object = None

        return llm.APIInstance(
            api=self,
            api_prompt=api_prompt,
            llm_context=llm_context,
            tools=llm_tools.tools,
            custom_serializer=llm.selector_serializer,
        )

    async def _async_get_api_prompt(
        self, llm_context: llm.LLMContext, exposed_entities: dict | None
    ) -> tuple[Prompt, str] | str:
        """Return the prompt for the API."""

        area_name = None
        floor_name = None
        supports_timers = False

        area: ar.AreaEntry | None = None
        floor: fr.FloorEntry | None = None
        if llm_context.device_id:
            device_reg = dr.async_get(self.hass)
            device = device_reg.async_get(llm_context.device_id)

            if device:
                area_reg = ar.async_get(self.hass)
                if device.area_id and (area := area_reg.async_get_area(device.area_id)):
                    area_name = area.name
                    floor_reg = fr.async_get(self.hass)
                    if area.floor_id and (
                        floor := floor_reg.async_get_floor(area.floor_id)
                    ):
                        floor_name = floor.name

            supports_timers = async_device_supports_timers(
                self.hass, llm_context.device_id
            )

        location = f"{area_name} (floor: {floor_name})" if floor_name else area_name
        context = PromptContext(
            hass=self.hass,
            ha_name=self.hass.config.location_name,
            user_name=self._request_user_name,
            llm_context=llm_context,
            location=location,
            exposed_entities=exposed_entities,
            supports_timers=supports_timers,
        )

        return await self._prompt_manager.get_api_prompt(
            context, self.conversation_config_entry
        )

    async def _async_get_tools(
        self, llm_context: llm.LLMContext
    ) -> llm_component.LLMTools:
        """Return Assist tools allowed by the custom intent configuration."""
        config_entry = self.conversation_config_entry
        if config_entry:
            ignored_intents_section = config_entry.options.get(
                CONF_IGNORED_INTENTS_SECTION, {}
            )
            ignore_intents = set(
                ignored_intents_section.get(
                    CONF_IGNORED_INTENTS, DEFAULT_IGNORED_INTENTS
                )
            )
        else:
            ignore_intents = DEFAULT_IGNORED_INTENTS

        ignored_tool_names = {
            unicode_slug.slugify(name, separator="_", lowercase=False)
            for name in ignore_intents
        }
        assist_tools = await llm_component.async_get_tools(
            self.hass, llm_context, llm.LLM_API_ASSIST
        )
        tools: list[llm.Tool] = []
        ignore_date = "HassGetCurrentDate" in ignore_intents
        ignore_time = "HassGetCurrentTime" in ignore_intents
        for tool in assist_tools.tools:
            if isinstance(tool, llm.IntentTool):
                if tool.name in ignored_tool_names:
                    continue
                tool = CardPreservingIntentTool(tool)
            elif tool.name == "GetDateTime":
                if ignore_date and ignore_time:
                    continue
                if ignore_date or ignore_time:
                    tool = FilteredDateTimeTool(tool, ignore_date, ignore_time)
            tools.append(tool)
        return llm_component.LLMTools(tools=tools)


class CardPreservingIntentTool(llm.Tool):
    """Delegate to a Home Assistant intent tool without dropping response cards."""

    def __init__(self, tool: llm.IntentTool) -> None:
        """Initialize the wrapper."""
        self._tool = tool
        self.name = tool.name
        self.description = tool.description
        self.parameters = tool.parameters

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict:
        """Call the intent and restore its card for downstream consumers."""
        result = await self._tool.async_call(hass, tool_input, llm_context)
        if isinstance(result, llm.IntentResponseDict):
            card = result.original.as_dict().get("card")
            if card:
                result["card"] = card
        return result


class FilteredDateTimeTool(llm.Tool):
    """Expose the allowed half of Home Assistant's combined date/time tool."""

    def __init__(
        self, tool: llm.Tool, ignore_date: bool, ignore_time: bool
    ) -> None:
        """Initialize the wrapper."""
        self._tool = tool
        self._ignore_date = ignore_date
        self._ignore_time = ignore_time
        self.name = tool.name
        self.description = (
            "Provides the current time."
            if ignore_date
            else "Provides the current date."
        )
        self.parameters = tool.parameters

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict:
        """Call the combined tool and remove ignored fields."""
        result = await self._tool.async_call(hass, tool_input, llm_context)
        values = result.get("result")
        if isinstance(values, dict):
            if self._ignore_date:
                values.pop("date", None)
                values.pop("weekday", None)
            if self._ignore_time:
                values.pop("time", None)
                values.pop("timezone", None)
        return result
