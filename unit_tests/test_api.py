"""Unit tests for the Custom Conversation API module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
import voluptuous as vol

from custom_components.custom_conversation.api import (
    CardPreservingIntentTool,
    CustomLLMAPI,
    FilteredDateTimeTool,
)
from custom_components.custom_conversation.const import (
    CONF_IGNORED_INTENTS,
    CONF_IGNORED_INTENTS_SECTION,
    LLM_API_ID,
)
from custom_components.custom_conversation.prompt_manager import (
    PromptContext,
    PromptManager,
)
from homeassistant.components import llm as llm_component
from homeassistant.components.script.llm import ScriptTool
from homeassistant.core import Context
from homeassistant.helpers import device_registry as dr, intent, llm
from homeassistant.setup import async_setup_component


@pytest.fixture
def mock_prompt_manager() -> MagicMock:
    """Fixture for a mocked PromptManager."""
    return MagicMock(spec=PromptManager)

@pytest.fixture
def custom_llm_api(hass, config_entry, mock_prompt_manager) -> CustomLLMAPI:
    """Fixture for a CustomLLMAPI instance with mocked PromptManager."""
    with patch("custom_components.custom_conversation.api.PromptManager", return_value=mock_prompt_manager):
        return CustomLLMAPI(
            hass=hass,
            user_name="Test User",
            conversation_config_entry=config_entry,
        )

@pytest.fixture
def mock_exposed_entities_data() -> dict:
    """Fixture for a default mock_exposed_entities dictionary."""
    return {"light.test": {"names": "Test Light", "domain": "light", "state": "on"}}

@pytest.fixture
def mock_floor(floor_registry):
    """Fixture for a mocked floor."""
    return floor_registry.async_create("Test Floor")

@pytest.fixture
def mock_area(area_registry, mock_floor):
    """Fixture for a mocked area."""
    return area_registry.async_create("Test Area", floor_id=mock_floor.floor_id)

@pytest.fixture
def mock_assist_device(device_registry, mock_area, hass):
    """Fixture for a mocked assist device."""
    config_entry = MockConfigEntry(domain="assist.satellite", data={})
    config_entry.add_to_hass(hass)
    device_registry = dr.async_get(hass)
    device = device_registry.async_get_or_create(config_entry_id=config_entry.entry_id, identifiers={("assist", "test_device")}, name="Test Device")
    device_registry.async_update_device(device_id=device.id, area_id=mock_area.id)
    return device

@pytest.fixture
def mock_llm_context(mock_assist_device) -> MagicMock:
    """Fixture for a mocked LLMContext."""
    context = MagicMock(spec=llm.LLMContext)
    context.assistant = "conversation.home_assistant"
    context.language = "en"
    context.device_id = mock_assist_device.id
    context.platform = "test_platform"
    context.context = Context()
    return context

@pytest.mark.asyncio
async def test_custom_llm_api_init(hass, config_entry):
    """Test CustomLLMAPI initialization."""
    api = CustomLLMAPI(
        hass=hass,
        user_name="Test User",
        conversation_config_entry=config_entry,
    )
    assert api._hass is hass
    assert api._request_user_name == "Test User"
    assert api.conversation_config_entry is config_entry
    assert api.id == LLM_API_ID
    assert api.name == "Custom Conversation LLM API"

def test_custom_llm_api_set_langfuse_client(custom_llm_api, mock_prompt_manager):
    """Test setting the Langfuse client."""
    mock_client = MagicMock()
    custom_llm_api.set_langfuse_client(mock_client)
    mock_prompt_manager.set_langfuse_client.assert_called_once_with(mock_client)

@pytest.mark.asyncio
async def test_custom_llm_api_get_api_instance(custom_llm_api, mock_llm_context, mock_exposed_entities_data, mock_assist_device):
    """Test getting an API instance."""
    mock_api_prompt = "Test API Prompt"
    mock_tools = [MagicMock(spec=llm.Tool)]
    platform_tools = llm_component.LLMTools(
        tools=mock_tools, prompt="Platform prompt"
    )

    with patch("custom_components.custom_conversation.api._get_exposed_entities", return_value=mock_exposed_entities_data) as mock_get_exposed, \
         patch.object(custom_llm_api, "_async_get_api_prompt", return_value=mock_api_prompt) as mock_get_prompt, \
         patch.object(custom_llm_api, "_async_get_tools", new_callable=AsyncMock, return_value=platform_tools) as mock_get_tools, \
         patch("homeassistant.helpers.llm.APIInstance") as mock_api_instance_cls, \
         patch("homeassistant.helpers.llm.selector_serializer") as mock_serializer:

        instance = await custom_llm_api.async_get_api_instance(mock_llm_context)

        mock_get_exposed.assert_called_once_with(
            custom_llm_api.hass, mock_llm_context.assistant, include_state=False
        )
        mock_get_prompt.assert_called_once_with(mock_llm_context, mock_exposed_entities_data)
        mock_get_tools.assert_awaited_once_with(mock_llm_context)
        mock_api_instance_cls.assert_called_once_with(
            api=custom_llm_api,
            api_prompt=mock_api_prompt,
            llm_context=mock_llm_context,
            tools=mock_tools,
            custom_serializer=mock_serializer,
        )
        assert instance is mock_api_instance_cls.return_value


@pytest.mark.asyncio
async def test_custom_llm_api_normalizes_langfuse_prompt(
    custom_llm_api, mock_llm_context, mock_exposed_entities_data
):
    """Test Langfuse prompt metadata stays outside the HA API prompt field."""
    prompt_object = MagicMock()

    with (
        patch(
            "custom_components.custom_conversation.api._get_exposed_entities",
            return_value=mock_exposed_entities_data,
        ),
        patch.object(
            custom_llm_api,
            "_async_get_api_prompt",
            new_callable=AsyncMock,
            return_value=(prompt_object, "Compiled prompt"),
        ),
        patch.object(
            custom_llm_api,
            "_async_get_tools",
            new_callable=AsyncMock,
            return_value=llm_component.LLMTools(
                tools=[], prompt="Platform prompt"
            ),
        ),
    ):
        instance = await custom_llm_api.async_get_api_instance(mock_llm_context)

    assert instance.api_prompt == "Compiled prompt"
    assert custom_llm_api.prompt_object is prompt_object


@pytest.mark.asyncio
async def test_custom_llm_api_get_api_prompt(custom_llm_api, hass, mock_llm_context, mock_prompt_manager, config_entry, mock_exposed_entities_data, device_registry, area_registry, floor_registry):
    """Test generating the API prompt."""

    with patch("custom_components.custom_conversation.api.async_device_supports_timers", return_value=True) as mock_supports_timers:

        expected_prompt = "Generated Prompt"
        mock_prompt_manager.get_api_prompt.return_value = expected_prompt

        prompt = await custom_llm_api._async_get_api_prompt(mock_llm_context, mock_exposed_entities_data)

        mock_supports_timers.assert_called_once_with(hass, mock_llm_context.device_id)

        mock_prompt_manager.get_api_prompt.assert_called_once()
        context_arg = mock_prompt_manager.get_api_prompt.call_args[0][0]
        config_entry_arg = mock_prompt_manager.get_api_prompt.call_args[0][1]

        assert isinstance(context_arg, PromptContext)
        assert context_arg.hass is hass
        assert context_arg.ha_name == "test home"
        assert context_arg.user_name == "Test User"
        assert context_arg.llm_context is mock_llm_context
        assert context_arg.location == "Test Area (floor: Test Floor)"
        assert context_arg.exposed_entities is mock_exposed_entities_data
        assert context_arg.supports_timers is True
        assert config_entry_arg is config_entry

        assert prompt == expected_prompt

@pytest.mark.asyncio
async def test_custom_llm_api_get_tools(
    custom_llm_api, hass, mock_llm_context, config_entry
):
    """Test Assist platform discovery and ignored-intent filtering."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={
            CONF_IGNORED_INTENTS_SECTION: {
                CONF_IGNORED_INTENTS: ["HassTurnOff", "My Intent"]
            }
        },
    )
    await hass.async_block_till_done()

    handler = MagicMock(spec=intent.IntentHandler, description=None, slot_schema=None)
    non_intent_tool = MagicMock(spec=llm.Tool)
    non_intent_tool.name = "GetDateTime"
    tools_from_platforms = [
        llm.IntentTool("HassTurnOn", handler),
        llm.IntentTool("HassTurnOff", handler),
        llm.IntentTool("My_Intent", handler),
        non_intent_tool,
    ]

    with patch(
        "custom_components.custom_conversation.api.llm_component.async_get_tools",
        new_callable=AsyncMock,
        return_value=llm_component.LLMTools(tools=tools_from_platforms),
    ) as mock_get_tools:
        tools = await custom_llm_api._async_get_tools(mock_llm_context)

    mock_get_tools.assert_awaited_once_with(
        hass, mock_llm_context, llm.LLM_API_ASSIST
    )
    assert [tool.name for tool in tools.tools] == ["HassTurnOn", "GetDateTime"]
    assert isinstance(tools.tools[0], CardPreservingIntentTool)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ignored", "expected_fields"),
    [
        (["HassGetCurrentDate"], {"time", "timezone"}),
        (["HassGetCurrentTime"], {"date", "weekday"}),
    ],
)
async def test_filtered_date_time_tool(
    custom_llm_api, mock_llm_context, config_entry, ignored, expected_fields
):
    """Test independent legacy date/time settings filter the combined tool."""
    custom_llm_api.hass.config_entries.async_update_entry(
        config_entry,
        options={
            CONF_IGNORED_INTENTS_SECTION: {
                CONF_IGNORED_INTENTS: ignored,
            }
        },
    )
    combined_tool = MagicMock(spec=llm.Tool)
    combined_tool.name = "GetDateTime"
    combined_tool.description = "Provides the current date and time."
    combined_tool.parameters = vol.Schema({})
    combined_tool.async_call = AsyncMock(
        return_value={
            "success": True,
            "result": {
                "date": "2026-08-12",
                "weekday": "Wednesday",
                "time": "12:34:56",
                "timezone": "UTC",
            },
        }
    )
    with patch(
        "custom_components.custom_conversation.api.llm_component.async_get_tools",
        new_callable=AsyncMock,
        return_value=llm_component.LLMTools(tools=[combined_tool]),
    ):
        llm_tools = await custom_llm_api._async_get_tools(mock_llm_context)

    tool = llm_tools.tools[0]
    assert isinstance(tool, FilteredDateTimeTool)
    result = await tool.async_call(
        custom_llm_api.hass,
        llm.ToolInput(tool_name=tool.name, tool_args={}),
        mock_llm_context,
    )
    assert set(result["result"]) == expected_fields


@pytest.mark.asyncio
async def test_card_preserving_intent_tool(hass, mock_llm_context):
    """Test cards removed by HA's serializer remain available to event consumers."""
    response = intent.IntentResponse(language="en")
    response.async_set_card("Title", "Content")
    serialized = llm.IntentResponseDict(response)
    delegated_tool = MagicMock(spec=llm.IntentTool)
    delegated_tool.name = "CardIntent"
    delegated_tool.description = "Return a card"
    delegated_tool.parameters = vol.Schema({})
    delegated_tool.async_call = AsyncMock(return_value=serialized)
    tool = CardPreservingIntentTool(delegated_tool)

    result = await tool.async_call(
        hass,
        llm.ToolInput(tool_name=tool.name, tool_args={}),
        mock_llm_context,
    )

    assert result["card"] == {
        "simple": {"title": "Title", "content": "Content"}
    }


@pytest.mark.asyncio
async def test_custom_llm_api_discovers_exposed_script(
    custom_llm_api, hass, mock_llm_context
):
    """Test exposed scripts are discovered through the HA 2026.8 platform API."""
    assert await async_setup_component(
        hass,
        "script",
        {"script": {"test_script": {"sequence": []}}},
    )
    await hass.async_block_till_done()

    with patch(
        "homeassistant.components.script.llm.async_should_expose",
        return_value=True,
    ):
        llm_tools = await custom_llm_api._async_get_tools(mock_llm_context)

    tools = llm_tools.tools
    tool_names = {tool.name for tool in tools}
    assert {"GetLiveContext", "test_script"} <= tool_names
    assert "GetDateTime" not in tool_names
    script_tool = next(tool for tool in tools if isinstance(tool, ScriptTool))
    result = await script_tool.async_call(
        hass,
        llm.ToolInput(tool_name=script_tool.name, tool_args={}),
        mock_llm_context,
    )
    assert result["success"] is True
