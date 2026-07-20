"""Unit tests for the cc_llm module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.custom_conversation import CustomConversationConfigEntry
from custom_components.custom_conversation.cc_llm import async_update_llm_data
from custom_components.custom_conversation.const import DOMAIN, LLM_API_ID
from custom_components.custom_conversation.prompt_manager import (
    LangfuseClient,
    PromptManager,
)
from homeassistant.auth.models import User
from homeassistant.components.conversation import (
    ChatLog,
    ConversationInput,
    ConverseError,
    SystemContent,
)
from homeassistant.core import Context, HomeAssistant
from homeassistant.exceptions import HomeAssistantError


@pytest.fixture
def mock_user():
    """Mock user object."""
    user = MagicMock(spec=User)
    user.id = "test_user_id"
    user.name = "Test User"
    return user


@pytest.fixture
def mock_context(mock_user):
    """Mock context object."""
    context = MagicMock(spec=Context)
    context.user_id = mock_user.id
    return context


@pytest.fixture
def mock_user_input(mock_context):
    """Mock ConversationInput object."""
    return ConversationInput(
        text="Hello",
        context=mock_context,
        conversation_id="test-convo-id",
        device_id="test-device-id",
        satellite_id="test-satellite-id",
        language="en",
        agent_id="test-agent-id",
        extra_system_prompt=None,
    )


@pytest.fixture
def mock_chat_log(hass):
    """Mock ChatLog object."""
    log = ChatLog(
        hass,
        conversation_id="test-convo-id",
        content=[SystemContent(content="initial")],
    )
    log.llm_api = None
    log.extra_system_prompt = None
    return log


@pytest.fixture
def mock_prompt_manager():
    """Mock PromptManager object."""
    manager = MagicMock(spec=PromptManager)
    manager.async_get_base_prompt = AsyncMock(return_value="Base Prompt")
    return manager


async def test_async_update_llm_data_no_api(
    hass: HomeAssistant,
    mock_user_input: ConversationInput,
    config_entry: CustomConversationConfigEntry,
    mock_chat_log: ChatLog,
    mock_prompt_manager: PromptManager,
    mock_user: User,
):
    """Test async_update_llm_data when llm_api_name is None."""
    with patch(
        "homeassistant.auth.AuthManager.async_get_user",
        new_callable=AsyncMock,
        return_value=mock_user,
    ):
        await async_update_llm_data(
            hass,
            mock_user_input,
            config_entry,
            mock_chat_log,
            mock_prompt_manager,
            llm_api_name=None,
        )

    mock_prompt_manager.async_get_base_prompt.assert_awaited_once()
    prompt_context = mock_prompt_manager.async_get_base_prompt.call_args[0][0]
    assert prompt_context.ha_name == "test home"
    assert prompt_context.user_name == "Test User"
    assert mock_chat_log.content[0].content == "Base Prompt"
    assert mock_chat_log.content[0].role == "system"
    assert mock_chat_log.llm_api is None
    assert mock_chat_log.extra_system_prompt is None


async def test_async_update_llm_data_prefers_explicit_langfuse_client(
    hass: HomeAssistant,
    mock_user_input: ConversationInput,
    config_entry: CustomConversationConfigEntry,
    mock_chat_log: ChatLog,
    mock_prompt_manager: PromptManager,
):
    """Test mutable hass data cannot replace the conversation's client."""
    explicit_client = MagicMock(spec=LangfuseClient)
    stored_client = MagicMock(spec=LangfuseClient)
    hass.data.setdefault(DOMAIN, {})[config_entry.entry_id] = {
        "langfuse_client": stored_client
    }
    api = MagicMock()
    api.async_get_api_instance = AsyncMock(side_effect=HomeAssistantError("API failed"))

    with (
        patch(
            "custom_components.custom_conversation.cc_llm.CustomLLMAPI",
            return_value=api,
        ),
        pytest.raises(ConverseError),
    ):
        await async_update_llm_data(
            hass,
            mock_user_input,
            config_entry,
            mock_chat_log,
            mock_prompt_manager,
            llm_api_name=LLM_API_ID,
            langfuse_client=explicit_client,
        )

    api.set_langfuse_client.assert_called_once_with(explicit_client)


async def test_extra_prompt_updates_explicit_langfuse_client(
    hass: HomeAssistant,
    mock_user_input: ConversationInput,
    config_entry: CustomConversationConfigEntry,
    mock_chat_log: ChatLog,
    mock_prompt_manager: PromptManager,
):
    """Test extra-prompt metadata stays on the root observation client."""
    explicit_client = MagicMock(spec=LangfuseClient)
    mock_user_input.extra_system_prompt = "Extra instructions"

    await async_update_llm_data(
        hass,
        mock_user_input,
        config_entry,
        mock_chat_log,
        mock_prompt_manager,
        llm_api_name=None,
        langfuse_client=explicit_client,
    )

    explicit_client.update_current_span.assert_called_once_with(
        metadata={"tags": ["extra_system_prompt"]}
    )
