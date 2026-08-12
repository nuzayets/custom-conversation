"""Tests for completion streaming and serialization."""

import asyncio
from contextlib import contextmanager, nullcontext
from datetime import time
from decimal import Decimal
from enum import Enum
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from litellm import OpenAIError, RateLimitError
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
import pytest

from custom_components.custom_conversation.const import (
    CONF_AGENTS_SECTION,
    CONF_ENABLE_HASS_AGENT,
    CONF_ENABLE_LLM_AGENT,
    CONF_LANGFUSE_SECRET_KEY,
    CONF_LANGFUSE_SECTION,
    CONF_MAX_TOKENS,
    CONF_PRIMARY_API_KEY,
    CONF_PRIMARY_BASE_URL,
    CONF_PRIMARY_CHAT_MODEL,
    CONF_PRIMARY_PROVIDER,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    LLM_API_ID,
)
from custom_components.custom_conversation.conversation import (
    CustomConversationEntity,
    _async_close_stream,
    _convert_content_to_param,
)
from custom_components.custom_conversation.prompt_manager import LangfuseClient
from homeassistant.components import conversation
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import Context
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import intent
from homeassistant.setup import async_setup_component


class ResultKind(Enum):
    """Non-JSON-native result value."""

    READY = "ready"


def _entry():
    return Mock(
        entry_id="entry-id",
        title="Test entry",
        data={
            CONF_PRIMARY_PROVIDER: "openai",
            CONF_PRIMARY_CHAT_MODEL: "gpt-test",
            CONF_PRIMARY_BASE_URL: "https://example.com",
            CONF_PRIMARY_API_KEY: "api-key",
        },
        options={
            CONF_TEMPERATURE: 0.2,
            CONF_TOP_P: 0.9,
            CONF_MAX_TOKENS: 128,
            CONF_LANGFUSE_SECTION: {
                CONF_LANGFUSE_SECRET_KEY: "must-not-leak",
            },
        },
    )


def _litellm_wrapper(completion_stream=None):
    return CustomStreamWrapper(
        completion_stream=completion_stream,
        model="openai/gpt-test",
        logging_obj=SimpleNamespace(
            model_call_details={
                "custom_llm_provider": "openai",
                "litellm_params": {},
            },
            stream_options=None,
            messages=[],
        ),
        custom_llm_provider="openai",
    )


def _content_chunk(content: str):
    return _litellm_wrapper().model_response_creator(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": content},
                    "finish_reason": None,
                }
            ]
        }
    )


def _usage_chunk():
    return _litellm_wrapper().model_response_creator(
        {
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "total_tokens": 6,
            }
        }
    )


def _finish_chunk():
    return _litellm_wrapper().model_response_creator(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ]
        }
    )


def _entity(hass, raw_stream, events):
    generation = Mock()

    @contextmanager
    def observation_context():
        events.append("observation-enter")
        try:
            yield generation
        finally:
            events.append("observation-exit")

    langfuse_client = Mock(spec=LangfuseClient)
    langfuse_client.observe.return_value = observation_context()
    router = Mock()
    router.acompletion = AsyncMock(return_value=raw_stream)
    entity = CustomConversationEntity(
        _entry(),
        Mock(),
        hass,
        langfuse_client,
    )
    entity._router = router
    return entity, langfuse_client, generation, router


async def test_generation_observation_spans_stream_consumption(hass):
    """Test the generation remains active until its stream is exhausted."""
    events = []

    async def raw_stream():
        events.append("stream-start")
        assert "observation-exit" not in events
        yield _content_chunk("hello")
        yield _usage_chunk()
        yield _finish_chunk()
        events.append("stream-end")

    entity, langfuse_client, generation, router = _entity(hass, raw_stream(), events)

    deltas = [
        delta
        async for delta in entity._async_generate_completion(
            entity.entry,
            [],
            None,
            "conversation-id",
        )
    ]

    assert deltas == [{"role": "assistant", "content": "hello"}]
    assert events == [
        "observation-enter",
        "stream-start",
        "stream-end",
        "observation-exit",
    ]
    router.acompletion.assert_awaited_once()
    observe_kwargs = langfuse_client.observe.call_args.kwargs
    assert observe_kwargs["as_type"] == "generation"
    assert "must-not-leak" not in repr(observe_kwargs)
    final_update = generation.update.call_args_list[-1].kwargs
    assert final_update["output"] == {"content": "hello"}
    assert final_update["model"] == "openai/gpt-test"
    assert final_update["usage_details"] == {
        "input": 4,
        "output": 2,
        "total": 6,
    }
    assert final_update["completion_start_time"] is not None


async def test_generation_records_stream_errors(hass):
    """Test stream errors annotate and close the generation."""
    events = []

    async def raw_stream():
        yield _content_chunk("partial")
        raise RuntimeError("stream failed")

    entity, _client, generation, _router = _entity(hass, raw_stream(), events)

    with pytest.raises(HomeAssistantError, match="Error generating LLM completion"):
        async for _delta in entity._async_generate_completion(
            entity.entry,
            [],
            None,
            "conversation-id",
        ):
            pass

    assert events[-1] == "observation-exit"
    assert any(
        call.kwargs.get("level") == "ERROR"
        and call.kwargs.get("status_message") == "stream failed"
        for call in generation.update.call_args_list
    )


@pytest.mark.parametrize(
    "api_error",
    [
        RateLimitError(
            message="rate limited",
            llm_provider="openai",
            model="gpt-test",
        ),
        OpenAIError(original_exception=RuntimeError("API failed")),
    ],
)
async def test_llm_api_errors_keep_specific_error_types(hass, api_error):
    """Test lazy stream startup does not wrap LiteLLM API errors."""
    events = []

    async def unused_stream():
        yield _content_chunk("unused")

    entity, _client, generation, router = _entity(
        hass,
        unused_stream(),
        events,
    )
    router.acompletion.side_effect = api_error

    class ChatLog:
        llm_api = None
        content = []
        conversation_id = "conversation-id"
        unresponded_tool_results = False

        async def async_add_delta_content_stream(self, _agent_id, stream):
            async for delta in stream:
                yield delta

    with (
        patch(
            "custom_components.custom_conversation.conversation.async_update_llm_data",
            AsyncMock(return_value=None),
        ),
        pytest.raises(type(api_error)),
    ):
        await entity._async_handle_message_with_llm(
            Mock(agent_id="agent-id"),
            ChatLog(),
        )

    assert any(
        call.kwargs.get("level") == "ERROR" for call in generation.update.call_args_list
    )


async def test_generation_records_consumer_cancellation(hass):
    """Test closing a partially consumed stream records cancellation."""
    events = []

    async def raw_stream():
        try:
            yield _content_chunk("partial")
            await asyncio.Event().wait()
        finally:
            events.append("raw-stream-closed")

    entity, _client, generation, _router = _entity(hass, raw_stream(), events)
    stream = entity._async_generate_completion(
        entity.entry,
        [],
        None,
        "conversation-id",
    )

    assert await stream.__anext__() == {
        "role": "assistant",
        "content": "partial",
    }
    await stream.aclose()

    assert events[-2:] == ["raw-stream-closed", "observation-exit"]
    assert generation.update.call_args_list[-1].kwargs["output"] == {
        "content": "partial"
    }
    assert any(
        call.kwargs.get("level") == "WARNING"
        and call.kwargs.get("status_message") == "Generation cancelled"
        for call in generation.update.call_args_list
    )


async def test_generation_records_task_cancellation(hass):
    """Test cancellation while awaiting a provider chunk closes all streams."""
    events = []
    first_chunk_received = asyncio.Event()

    async def raw_stream():
        try:
            yield _content_chunk("partial")
            await asyncio.Event().wait()
        finally:
            events.append("raw-stream-closed")

    entity, _client, generation, _router = _entity(hass, raw_stream(), events)

    async def consume():
        async for _delta in entity._async_generate_completion(
            entity.entry,
            [],
            None,
            "conversation-id",
        ):
            first_chunk_received.set()

    task = asyncio.create_task(consume())
    await first_chunk_received.wait()
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert events[-2:] == ["raw-stream-closed", "observation-exit"]
    assert any(
        call.kwargs.get("level") == "WARNING"
        and call.kwargs.get("status_message") == "Generation cancelled"
        for call in generation.update.call_args_list
    )


async def test_close_stream_supports_legacy_litellm_wrapper():
    """Test the LiteLLM 1.81 wrapper fallback closes its provider stream."""
    closed = asyncio.Event()

    async def provider_stream():
        try:
            yield "chunk"
            await asyncio.Event().wait()
        finally:
            closed.set()

    provider = provider_stream()
    assert await provider.__anext__() == "chunk"
    legacy_wrapper = _litellm_wrapper(provider)

    await _async_close_stream(legacy_wrapper)

    assert closed.is_set()


async def test_router_is_built_once_for_concurrent_requests(hass):
    """Test concurrent first use shares one Router construction."""
    entity = CustomConversationEntity(_entry(), Mock(), hass)
    router = Mock()

    with patch.object(entity, "_build_router", return_value=router) as build_router:
        first, second = await asyncio.gather(
            entity._async_get_router(entity.entry),
            entity._async_get_router(entity.entry),
        )

    assert first is router
    assert second is router
    build_router.assert_called_once_with(entity.entry)


async def test_router_build_preloads_cost_calculator(hass):
    """Test Router construction resolves completion-time lazy imports."""
    accesses: list[str] = []

    class LazyImports:
        @property
        def completion_cost(self) -> None:
            accesses.append("completion_cost")

        @property
        def cost_per_token(self) -> None:
            accesses.append("cost_per_token")

        @property
        def response_cost_calculator(self) -> None:
            accesses.append("response_cost_calculator")

    entity = CustomConversationEntity(_entry(), Mock(), hass)
    router = Mock()
    with (
        patch(
            "custom_components.custom_conversation.conversation.Router",
            return_value=router,
        ),
        patch(
            "custom_components.custom_conversation.conversation.litellm",
            new=LazyImports(),
        ),
    ):
        result = await entity._async_get_router(entity.entry)

    assert result is router
    assert accesses == [
        "completion_cost",
        "cost_per_token",
        "response_cost_calculator",
    ]


async def test_fallback_tags_only_successful_agent(hass, config_entry):
    """Test HASS-to-LLM fallback does not claim HASS handled the request."""
    assert await async_setup_component(hass, "custom_conversation", {})
    hass.config_entries.async_update_entry(
        config_entry,
        options={
            **config_entry.options,
            CONF_LLM_HASS_API: LLM_API_ID,
            CONF_AGENTS_SECTION: {
                CONF_ENABLE_HASS_AGENT: True,
                CONF_ENABLE_LLM_AGENT: True,
            },
        },
    )
    await hass.config_entries.async_reload(config_entry.entry_id)
    agent = conversation.async_get_agent(hass, config_entry.entry_id)

    langfuse_client = Mock(spec=LangfuseClient)
    langfuse_client.propagate.side_effect = lambda **_kwargs: nullcontext()
    langfuse_client.observe.side_effect = lambda **_kwargs: nullcontext(Mock())
    agent._langfuse_client = langfuse_client

    hass_response = intent.IntentResponse(language="en")
    hass_response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, "unknown")
    hass_result = conversation.ConversationResult(
        response=hass_response,
        conversation_id="conversation-id",
    )
    llm_response = intent.IntentResponse(language="en")
    llm_response.async_set_speech("handled")
    llm_result = conversation.ConversationResult(
        response=llm_response,
        conversation_id="conversation-id",
    )

    with (
        patch.object(
            agent,
            "_async_handle_message_with_hass",
            AsyncMock(return_value=hass_result),
        ),
        patch.object(
            agent,
            "_async_handle_message_with_llm",
            AsyncMock(return_value=(llm_result, {})),
        ),
    ):
        result = await conversation.async_converse(
            hass,
            "hello",
            "conversation-id",
            Context(),
            agent_id=config_entry.entry_id,
        )

    assert result.response.speech["plain"]["speech"] == "handled"
    tag_sets = [
        call.kwargs["tags"] for call in langfuse_client.propagate.call_args_list
    ]
    assert ["handling_agent:llm"] in tag_sets
    assert ["handling_agent:home_assistant"] not in tag_sets


def test_tool_results_serialize_home_assistant_values():
    """Test nested non-JSON-native tool results do not crash."""
    content = conversation.ToolResultContent(
        agent_id="agent-id",
        tool_call_id="tool-call-id",
        tool_name="test-tool",
        tool_result={
            "time": time(8, 30),
            "decimal": Decimal("1.25"),
            "enum": ResultKind.READY,
            "set": {"one", "two"},
            "nested": [{"time": time(9, 45)}],
        },
    )

    message = _convert_content_to_param(content)
    serialized = json.loads(message["content"])

    assert serialized["time"]["isoformat"] == "08:30:00"
    assert "1.25" in serialized["decimal"]["repr"]
    assert "READY" in serialized["enum"]["repr"]
    assert sorted(serialized["set"]) == ["one", "two"]
    assert serialized["nested"][0]["time"]["isoformat"] == "09:45:00"
