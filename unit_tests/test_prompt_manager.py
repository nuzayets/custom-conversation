"""Tests for the Custom Conversation prompt manager."""

import asyncio
import os
from threading import Event
from unittest.mock import MagicMock, Mock, patch

from langfuse.api import ConfigCategory, ScoreConfigDataType
import pytest

from custom_components.custom_conversation.const import (
    CONF_API_PROMPT_BASE,
    CONF_ENABLE_LANGFUSE,
    CONF_LANGFUSE_API_PROMPT_ID,
    CONF_LANGFUSE_BASE_PROMPT_ID,
    CONF_LANGFUSE_HOST,
    CONF_LANGFUSE_PUBLIC_KEY,
    CONF_LANGFUSE_SCORE_ENABLED,
    CONF_LANGFUSE_SECRET_KEY,
    CONF_LANGFUSE_SECTION,
    CONF_LANGFUSE_TRACING_ENABLED,
    DEFAULT_API_PROMPT_BASE,
    DEFAULT_INSTRUCTIONS_PROMPT,
    DEFAULT_PROMPT_NO_ENABLED_ENTITIES,
    LANGFUSE_SCORE_NAME,
    LANGFUSE_SCORE_NEGATIVE,
    LANGFUSE_SCORE_POSITIVE,
)
from custom_components.custom_conversation.prompt_manager import (
    LangfuseClient,
    LangfuseInitError,
    PromptContext,
    PromptManager,
)
from homeassistant.const import EVENT_HOMEASSISTANT_STOP


def _langfuse_config_entry(
    *,
    prompts_enabled: bool = True,
    tracing_enabled: bool = True,
    score_enabled: bool = False,
    host: str = "https://langfuse.example.com",
    public_key: str = "public-key",
    secret_key: str = "secret-key",
):
    return Mock(
        options={
            CONF_LANGFUSE_SECTION: {
                CONF_ENABLE_LANGFUSE: prompts_enabled,
                CONF_LANGFUSE_API_PROMPT_ID: "api-prompt",
                CONF_LANGFUSE_BASE_PROMPT_ID: "base-prompt",
                CONF_LANGFUSE_HOST: host,
                CONF_LANGFUSE_PUBLIC_KEY: public_key,
                CONF_LANGFUSE_SCORE_ENABLED: score_enabled,
                CONF_LANGFUSE_SECRET_KEY: secret_key,
                CONF_LANGFUSE_TRACING_ENABLED: tracing_enabled,
            }
        }
    )


@pytest.fixture
def prompt_manager(hass):
    """Create a PromptManager instance."""
    return PromptManager(hass)


async def test_get_base_prompt_default(prompt_manager, hass):
    """Test getting base prompt with defaults."""
    context = PromptContext(
        hass=hass,
        ha_name="Test Home",
        user_name="Test User",
    )

    prompt = await prompt_manager.async_get_base_prompt(context)

    assert "Current time is" in prompt
    assert DEFAULT_INSTRUCTIONS_PROMPT.strip() in prompt


async def test_get_base_prompt_custom(prompt_manager, hass, config_entry):
    """Test getting base prompt with custom configuration."""
    context = PromptContext(
        hass=hass,
        ha_name="Test Home",
        user_name="Test User",
    )

    prompt = await prompt_manager.async_get_base_prompt(context, config_entry)

    assert "Custom base prompt for Test Home" in prompt
    assert "Custom instructions for Test User" in prompt


async def test_get_api_prompt_no_entities(prompt_manager, hass):
    """Test API prompt when no entities are exposed."""
    context = PromptContext(
        hass=hass,
        ha_name="Test Home",
        exposed_entities=None,
    )

    prompt = await prompt_manager.get_api_prompt(context)

    assert prompt == DEFAULT_PROMPT_NO_ENABLED_ENTITIES


async def test_get_api_prompt_with_location(prompt_manager, hass, config_entry):
    """Test API prompt with location information."""
    context = PromptContext(
        hass=hass,
        ha_name="Test Home",
        location="Living Room",
        exposed_entities={"light.test": {"name": "Test Light"}},
    )

    prompt = await prompt_manager.get_api_prompt(context, config_entry)

    assert "Custom API base prompt" in prompt
    assert "Custom location prompt for Living Room" in prompt
    assert "Use GetLiveContext" in prompt
    assert "Test Light" in prompt


async def test_get_api_prompt_no_timers(prompt_manager, hass):
    """Test API prompt when timers are not supported."""
    context = PromptContext(
        hass=hass,
        ha_name="Test Home",
        exposed_entities={"light.test": {"name": "Test Light"}},
        supports_timers=False,
    )

    prompt = await prompt_manager.get_api_prompt(context)

    assert "This device is not able to start timers" in prompt



def test_get_prompt_config_no_config_entry(prompt_manager):
    """Test getting prompt config with no config entry."""
    result = prompt_manager._get_prompt_config(None, "test_key", "default_value")

    assert result == "default_value"


def test_get_prompt_config_with_config_entry(prompt_manager, config_entry):
    """Test getting prompt config with config entry."""
    result = prompt_manager._get_prompt_config(
        config_entry,
        CONF_API_PROMPT_BASE,
        DEFAULT_API_PROMPT_BASE
    )

    assert result == "Custom API base prompt"


@pytest.mark.parametrize("tracing_enabled", [True, False])
async def test_langfuse_client_keeps_entry_tracing_option(hass, tracing_enabled):
    """Test SDK ingestion stays active while entry tracing remains configurable."""
    config_entry = _langfuse_config_entry(tracing_enabled=tracing_enabled)
    client = Mock()

    with patch(
        "custom_components.custom_conversation.prompt_manager.Langfuse",
        return_value=client,
    ) as langfuse_constructor:
        result = await LangfuseClient.create(hass, config_entry)

    langfuse_constructor.assert_called_once_with(
        public_key="public-key",
        secret_key="secret-key",
        base_url="https://langfuse.example.com",
        tracing_enabled=True,
    )
    assert isinstance(result, LangfuseClient)
    assert result.tracing_enabled is tracing_enabled


async def test_langfuse_rejects_blank_entry_credentials_despite_environment(hass):
    """Test process environment cannot supply another entry's credentials."""
    config_entry = _langfuse_config_entry(public_key="", secret_key="")

    with (
        patch.dict(
            os.environ,
            {
                "LANGFUSE_PUBLIC_KEY": "environment-public-key",
                "LANGFUSE_SECRET_KEY": "environment-secret-key",
            },
        ),
        patch(
            "custom_components.custom_conversation.prompt_manager.Langfuse"
        ) as constructor,
        pytest.raises(LangfuseInitError, match="keys are required"),
    ):
        await LangfuseClient.create(hass, config_entry)

    constructor.assert_not_called()


async def test_langfuse_entry_endpoint_overrides_environment(hass):
    """Test a blank optional host selects cloud, not a process environment URL."""
    config_entry = _langfuse_config_entry(host="")
    client = MagicMock()

    with (
        patch.dict(
            os.environ,
            {"LANGFUSE_BASE_URL": "https://environment.example.com"},
        ),
        patch(
            "custom_components.custom_conversation.prompt_manager.Langfuse",
            return_value=client,
        ) as constructor,
    ):
        wrapper = await LangfuseClient.create(hass, config_entry)

    constructor.assert_called_once_with(
        public_key="public-key",
        secret_key="secret-key",
        base_url="https://cloud.langfuse.com",
        tracing_enabled=True,
    )
    assert isinstance(wrapper, LangfuseClient)
    await wrapper.cleanup()
    await LangfuseClient.shutdown_all(hass)


async def test_langfuse_client_created_for_tracing_only(hass):
    """Test tracing initializes without prompt management."""
    config_entry = _langfuse_config_entry(prompts_enabled=False)
    client = Mock()

    with patch(
        "custom_components.custom_conversation.prompt_manager.Langfuse",
        return_value=client,
    ):
        result = await LangfuseClient.create(hass, config_entry)

    assert isinstance(result, LangfuseClient)
    assert result.tracing_enabled is True


async def test_langfuse_scoring_requires_entry_tracing(hass):
    """Test scoring cannot target absent or stale conversation traces."""
    config_entry = _langfuse_config_entry(
        prompts_enabled=False,
        tracing_enabled=False,
        score_enabled=True,
    )

    with (
        patch(
            "custom_components.custom_conversation.prompt_manager.Langfuse"
        ) as constructor,
        pytest.raises(LangfuseInitError, match="scoring requires tracing"),
    ):
        await LangfuseClient.create(hass, config_entry)

    constructor.assert_not_called()


async def test_langfuse_score_create_cancellation_reuses_inflight_task(hass):
    """Test retrying a cancelled score create cannot duplicate the request."""
    started = Event()
    release = Event()
    create_calls = []
    client = MagicMock()
    client.api.score_configs.get.return_value = Mock(data=[])

    def create_score_config(**_kwargs):
        create_calls.append("create")
        started.set()
        release.wait()
        return Mock(id="score-config-id")

    client.api.score_configs.create = create_score_config

    with patch(
        "custom_components.custom_conversation.prompt_manager.Langfuse",
        return_value=client,
    ) as constructor:
        first_task = asyncio.create_task(
            LangfuseClient.create(
                hass,
                _langfuse_config_entry(score_enabled=True),
            )
        )
        await hass.async_add_executor_job(started.wait)
        first_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_task

        retry_task = asyncio.create_task(
            LangfuseClient.create(
                hass,
                _langfuse_config_entry(score_enabled=True),
            )
        )
        await asyncio.sleep(0)
        assert create_calls == ["create"]
        release.set()
        wrapper = await retry_task

    constructor.assert_called_once()
    assert isinstance(wrapper, LangfuseClient)
    assert wrapper.score_config_id == "score-config-id"
    assert create_calls == ["create"]
    await wrapper.cleanup()
    await LangfuseClient.shutdown_all(hass)


async def test_langfuse_shutdown_waits_for_inflight_score_setup(hass):
    """Test SDK shutdown follows completion of resource-owned score work."""
    started = Event()
    release = Event()
    events = []
    client = MagicMock()
    client.api.score_configs.get.return_value = Mock(data=[])

    def get_score_configs():
        events.append("get-start")
        started.set()
        release.wait()
        events.append("get-end")
        return Mock(data=[])

    def shutdown():
        events.append("shutdown")

    client.api.score_configs.get = get_score_configs
    client.api.score_configs.create.return_value = Mock(id="score-config-id")
    client.shutdown.side_effect = shutdown

    async def shutdown_on_stop(_event):
        await LangfuseClient.shutdown_all(hass)

    hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, shutdown_on_stop)

    with patch(
        "custom_components.custom_conversation.prompt_manager.Langfuse",
        return_value=client,
    ):
        create_task = asyncio.create_task(
            LangfuseClient.create(
                hass,
                _langfuse_config_entry(score_enabled=True),
            )
        )
        await hass.async_add_executor_job(started.wait)
        create_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await create_task

        shutdown_task = asyncio.create_task(hass.async_stop())
        await asyncio.sleep(0)
        assert events == ["get-start"]
        release.set()
        await shutdown_task

    assert events == ["get-start", "get-end", "shutdown"]


async def test_langfuse_score_config_initialization_is_shared(hass):
    """Test concurrent same-project entries create one score config."""
    client = MagicMock()
    client.api.score_configs.get.return_value = Mock(data=[])
    client.api.score_configs.create.return_value = Mock(id="shared-score-config")

    with patch(
        "custom_components.custom_conversation.prompt_manager.Langfuse",
        return_value=client,
    ):
        first, second = await asyncio.gather(
            LangfuseClient.create(
                hass,
                _langfuse_config_entry(score_enabled=True),
            ),
            LangfuseClient.create(
                hass,
                _langfuse_config_entry(score_enabled=True),
            ),
        )

    assert isinstance(first, LangfuseClient)
    assert isinstance(second, LangfuseClient)
    assert first.score_config_id == "shared-score-config"
    assert second.score_config_id == "shared-score-config"
    client.api.score_configs.get.assert_called_once_with()
    client.api.score_configs.create.assert_called_once_with(
        name=LANGFUSE_SCORE_NAME,
        data_type=ScoreConfigDataType.CATEGORICAL,
        categories=[
            ConfigCategory(label=LANGFUSE_SCORE_POSITIVE, value=1),
            ConfigCategory(label=LANGFUSE_SCORE_NEGATIVE, value=0),
        ],
        description="Score for Custom Conversation Home Assistant integration",
    )

    await first.cleanup()
    await second.cleanup()
    await LangfuseClient.shutdown_all(hass)


async def test_langfuse_client_retains_failed_resource_until_hass_stop(hass):
    """Test failed setup leaves a reusable SDK resource until HA stops."""
    config_entry = _langfuse_config_entry(score_enabled=True)
    client = MagicMock()
    client.api.score_configs.get.side_effect = RuntimeError("score API failed")

    with (
        patch(
            "custom_components.custom_conversation.prompt_manager.Langfuse",
            return_value=client,
        ),
        pytest.raises(LangfuseInitError),
    ):
        await LangfuseClient.create(hass, config_entry)

    client.shutdown.assert_not_called()
    client.flush.assert_called_once_with()
    await LangfuseClient.shutdown_all(hass)
    client.shutdown.assert_called_once_with()


async def test_langfuse_client_cleanup_shuts_down_sdk(hass):
    """Test cleanup terminates Langfuse workers."""
    client = Mock()
    wrapper = LangfuseClient(hass, client, {}, True)

    await wrapper.cleanup()
    await wrapper.cleanup()

    client.shutdown.assert_called_once_with()


def test_langfuse_clients_keep_observations_entry_specific(hass):
    """Test each wrapper starts observations on its own SDK client."""
    first_client = Mock()
    second_client = Mock()
    first = LangfuseClient(hass, first_client, {}, True)
    second = LangfuseClient(hass, second_client, {}, True)

    first.observe(name="first")
    second.observe(name="second")

    first_client.start_as_current_observation.assert_called_once_with(
        name="first", as_type="span"
    )
    second_client.start_as_current_observation.assert_called_once_with(
        name="second", as_type="span"
    )


async def test_langfuse_shared_resource_survives_entry_reload(hass):
    """Test a same-project resource remains usable across entry reloads."""
    client = MagicMock()

    with patch(
        "custom_components.custom_conversation.prompt_manager.Langfuse",
        return_value=client,
    ) as constructor:
        first = await LangfuseClient.create(hass, _langfuse_config_entry())
        second = await LangfuseClient.create(hass, _langfuse_config_entry())

    assert isinstance(first, LangfuseClient)
    assert isinstance(second, LangfuseClient)
    constructor.assert_called_once()

    await first.cleanup()
    client.shutdown.assert_not_called()
    client.flush.assert_not_called()

    await second.cleanup()
    client.shutdown.assert_not_called()
    client.flush.assert_called_once_with()

    with patch(
        "custom_components.custom_conversation.prompt_manager.Langfuse"
    ) as constructor:
        reloaded = await LangfuseClient.create(hass, _langfuse_config_entry())
    constructor.assert_not_called()
    assert isinstance(reloaded, LangfuseClient)

    await reloaded.cleanup()
    assert client.flush.call_count == 2
    await LangfuseClient.shutdown_all(hass)
    client.shutdown.assert_called_once_with()


async def test_langfuse_cleanup_cancellation_finishes_release(hass):
    """Test caller cancellation cannot strand a managed resource reference."""
    client = MagicMock()

    with patch(
        "custom_components.custom_conversation.prompt_manager.Langfuse",
        return_value=client,
    ):
        wrapper = await LangfuseClient.create(hass, _langfuse_config_entry())

    resources, lock = LangfuseClient._resource_state(hass)
    await lock.acquire()
    cleanup_task = asyncio.create_task(wrapper.cleanup())
    await asyncio.sleep(0)
    cleanup_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cleanup_task

    assert wrapper._released is False
    assert resources["public-key"].references == 1
    lock.release()

    await wrapper.cleanup()
    assert wrapper._released is True
    assert resources["public-key"].references == 0
    client.flush.assert_called_once_with()

    await wrapper.cleanup()
    client.flush.assert_called_once_with()
    await LangfuseClient.shutdown_all(hass)


async def test_langfuse_rejects_conflicting_same_project_credentials(hass):
    """Test a public-key collision cannot silently reuse another config."""
    client = MagicMock()

    with patch(
        "custom_components.custom_conversation.prompt_manager.Langfuse",
        return_value=client,
    ):
        first = await LangfuseClient.create(hass, _langfuse_config_entry())
        with pytest.raises(LangfuseInitError, match="active SDK resource"):
            await LangfuseClient.create(
                hass,
                _langfuse_config_entry(secret_key="different-secret"),
            )

    assert isinstance(first, LangfuseClient)
    await first.cleanup()
    client.shutdown.assert_not_called()
    client.flush.assert_called_once_with()
    await LangfuseClient.shutdown_all(hass)
    client.shutdown.assert_called_once_with()


async def test_langfuse_shutdown_all_continues_after_failure(hass):
    """Test one SDK shutdown failure does not skip other resources."""
    first_client = MagicMock()
    first_client.shutdown.side_effect = RuntimeError("shutdown failed")
    second_client = MagicMock()

    with patch(
        "custom_components.custom_conversation.prompt_manager.Langfuse",
        side_effect=[first_client, second_client],
    ):
        await LangfuseClient.create(
            hass,
            _langfuse_config_entry(public_key="first-public-key"),
        )
        await LangfuseClient.create(
            hass,
            _langfuse_config_entry(public_key="second-public-key"),
        )

    await LangfuseClient.shutdown_all(hass)

    first_client.shutdown.assert_called_once_with()
    second_client.shutdown.assert_called_once_with()


async def test_langfuse_constructor_cancellation_retains_trackable_resource(hass):
    """Test cancellation cannot orphan an executor-created SDK client."""
    started = Event()
    release = Event()
    client = MagicMock()

    def construct_client(**_kwargs):
        started.set()
        release.wait()
        return client

    with patch(
        "custom_components.custom_conversation.prompt_manager.Langfuse",
        side_effect=construct_client,
    ) as constructor:
        task = asyncio.create_task(
            LangfuseClient.create(hass, _langfuse_config_entry())
        )
        await hass.async_add_executor_job(started.wait)
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        wrapper = await LangfuseClient.create(hass, _langfuse_config_entry())

    constructor.assert_called_once()
    assert isinstance(wrapper, LangfuseClient)
    await wrapper.cleanup()
    await LangfuseClient.shutdown_all(hass)
    client.shutdown.assert_called_once_with()


async def test_langfuse_stop_waits_for_client_construction(hass):
    """Test HA stop cannot orphan an in-flight SDK constructor."""
    started = Event()
    release = Event()
    events = []
    client = MagicMock()

    def construct_client(**_kwargs):
        events.append("construct-start")
        started.set()
        release.wait()
        events.append("construct-end")
        return client

    def shutdown():
        events.append("shutdown")

    client.shutdown.side_effect = shutdown

    async def shutdown_on_stop(_event):
        await LangfuseClient.shutdown_all(hass)

    hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, shutdown_on_stop)

    with patch(
        "custom_components.custom_conversation.prompt_manager.Langfuse",
        side_effect=construct_client,
    ):
        create_task = asyncio.create_task(
            LangfuseClient.create(hass, _langfuse_config_entry())
        )
        await hass.async_add_executor_job(started.wait)
        stop_task = asyncio.create_task(hass.async_stop())
        await asyncio.sleep(0)
        assert events == ["construct-start"]
        release.set()
        wrapper = await create_task
        await stop_task

    assert isinstance(wrapper, LangfuseClient)
    assert events == ["construct-start", "construct-end", "shutdown"]


async def test_langfuse_stop_waits_for_last_owner_flush(hass):
    """Test HA stop cannot shut down during a last-owner flush."""
    started = Event()
    release = Event()
    events = []
    client = MagicMock()

    with patch(
        "custom_components.custom_conversation.prompt_manager.Langfuse",
        return_value=client,
    ):
        wrapper = await LangfuseClient.create(hass, _langfuse_config_entry())

    def flush():
        events.append("flush-start")
        started.set()
        release.wait()
        events.append("flush-end")

    def shutdown():
        events.append("shutdown")

    client.flush = flush
    client.shutdown.side_effect = shutdown

    async def shutdown_on_stop(_event):
        await LangfuseClient.shutdown_all(hass)

    hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, shutdown_on_stop)
    cleanup_task = asyncio.create_task(wrapper.cleanup())
    await hass.async_add_executor_job(started.wait)
    stop_task = asyncio.create_task(hass.async_stop())
    await asyncio.sleep(0)
    assert events == ["flush-start"]
    release.set()
    await cleanup_task
    await stop_task

    assert events == ["flush-start", "flush-end", "shutdown"]


async def test_langfuse_score_setup_cancellation_releases_reference(hass):
    """Test cancellation during score setup rolls back resource ownership."""
    started = Event()
    release = Event()
    client = MagicMock()
    client.flush.side_effect = RuntimeError("flush failed")

    def get_score_configs():
        started.set()
        release.wait()
        return Mock(data=[])

    client.api.score_configs.get = get_score_configs
    client.api.score_configs.create.return_value = Mock(id="score-config-id")

    with patch(
        "custom_components.custom_conversation.prompt_manager.Langfuse",
        return_value=client,
    ) as constructor:
        task = asyncio.create_task(
            LangfuseClient.create(
                hass,
                _langfuse_config_entry(score_enabled=True),
            )
        )
        await hass.async_add_executor_job(started.wait)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        wrapper = await LangfuseClient.create(
            hass,
            _langfuse_config_entry(score_enabled=True),
        )

    constructor.assert_called_once()
    client.flush.assert_called_once_with()
    assert isinstance(wrapper, LangfuseClient)
    assert wrapper.score_config_id == "score-config-id"
    await wrapper.cleanup()
    await LangfuseClient.shutdown_all(hass)

