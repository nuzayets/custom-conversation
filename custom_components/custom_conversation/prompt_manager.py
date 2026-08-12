"""Prompt management for  Custom Conversation component."""

from __future__ import annotations

import asyncio
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from langfuse import Langfuse, propagate_attributes
from langfuse.api import ConfigCategory, ScoreConfigDataType
from langfuse.model import Prompt

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import TemplateError
from homeassistant.helpers import template
from homeassistant.util import dt as dt_util, yaml as yaml_util

from .const import (
    CONF_API_PROMPT_BASE,
    CONF_CUSTOM_PROMPTS_SECTION,
    CONF_ENABLE_LANGFUSE,
    CONF_INSTRUCTIONS_PROMPT,
    CONF_LANGFUSE_API_PROMPT_ID,
    CONF_LANGFUSE_API_PROMPT_LABEL,
    CONF_LANGFUSE_BASE_PROMPT_ID,
    CONF_LANGFUSE_BASE_PROMPT_LABEL,
    CONF_LANGFUSE_HOST,
    CONF_LANGFUSE_PUBLIC_KEY,
    CONF_LANGFUSE_SCORE_ENABLED,
    CONF_LANGFUSE_SECRET_KEY,
    CONF_LANGFUSE_SECTION,
    CONF_LANGFUSE_TRACING_ENABLED,
    CONF_PROMPT_BASE,
    CONF_PROMPT_DEVICE_KNOWN_LOCATION,
    CONF_PROMPT_DEVICE_UNKNOWN_LOCATION,
    CONF_PROMPT_EXPOSED_ENTITIES,
    CONF_PROMPT_LIVE_CONTEXT,
    CONF_PROMPT_NO_ENABLED_ENTITIES,
    CONF_PROMPT_TIMERS_UNSUPPORTED,
    DEFAULT_API_PROMPT_BASE,
    DEFAULT_API_PROMPT_DEVICE_KNOWN_LOCATION,
    DEFAULT_API_PROMPT_DEVICE_UNKNOWN_LOCATION,
    DEFAULT_API_PROMPT_EXPOSED_ENTITIES,
    DEFAULT_API_PROMPT_LIVE_CONTEXT,
    DEFAULT_API_PROMPT_TIMERS_UNSUPPORTED,
    DEFAULT_BASE_PROMPT,
    DEFAULT_INSTRUCTIONS_PROMPT,
    DEFAULT_PROMPT_NO_ENABLED_ENTITIES,
    DOMAIN,
    LANGFUSE_SCORE_NAME,
    LANGFUSE_SCORE_NEGATIVE,
    LANGFUSE_SCORE_POSITIVE,
    LOGGER,
)

_LANGFUSE_RESOURCES = "langfuse_resources"
_LANGFUSE_RESOURCE_LOCK = "langfuse_resource_lock"
_LANGFUSE_CLOUD_URL = "https://cloud.langfuse.com"


@dataclass
class _LangfuseResource:
    """Shared SDK resource for entries using the same Langfuse project."""

    client: Langfuse
    host: str | None
    secret_key: str
    references: int = 1
    score_config_id: str | None = None
    score_config_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    score_config_task: asyncio.Task[str] | None = None
    flush_task: asyncio.Task[None] | None = None


class LangfuseError(Exception):
    """Base class for Langfuse errors."""


class LangfuseInitError(LangfuseError):
    """Error initializing Langfuse client."""


class LangfuseResourceConflictError(LangfuseInitError):
    """Error raised when an SDK singleton has conflicting credentials."""


class LangfusePromptError(LangfuseError):
    """Error getting or compiling Langfuse prompt."""


@dataclass
class PromptContext:
    """Context for prompt generation."""

    hass: HomeAssistant
    ha_name: str
    user_name: str | None = None
    llm_context: Any | None = None
    location: str | None = None
    exposed_entities: dict | None = None
    supports_timers: bool = True


class PromptManager:
    """Manager for Custom Conversation prompts."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the prompt manager."""
        self.hass = hass
        self._langfuse_client = None

    def _get_prompt_config(
        self, config_entry: ConfigEntry | None, key: str, default: str
    ) -> str:
        """Get prompt configuration with fallback to defaults."""
        if not config_entry:
            return default

        return config_entry.options.get(CONF_CUSTOM_PROMPTS_SECTION, {}).get(
            key, default
        )

    async def _get_langfuse_prompt(
        self, prompt_id: str, variables: dict[str, Any]
    ) -> tuple[Prompt, str] | None:
        """Get a prompt from Langfuse if enabled."""
        if not self._langfuse_client:
            return None

        try:
            return await self._langfuse_client.get_prompt(prompt_id, variables)
        except Exception as err:
            LOGGER.error("Error getting Langfuse prompt: %s", err)
            return None

    async def async_get_base_prompt(
        self, context: PromptContext, config_entry: ConfigEntry | None = None
    ) -> tuple[Prompt, str] | str:
        """Get the base prompt with rendered template."""
        if config_entry and config_entry.options.get(CONF_LANGFUSE_SECTION, {}).get(
            CONF_ENABLE_LANGFUSE
        ):
            result = await self._get_langfuse_prompt(
                config_entry.options.get(CONF_LANGFUSE_SECTION, {}).get(
                    CONF_LANGFUSE_BASE_PROMPT_ID
                ),
                {
                    "current_time": dt_util.now().strftime("%H:%M"),
                    "current_date": dt_util.now().strftime("%Y-%m-%d"),
                    "ha_name": context.ha_name,
                    "user_name": context.user_name,
                },
            )
            if result is not None:
                prompt_object, langfuse_prompt = result
                if langfuse_prompt:
                    return prompt_object, langfuse_prompt

        try:
            base_prompt = self._get_prompt_config(
                config_entry, CONF_PROMPT_BASE, DEFAULT_BASE_PROMPT
            )
            instructions_prompt = self._get_prompt_config(
                config_entry, CONF_INSTRUCTIONS_PROMPT, DEFAULT_INSTRUCTIONS_PROMPT
            )

            return template.Template(
                base_prompt + "\n" + instructions_prompt,
                context.hass,
            ).async_render(
                {
                    "ha_name": context.ha_name,
                    "user_name": context.user_name,
                    "llm_context": context.llm_context,
                },
                parse_result=False,
            )
        except TemplateError as err:
            LOGGER.error("Error rendering base prompt: %s", err)
            raise

    async def get_api_prompt(
        self, context: PromptContext, config_entry: ConfigEntry | None = None
    ) -> tuple[Prompt, str] | str:
        """Get the API prompt based on context."""
        if config_entry and config_entry.options.get(CONF_LANGFUSE_SECTION, {}).get(
            CONF_ENABLE_LANGFUSE
        ):
            result = await self._get_langfuse_prompt(
                config_entry.options.get(CONF_LANGFUSE_SECTION, {}).get(
                    CONF_LANGFUSE_API_PROMPT_ID
                ),
                {
                    "current_time": dt_util.now().strftime("%H:%M"),
                    "current_date": dt_util.now().strftime("%Y-%m-%d"),
                    "ha_name": context.ha_name,
                    "user_name": (
                        context.user_name if context.user_name else "unknown"
                    ),
                    "location": (context.location if context.location else "unknown"),
                    "exposed_entities": (
                        yaml_util.dump(list(context.exposed_entities.values()))
                        if context.exposed_entities
                        else None
                    ),
                    "supports_timers": (
                        "This device is not able to start timers."
                        if not context.supports_timers
                        else ""
                    ),
                    "live_context": self._get_prompt_config(
                        config_entry,
                        CONF_PROMPT_LIVE_CONTEXT,
                        DEFAULT_API_PROMPT_LIVE_CONTEXT,
                    ),
                },
            )
            if result is not None:
                prompt_object, langfuse_prompt = result
                if langfuse_prompt:
                    return prompt_object, langfuse_prompt
        prompt_parts = []

        if not context.exposed_entities:
            return self._get_prompt_config(
                config_entry,
                CONF_PROMPT_NO_ENABLED_ENTITIES,
                DEFAULT_PROMPT_NO_ENABLED_ENTITIES,
            )

        # Add base API prompt
        prompt_parts.append(
            self._get_prompt_config(
                config_entry, CONF_API_PROMPT_BASE, DEFAULT_API_PROMPT_BASE
            )
        )

        # Add location-specific prompt
        if context.location:
            location_prompt = self._get_prompt_config(
                config_entry,
                CONF_PROMPT_DEVICE_KNOWN_LOCATION,
                DEFAULT_API_PROMPT_DEVICE_KNOWN_LOCATION,
            )
            prompt_parts.append(
                template.Template(location_prompt, context.hass).async_render(
                    {"location": context.location}, parse_result=False
                )
            )
        else:
            prompt_parts.append(
                self._get_prompt_config(
                    config_entry,
                    CONF_PROMPT_DEVICE_UNKNOWN_LOCATION,
                    DEFAULT_API_PROMPT_DEVICE_UNKNOWN_LOCATION,
                )
            )

        # Add timer capability prompt if needed
        if not context.supports_timers:
            prompt_parts.append(
                self._get_prompt_config(
                    config_entry,
                    CONF_PROMPT_TIMERS_UNSUPPORTED,
                    DEFAULT_API_PROMPT_TIMERS_UNSUPPORTED,
                )
            )

        # Add exposed entities prompt and data
        if context.exposed_entities:
            prompt_parts.append(
                self._get_prompt_config(
                    config_entry,
                    CONF_PROMPT_LIVE_CONTEXT,
                    DEFAULT_API_PROMPT_LIVE_CONTEXT,
                )
            )
            prompt_parts.append(
                self._get_prompt_config(
                    config_entry,
                    CONF_PROMPT_EXPOSED_ENTITIES,
                    DEFAULT_API_PROMPT_EXPOSED_ENTITIES,
                )
            )
            prompt_parts.append(yaml_util.dump(list(context.exposed_entities.values())))

        return "\n".join(prompt_parts)

    def set_langfuse_client(self, langfuse_client: Any) -> None:
        """Set the Langfuse client."""
        self._langfuse_client = langfuse_client


class LangfuseClient:
    """Client for Langfuse prompt management."""

    def __init__(
        self,
        hass: HomeAssistant,
        client: Langfuse,
        prompts: dict,
        tracing_enabled: bool,
        score_config_id: str | None = None,
        resource_key: str | None = None,
    ) -> None:
        """Initialize the client."""
        self._client = client
        self.hass = hass
        self.prompts = prompts
        self.tracing_enabled = tracing_enabled
        self.score_config_id = score_config_id
        self._resource_key = resource_key
        self._managed_resource = resource_key is not None
        self._released = False
        self._cleanup_task: asyncio.Task[None] | None = None

    @staticmethod
    def _resource_state(hass: HomeAssistant):
        domain_data = hass.data.setdefault(DOMAIN, {})
        resources = domain_data.setdefault(_LANGFUSE_RESOURCES, {})
        lock = domain_data.setdefault(_LANGFUSE_RESOURCE_LOCK, asyncio.Lock())
        return resources, lock

    @classmethod
    async def _acquire_sdk_client(
        cls,
        hass: HomeAssistant,
        langfuse_options: dict[str, Any],
    ) -> tuple[Langfuse, str]:
        public_key = langfuse_options[CONF_LANGFUSE_PUBLIC_KEY]
        secret_key = langfuse_options[CONF_LANGFUSE_SECRET_KEY]
        host = langfuse_options.get(CONF_LANGFUSE_HOST) or _LANGFUSE_CLOUD_URL
        resources, lock = cls._resource_state(hass)
        async with lock:
            if resource := resources.get(public_key):
                if resource.host != host or resource.secret_key != secret_key:
                    raise LangfuseResourceConflictError(
                        "Langfuse credentials changed for an active SDK resource"
                    )
                resource.references += 1
                return resource.client, public_key

            async def create_client() -> Langfuse:
                return await hass.async_add_executor_job(
                    lambda: Langfuse(
                        public_key=public_key,
                        secret_key=secret_key,
                        base_url=host,
                        tracing_enabled=True,
                    )
                )

            client_job = hass.async_create_task(
                create_client(),
                name=f"{DOMAIN} Langfuse client construction",
                eager_start=False,
            )
            try:
                client = await asyncio.shield(client_job)
            except asyncio.CancelledError as cancellation:
                try:
                    client = await client_job
                except Exception as err:
                    raise cancellation from err
                resources[public_key] = _LangfuseResource(
                    client=client,
                    host=host,
                    secret_key=secret_key,
                    references=0,
                )
                raise
            resources[public_key] = _LangfuseResource(
                client=client,
                host=host,
                secret_key=secret_key,
            )
            return client, public_key

    @classmethod
    async def _release_sdk_client(
        cls,
        hass: HomeAssistant,
        resource_key: str,
    ) -> None:
        resources, lock = cls._resource_state(hass)
        flush_task = None
        async with lock:
            resource = resources.get(resource_key)
            if resource is None:
                return
            resource.references -= 1
            if resource.references == 0:
                if resource.flush_task is None or resource.flush_task.done():
                    resource.flush_task = hass.async_create_task(
                        cls._flush_resource(hass, resource),
                        name=f"{DOMAIN} Langfuse flush",
                        eager_start=False,
                    )
                flush_task = resource.flush_task
        if flush_task is not None:
            await asyncio.shield(flush_task)

    @staticmethod
    async def _flush_resource(
        hass: HomeAssistant,
        resource: _LangfuseResource,
    ) -> None:
        await hass.async_add_executor_job(resource.client.flush)

    @classmethod
    async def _ensure_score_config(
        cls,
        hass: HomeAssistant,
        resource_key: str,
    ) -> str:
        resources, lock = cls._resource_state(hass)
        async with lock:
            resource = resources[resource_key]
        async with resource.score_config_lock:
            if resource.score_config_id is not None:
                return resource.score_config_id
            if resource.score_config_task is None:
                resource.score_config_task = hass.async_create_task(
                    cls._initialize_score_config(hass, resource),
                    name=f"{DOMAIN} score configuration",
                    eager_start=False,
                )
                resource.score_config_task.add_done_callback(
                    lambda task: cls._score_config_task_done(resource, task)
                )
            score_config_task = resource.score_config_task
        return await asyncio.shield(score_config_task)

    @staticmethod
    def _score_config_task_done(
        resource: _LangfuseResource,
        task: asyncio.Task[str],
    ) -> None:
        if task.cancelled():
            if resource.score_config_task is task:
                resource.score_config_task = None
            return
        if task.exception() is not None and resource.score_config_task is task:
            resource.score_config_task = None

    @staticmethod
    async def _initialize_score_config(
        hass: HomeAssistant,
        resource: _LangfuseResource,
    ) -> str:
        score_configs = await hass.async_add_executor_job(
            resource.client.api.score_configs.get
        )
        score_config = next(
            (
                score
                for score in score_configs.data
                if score.name == LANGFUSE_SCORE_NAME
            ),
            None,
        )
        if score_config is None:
            score_config = await hass.async_add_executor_job(
                lambda: resource.client.api.score_configs.create(
                    name=LANGFUSE_SCORE_NAME,
                    data_type=ScoreConfigDataType.CATEGORICAL,
                    categories=[
                        ConfigCategory(
                            label=LANGFUSE_SCORE_POSITIVE,
                            value=1,
                        ),
                        ConfigCategory(
                            label=LANGFUSE_SCORE_NEGATIVE,
                            value=0,
                        ),
                    ],
                    description="Score for Custom Conversation Home Assistant integration",
                )
            )
        resource.score_config_id = score_config.id
        return resource.score_config_id

    @classmethod
    async def shutdown_all(cls, hass: HomeAssistant) -> None:
        """Shut down shared SDK resources when Home Assistant stops."""
        resources, lock = cls._resource_state(hass)
        async with lock:
            retained_resources = list(resources.values())
            resources.clear()
        for resource in retained_resources:
            if resource.score_config_task is not None:
                try:
                    await resource.score_config_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    LOGGER.warning(
                        "Error finishing Langfuse score configuration",
                        exc_info=True,
                    )
            if resource.flush_task is not None:
                try:
                    await resource.flush_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    LOGGER.warning(
                        "Error finishing Langfuse flush",
                        exc_info=True,
                    )
            try:
                await hass.async_add_executor_job(resource.client.shutdown)
            except Exception:
                LOGGER.warning(
                    "Error shutting down Langfuse resource",
                    exc_info=True,
                )

    @classmethod
    async def create(
        cls, hass: HomeAssistant, config_entry: ConfigEntry
    ) -> LangfuseClient | None:
        """Create a Langfuse client instance."""
        langfuse_options = config_entry.options.get(CONF_LANGFUSE_SECTION, {})
        prompts_enabled = langfuse_options.get(CONF_ENABLE_LANGFUSE, False)
        tracing_enabled = langfuse_options.get(CONF_LANGFUSE_TRACING_ENABLED, False)
        score_enabled = langfuse_options.get(CONF_LANGFUSE_SCORE_ENABLED, False)
        if not (prompts_enabled or tracing_enabled or score_enabled):
            return None
        if score_enabled and not tracing_enabled:
            raise LangfuseInitError("Langfuse scoring requires tracing")
        if not langfuse_options.get(
            CONF_LANGFUSE_PUBLIC_KEY
        ) or not langfuse_options.get(CONF_LANGFUSE_SECRET_KEY):
            raise LangfuseInitError(
                "Langfuse public and secret keys are required when Langfuse is enabled"
            )
        # Set up prompt dictionary from config entry
        prompts = {
            langfuse_options.get(CONF_LANGFUSE_BASE_PROMPT_ID): langfuse_options.get(
                CONF_LANGFUSE_BASE_PROMPT_LABEL, "production"
            ),
            langfuse_options.get(CONF_LANGFUSE_API_PROMPT_ID): langfuse_options.get(
                CONF_LANGFUSE_API_PROMPT_LABEL, "production"
            ),
        }
        client: Langfuse | None = None
        resource_key: str | None = None
        try:
            client, resource_key = await cls._acquire_sdk_client(
                hass,
                langfuse_options,
            )
            score_config_id = None
            if score_enabled:
                score_config_id = await cls._ensure_score_config(
                    hass,
                    resource_key,
                )
            return cls(
                hass,
                client,
                prompts,
                tracing_enabled,
                score_config_id,
                resource_key,
            )
        except asyncio.CancelledError:
            if resource_key is not None:
                try:
                    await cls._release_sdk_client(hass, resource_key)
                except Exception:
                    LOGGER.warning(
                        "Error rolling back cancelled Langfuse initialization",
                        exc_info=True,
                    )
            raise
        except Exception as err:
            if resource_key is not None:
                try:
                    await cls._release_sdk_client(hass, resource_key)
                except Exception:
                    LOGGER.warning(
                        "Error cleaning up partially initialized Langfuse client",
                        exc_info=True,
                    )
            LOGGER.error("Error initializing Langfuse client: %s", err)
            if isinstance(err, LangfuseInitError):
                raise
            raise LangfuseInitError("Failed to initialize Langfuse client") from err

    async def get_prompt(
        self, prompt_id: str, variables: dict[str, Any]
    ) -> tuple[Prompt, str]:
        """Get and compile a prompt from Langfuse."""
        with self.observe(
            name="cc_get_langfuse_prompt",
            input={"prompt_id": prompt_id},
        ) as observation:
            try:
                prompt_object = await self.hass.async_add_executor_job(
                    lambda: self._client.get_prompt(
                        prompt_id, label=self.prompts[prompt_id], type="chat"
                    )
                )
                compiled_prompt = prompt_object.compile(**variables)[0]["content"]
                if observation is not None:
                    observation.update(output={"prompt_id": prompt_id})
            except Exception as err:
                if observation is not None:
                    observation.update(level="ERROR", status_message=str(err))
                LOGGER.error("Error getting Langfuse prompt: %s", err)
                raise LangfusePromptError(
                    f"Failed to get Langfuse prompt: {err}"
                ) from err
        return prompt_object, compiled_prompt

    def observe(
        self,
        *,
        name: str,
        as_type: Literal["span", "generation"] = "span",
        **kwargs: Any,
    ) -> AbstractContextManager[Any]:
        """Start an observation on this entry's client."""
        if not self.tracing_enabled:
            return nullcontext()
        return self._client.start_as_current_observation(
            name=name,
            as_type=as_type,
            **kwargs,
        )

    def propagate(self, **kwargs: Any) -> AbstractContextManager[Any]:
        """Propagate attributes within this entry's active trace."""
        if not self.tracing_enabled:
            return nullcontext()
        return propagate_attributes(**kwargs)

    def update_current_span(self, **kwargs: Any) -> None:
        """Update this entry's current span without affecting the conversation."""
        if not self.tracing_enabled:
            return
        try:
            self._client.update_current_span(**kwargs)
        except Exception:
            LOGGER.debug("Failed to update Langfuse span", exc_info=True)

    def get_current_trace_id(self) -> str | None:
        """Return this entry's current trace ID."""
        if not self.tracing_enabled:
            return None
        return self._client.get_current_trace_id()

    async def score(self, score: str, device_id: str) -> None:
        """Score a conversation using Langfuse."""
        if not self.score_config_id:
            LOGGER.warning("Score config ID not set, skipping scoring")
            return

        try:
            # Get the latest trace that matches this device
            traces = await self.hass.async_add_executor_job(
                lambda: self._client.api.trace.list(
                    name="cc_async_process",
                    tags=f"device_id:{device_id}",
                    from_timestamp=(
                        datetime.now(tz=timezone.utc) - timedelta(minutes=10)
                    ),
                    limit=1,
                )
            )
            LOGGER.debug("Traces found for device %s: %s", device_id, traces.data)
            if not traces.data:
                LOGGER.warning("No traces found for device %s", device_id)
                return
            # Score the latest trace
            latest_trace = traces.data[0]
            LOGGER.debug("Scoring trace %s with score %s", latest_trace.id, score)

            await self.hass.async_add_executor_job(
                lambda: self._client.create_score(
                    name=LANGFUSE_SCORE_NAME,
                    value=score,
                    comment="Score based on Home Assistant Service Call",
                    trace_id=latest_trace.id,
                    config_id=self.score_config_id,
                )
            )
        except Exception as err:
            LOGGER.error("Error scoring conversation: %s", err)

    async def cleanup(self) -> None:
        """Clean up Langfuse client resources."""
        if self._released:
            return
        if self._cleanup_task is None:
            self._cleanup_task = self.hass.async_create_task(
                self._async_cleanup(),
                name=f"{DOMAIN} Langfuse cleanup",
                eager_start=False,
            )
        await asyncio.shield(self._cleanup_task)

    async def _async_cleanup(self) -> None:
        """Release this wrapper's resource exactly once."""
        try:
            if self._managed_resource:
                if self._resource_key is not None:
                    await self._release_sdk_client(
                        self.hass,
                        self._resource_key,
                    )
            elif self._client is not None:
                await self.hass.async_add_executor_job(self._client.shutdown)
        except Exception as err:
            LOGGER.warning("Error cleaning up Langfuse client: %s", err)
        finally:
            self._released = True
