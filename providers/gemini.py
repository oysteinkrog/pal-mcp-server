"""Gemini model provider implementation."""

import base64
import logging
import time
from typing import TYPE_CHECKING, ClassVar, Optional

import httpx

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

from google import genai
from google.genai import types

from utils.env import get_env
from utils.image_utils import validate_image

from .base import ModelProvider
from .registries.gemini import GeminiModelRegistry
from .registry_provider_mixin import RegistryBackedProviderMixin
from .shared import ModelCapabilities, ModelResponse, ProviderType

logger = logging.getLogger(__name__)


class GeminiModelProvider(RegistryBackedProviderMixin, ModelProvider):
    """First-party Gemini integration built on the official Google SDK.

    The provider advertises detailed thinking-mode budgets, handles optional
    custom endpoints, and performs image pre-processing before forwarding a
    request to the Gemini APIs.
    """

    REGISTRY_CLASS = GeminiModelRegistry
    MODEL_CAPABILITIES: ClassVar[dict[str, ModelCapabilities]] = {}

    # Thinking mode configurations - percentages of model's max_thinking_tokens
    # These percentages work across all models that support thinking
    THINKING_BUDGETS = {
        "minimal": 0.005,  # 0.5% of max - minimal thinking for fast responses
        "low": 0.08,  # 8% of max - light reasoning tasks
        "medium": 0.33,  # 33% of max - balanced reasoning (default)
        "high": 0.67,  # 67% of max - complex analysis
        "max": 1.0,  # 100% of max - full thinking budget
    }

    # Gemini 4 ignores thinking_budget and takes a thinking level instead. A non-streaming call
    # gets 32K output tokens including thinking, and on large inputs HIGH spends all of them on
    # thinking and returns no text, so high and max map to MEDIUM.
    THINKING_LEVELS = {
        "minimal": "LOW",
        "low": "LOW",
        "medium": "MEDIUM",
        "high": "MEDIUM",
        "max": "MEDIUM",
    }

    # Gemini 4 replies longer than one 32K call are fetched with continuation tokens. These caps
    # bound the total reply and the number of requests one call can make.
    GEMINI4_MAX_OUTPUT_TOKENS = 131072
    GEMINI4_MAX_CONTINUATIONS = 8
    GEMINI4_HOP_RETRY_DELAYS = (5, 15, 30)

    def __init__(self, api_key: str, **kwargs):
        """Initialize Gemini provider with API key and optional base URL."""
        self._ensure_registry()
        super().__init__(api_key, **kwargs)
        self._client = None
        self._token_counters = {}  # Cache for token counting
        self._base_url = kwargs.get("base_url", None)  # Optional custom endpoint
        self._timeout_override = self._resolve_http_timeout()
        self._invalidate_capability_cache()

    # ------------------------------------------------------------------
    # Capability surface
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Client access
    # ------------------------------------------------------------------

    @property
    def client(self):
        """Lazy initialization of Gemini client."""
        if self._client is None:
            http_options_kwargs: dict[str, object] = {}
            if self._base_url:
                http_options_kwargs["base_url"] = self._base_url
            if self._timeout_override is not None:
                http_options_kwargs["timeout"] = self._timeout_override

            # GEMINI_VERTEX_PROJECT sends calls through Vertex AI, billed to the project's
            # Cloud billing account, using Application Default Credentials. The API key is
            # then only used to register the provider.
            vertex_project = get_env("GEMINI_VERTEX_PROJECT")
            client_kwargs: dict[str, object] = {}
            if vertex_project:
                client_kwargs.update(
                    vertexai=True,
                    project=vertex_project,
                    location=get_env("GEMINI_VERTEX_LOCATION") or "global",
                )
            else:
                client_kwargs["api_key"] = self.api_key

            if http_options_kwargs:
                http_options = types.HttpOptions(**http_options_kwargs)
                logger.debug(
                    "Initializing Gemini client with options: base_url=%s timeout=%s",
                    http_options_kwargs.get("base_url"),
                    http_options_kwargs.get("timeout"),
                )
                client_kwargs["http_options"] = http_options
            if vertex_project:
                logger.info("Gemini client uses Vertex AI project %s", vertex_project)
            self._client = genai.Client(**client_kwargs)
        return self._client

    def _resolve_http_timeout(self) -> Optional[float]:
        """Compute timeout override from shared custom timeout environment variables."""

        timeouts: list[float] = []
        for env_var in [
            "CUSTOM_CONNECT_TIMEOUT",
            "CUSTOM_READ_TIMEOUT",
            "CUSTOM_WRITE_TIMEOUT",
            "CUSTOM_POOL_TIMEOUT",
        ]:
            raw_value = get_env(env_var)
            if raw_value:
                try:
                    timeouts.append(float(raw_value))
                except (TypeError, ValueError):
                    logger.warning("Invalid %s value '%s'; ignoring.", env_var, raw_value)

        if timeouts:
            # Use the largest timeout to best approximate long-running requests
            resolved = max(timeouts)
            logger.debug("Using custom Gemini HTTP timeout: %ss", resolved)
            return resolved

        return None

    # ------------------------------------------------------------------
    # Request execution
    # ------------------------------------------------------------------

    def generate_content(
        self,
        prompt: str,
        model_name: str,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_output_tokens: Optional[int] = None,
        thinking_mode: str = "medium",
        images: Optional[list[str]] = None,
        **kwargs,
    ) -> ModelResponse:
        """
        Generate content using Gemini model.

        Args:
            prompt: The main user prompt/query to send to the model
            model_name: Canonical model name or its alias (e.g., "gemini-2.5-pro", "flash", "pro")
            system_prompt: Optional system instructions to prepend to the prompt for context/behavior
            temperature: Controls randomness in generation (0.0=deterministic, 1.0=creative), default 0.3
            max_output_tokens: Optional maximum number of tokens to generate in the response
            thinking_mode: Thinking budget level for models that support it ("minimal", "low", "medium", "high", "max"), default "medium"
            images: Optional list of image paths or data URLs to include with the prompt (for vision models)
            **kwargs: Additional keyword arguments (reserved for future use)

        Returns:
            ModelResponse: Contains the generated content, token usage stats, model metadata, and safety information
        """
        # Validate parameters and fetch capabilities
        self.validate_parameters(model_name, temperature)
        capabilities = self.get_capabilities(model_name)
        capability_map = self.get_all_model_capabilities()

        resolved_model_name = self._resolve_model_name(model_name)

        # Prepare content parts (text and potentially images)
        parts = []

        # Add system and user prompts as text
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        parts.append({"text": full_prompt})

        # Add images if provided and model supports vision
        if images and capabilities.supports_images:
            for image_path in images:
                try:
                    image_part = self._process_image(image_path)
                    if image_part:
                        parts.append(image_part)
                except Exception as e:
                    logger.warning(f"Failed to process image {image_path}: {e}")
                    # Continue with other images and text
                    continue
        elif images and not capabilities.supports_images:
            logger.warning(f"Model {resolved_model_name} does not support images, ignoring {len(images)} image(s)")

        # Create contents structure
        contents = [{"parts": parts}]

        effective_thinking_mode = thinking_mode

        # Prepare generation config
        generation_config = types.GenerateContentConfig(
            temperature=temperature,
            candidate_count=1,
        )

        # Add max output tokens if specified
        if max_output_tokens:
            generation_config.max_output_tokens = max_output_tokens

        # Add thinking configuration for models that support it
        if capabilities.supports_extended_thinking and effective_thinking_mode in self.THINKING_BUDGETS:
            # Get model's max thinking tokens and calculate actual budget
            model_config = capability_map.get(resolved_model_name)
            if resolved_model_name.startswith("gemini-4"):
                generation_config.thinking_config = types.ThinkingConfig(
                    thinking_level=self.THINKING_LEVELS[effective_thinking_mode]
                )
            elif model_config and model_config.max_thinking_tokens > 0:
                max_thinking_tokens = model_config.max_thinking_tokens
                actual_thinking_budget = int(max_thinking_tokens * self.THINKING_BUDGETS[effective_thinking_mode])
                generation_config.thinking_config = types.ThinkingConfig(thinking_budget=actual_thinking_budget)

        # Retry logic with progressive delays
        max_retries = 5  # Total of 5 attempts
        # Progressive delays. A Vertex 429 (tokens per minute) needs tens of seconds.
        retry_delays = [2, 8, 20, 45, 60]
        attempt_counter = {"value": 0}

        def _attempt() -> ModelResponse:
            attempt_counter["value"] += 1
            if resolved_model_name.startswith("gemini-4"):
                return self._generate_with_continuation(
                    resolved_model_name, parts, generation_config, effective_thinking_mode
                )
            response = self.client.models.generate_content(
                model=resolved_model_name,
                contents=contents,
                config=generation_config,
            )

            usage = self._extract_usage(response)

            finish_reason_str = "UNKNOWN"
            is_blocked_by_safety = False
            safety_feedback_details = None

            if response.candidates:
                candidate = response.candidates[0]

                try:
                    finish_reason_enum = candidate.finish_reason
                    if finish_reason_enum:
                        try:
                            finish_reason_str = finish_reason_enum.name
                        except AttributeError:
                            finish_reason_str = str(finish_reason_enum)
                    else:
                        finish_reason_str = "STOP"
                except AttributeError:
                    finish_reason_str = "STOP"

                if not response.text:
                    try:
                        safety_ratings = candidate.safety_ratings
                        if safety_ratings:
                            for rating in safety_ratings:
                                try:
                                    if rating.blocked:
                                        is_blocked_by_safety = True
                                        category_name = "UNKNOWN"
                                        probability_name = "UNKNOWN"

                                        try:
                                            category_name = rating.category.name
                                        except (AttributeError, TypeError):
                                            pass

                                        try:
                                            probability_name = rating.probability.name
                                        except (AttributeError, TypeError):
                                            pass

                                        safety_feedback_details = (
                                            f"Category: {category_name}, Probability: {probability_name}"
                                        )
                                        break
                                except (AttributeError, TypeError):
                                    continue
                    except (AttributeError, TypeError):
                        pass

            elif response.candidates is not None and len(response.candidates) == 0:
                is_blocked_by_safety = True
                finish_reason_str = "SAFETY"
                safety_feedback_details = "Prompt blocked, reason unavailable"

                try:
                    prompt_feedback = response.prompt_feedback
                    if prompt_feedback and prompt_feedback.block_reason:
                        try:
                            block_reason_name = prompt_feedback.block_reason.name
                        except AttributeError:
                            block_reason_name = str(prompt_feedback.block_reason)
                        safety_feedback_details = f"Prompt blocked, reason: {block_reason_name}"
                except (AttributeError, TypeError):
                    pass

            return ModelResponse(
                content=response.text,
                usage=usage,
                model_name=resolved_model_name,
                friendly_name="Gemini",
                provider=ProviderType.GOOGLE,
                metadata={
                    "thinking_mode": effective_thinking_mode if capabilities.supports_extended_thinking else None,
                    "finish_reason": finish_reason_str,
                    "is_blocked_by_safety": is_blocked_by_safety,
                    "safety_feedback": safety_feedback_details,
                },
            )

        try:
            return self._run_with_retries(
                operation=_attempt,
                max_attempts=max_retries,
                delays=retry_delays,
                log_prefix=f"Gemini API ({resolved_model_name})",
            )
        except Exception as exc:
            attempts = max(attempt_counter["value"], 1)
            error_msg = (
                f"Gemini API error for model {resolved_model_name} after {attempts} attempt"
                f"{'s' if attempts > 1 else ''}: {exc}"
            )
            raise RuntimeError(error_msg) from exc

    def _generate_with_continuation(
        self,
        model_name: str,
        parts: list[dict],
        generation_config: types.GenerateContentConfig,
        thinking_mode: str,
    ) -> ModelResponse:
        """Call a Gemini 4 model over REST and follow continuation tokens.

        One non-streaming call returns at most 32K tokens, thinking included. When a reply needs
        more, the API ends it with finish reason CONTINUATION and a continuationToken, and the same
        request is sent again with that token. google-genai 1.x drops the token, so this path does
        not use the SDK.
        """
        rest_parts = []
        for part in parts:
            if "inline_data" in part:
                inline = part["inline_data"]
                rest_parts.append({"inlineData": {"mimeType": inline["mime_type"], "data": inline["data"]}})
            else:
                rest_parts.append({"text": part["text"]})

        config: dict = {
            "candidateCount": 1,
            "maxOutputTokens": max(generation_config.max_output_tokens or 0, self.GEMINI4_MAX_OUTPUT_TOKENS),
        }
        if generation_config.temperature is not None:
            config["temperature"] = generation_config.temperature
        if generation_config.thinking_config and generation_config.thinking_config.thinking_level:
            config["thinkingConfig"] = {"thinkingLevel": generation_config.thinking_config.thinking_level.value}
        request = {"contents": [{"role": "user", "parts": rest_parts}], "generationConfig": config}

        url = self._rest_generate_url(model_name)
        timeout = self._timeout_override if self._timeout_override is not None else 1500.0

        texts: list[str] = []
        input_tokens = output_tokens = 0
        finish_reason = "UNKNOWN"
        token = None
        hops = 0
        with httpx.Client(timeout=timeout) as http:
            for hop in range(1, self.GEMINI4_MAX_CONTINUATIONS + 1):
                hops = hop
                body = dict(request, continuationToken=token) if token else request
                data = self._post_hop(http, url, body, is_continuation=token is not None)

                usage = data.get("usageMetadata", {})
                input_tokens = input_tokens or usage.get("promptTokenCount", 0)
                output_tokens += usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)

                candidates = data.get("candidates") or []
                if not candidates:
                    block = (data.get("promptFeedback") or {}).get("blockReason", "unknown")
                    raise ValueError(f"{model_name} returned no candidates (prompt blocked: {block})")
                candidate = candidates[0]
                finish_reason = candidate.get("finishReason", "STOP")
                content_parts = (candidate.get("content") or {}).get("parts", [])
                texts.append("".join(p.get("text", "") for p in content_parts if not p.get("thought")))

                token = candidate.get("continuationToken") or data.get("continuationToken")
                if finish_reason != "CONTINUATION" or not token:
                    break

        text = "".join(texts)
        if not text:
            raise ValueError(
                f"{model_name} returned no text after {hops} request(s) (finish reason {finish_reason}). "
                "Use a lower thinking_mode or a smaller input."
            )
        if finish_reason == "CONTINUATION":
            logger.warning("%s still wanted to continue after %d requests; reply is cut short", model_name, hops)

        return ModelResponse(
            content=text,
            usage={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            model_name=model_name,
            friendly_name="Gemini",
            provider=ProviderType.GOOGLE,
            metadata={
                "thinking_mode": thinking_mode,
                "finish_reason": finish_reason,
                "is_blocked_by_safety": finish_reason in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST"),
                "safety_feedback": None,
                "continuation_requests": hops,
            },
        )

    def _post_hop(self, http: httpx.Client, url: str, body: dict, is_continuation: bool) -> dict:
        """Send one request. After the first hop, retry busy errors here so the token is kept.

        A failure on the first hop is left to the outer retry loop, which starts over anyway.
        """
        delays = self.GEMINI4_HOP_RETRY_DELAYS if is_continuation else ()
        for attempt in range(len(delays) + 1):
            response = http.post(url, json=body, headers=self._rest_auth_headers())
            if response.status_code == 200:
                return response.json()
            if response.status_code not in (429, 500, 502, 503, 504) or attempt == len(delays):
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
            logger.warning(
                "Gemini continuation request got HTTP %s; trying again in %ss", response.status_code, delays[attempt]
            )
            time.sleep(delays[attempt])
        raise AssertionError("unreachable")

    def _rest_generate_url(self, model_name: str) -> str:
        """generateContent URL for the REST path: Vertex AI when GEMINI_VERTEX_PROJECT is set."""
        vertex_project = get_env("GEMINI_VERTEX_PROJECT")
        if vertex_project:
            location = get_env("GEMINI_VERTEX_LOCATION") or "global"
            host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
            return (
                f"https://{host}/v1beta1/projects/{vertex_project}/locations/{location}"
                f"/publishers/google/models/{model_name}:generateContent"
            )
        base_url = (self._base_url or "https://generativelanguage.googleapis.com").rstrip("/")
        return f"{base_url}/v1beta/models/{model_name}:generateContent"

    def _rest_auth_headers(self) -> dict[str, str]:
        """Vertex AI uses an Application Default Credentials token; AI Studio uses the API key."""
        if not get_env("GEMINI_VERTEX_PROJECT"):
            return {"x-goog-api-key": self.api_key}
        import google.auth
        import google.auth.transport.requests

        if getattr(self, "_adc_credentials", None) is None:
            self._adc_credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        if not self._adc_credentials.valid:
            self._adc_credentials.refresh(google.auth.transport.requests.Request())
        return {"Authorization": f"Bearer {self._adc_credentials.token}"}

    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.GOOGLE

    def _extract_usage(self, response) -> dict[str, int]:
        """Extract token usage from Gemini response."""
        usage = {}

        # Try to extract usage metadata from response
        # Note: The actual structure depends on the SDK version and response format
        try:
            metadata = response.usage_metadata
            if metadata:
                # Extract token counts with explicit None checks
                input_tokens = None
                output_tokens = None

                try:
                    value = metadata.prompt_token_count
                    if value is not None:
                        input_tokens = value
                        usage["input_tokens"] = value
                except (AttributeError, TypeError):
                    pass

                try:
                    value = metadata.candidates_token_count
                    if value is not None:
                        output_tokens = value
                        usage["output_tokens"] = value
                except (AttributeError, TypeError):
                    pass

                # Calculate total only if both values are available and valid
                if input_tokens is not None and output_tokens is not None:
                    usage["total_tokens"] = input_tokens + output_tokens
        except (AttributeError, TypeError):
            # response doesn't have usage_metadata
            pass

        return usage

    def _is_error_retryable(self, error: Exception) -> bool:
        """Determine if an error should be retried based on structured error codes.

        Uses Gemini API error structure instead of text pattern matching for reliability.

        Args:
            error: Exception from Gemini API call

        Returns:
            True if error should be retried, False otherwise
        """
        error_str = str(error).lower()

        # Check for 429 errors first - these need special handling
        if "429" in error_str or "quota" in error_str or "resource_exhausted" in error_str:
            # For Gemini, check for specific non-retryable error indicators
            # These typically indicate permanent failures or quota/size limits
            non_retryable_indicators = [
                "context length",
                "token limit",
                "request too large",
                "invalid request",
                "per day",
                "perday",
                "quota_exceeded",
            ]
            # A quota that refills within a minute is throttling, not a hard stop.
            if "quota exceeded" in error_str and not any(
                s in error_str for s in ("per minute", "per_minute", "perminute", "try again later")
            ):
                logger.debug(f"Non-retryable Gemini quota error: {error_str[:200]}...")
                return False

            # Also check if this is a structured error from Gemini SDK
            try:
                # Try to access error details if available
                error_details = None
                try:
                    error_details = error.details
                except AttributeError:
                    try:
                        error_details = error.reason
                    except AttributeError:
                        pass

                if error_details:
                    error_details_str = str(error_details).lower()
                    # Check for non-retryable error codes/reasons
                    if any(indicator in error_details_str for indicator in non_retryable_indicators):
                        logger.debug(f"Non-retryable Gemini error: {error_details}")
                        return False
            except Exception:
                pass

            # Check main error string for non-retryable patterns
            if any(indicator in error_str for indicator in non_retryable_indicators):
                logger.debug(f"Non-retryable Gemini error based on message: {error_str[:200]}...")
                return False

            # If it's a 429/quota error but doesn't match non-retryable patterns, it might be retryable rate limiting
            logger.debug(f"Retryable Gemini rate limiting error: {error_str[:100]}...")
            return True

        # For non-429 errors, check if they're retryable
        retryable_indicators = [
            "timeout",
            "connection",
            "network",
            "temporary",
            "unavailable",
            "retry",
            "internal error",
            "408",  # Request timeout
            "500",  # Internal server error
            "502",  # Bad gateway
            "503",  # Service unavailable
            "504",  # Gateway timeout
            "ssl",  # SSL errors
            "handshake",  # Handshake failures
        ]

        return any(indicator in error_str for indicator in retryable_indicators)

    def _process_image(self, image_path: str) -> Optional[dict]:
        """Process an image for Gemini API."""
        try:
            # Use base class validation
            image_bytes, mime_type = validate_image(image_path)

            # For data URLs, extract the base64 data directly
            if image_path.startswith("data:"):
                # Extract base64 data from data URL
                _, data = image_path.split(",", 1)
                return {"inline_data": {"mime_type": mime_type, "data": data}}
            else:
                # For file paths, encode the bytes
                image_data = base64.b64encode(image_bytes).decode()
                return {"inline_data": {"mime_type": mime_type, "data": image_data}}

        except ValueError as e:
            logger.warning(str(e))
            return None
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None

    def get_preferred_model(self, category: "ToolModelCategory", allowed_models: list[str]) -> Optional[str]:
        """Get Gemini's preferred model for a given category from allowed models.

        Args:
            category: The tool category requiring a model
            allowed_models: Pre-filtered list of models allowed by restrictions

        Returns:
            Preferred model name or None
        """
        from tools.models import ToolModelCategory

        if not allowed_models:
            return None

        capability_map = self.get_all_model_capabilities()

        # Helper to find best model from candidates
        def find_best(candidates: list[str]) -> Optional[str]:
            """Return best model from candidates (sorted for consistency)."""
            return sorted(candidates, reverse=True)[0] if candidates else None

        if category == ToolModelCategory.EXTENDED_REASONING:
            # For extended reasoning, prefer models with thinking support
            # First try Pro models that support thinking
            pro_thinking = [
                m
                for m in allowed_models
                if "pro" in m and m in capability_map and capability_map[m].supports_extended_thinking
            ]
            if pro_thinking:
                return find_best(pro_thinking)

            # Then any model that supports thinking
            any_thinking = [
                m for m in allowed_models if m in capability_map and capability_map[m].supports_extended_thinking
            ]
            if any_thinking:
                return find_best(any_thinking)

            # Finally, just prefer Pro models even without thinking
            pro_models = [m for m in allowed_models if "pro" in m]
            if pro_models:
                return find_best(pro_models)

        elif category == ToolModelCategory.FAST_RESPONSE:
            # Prefer Flash models for speed
            flash_models = [m for m in allowed_models if "flash" in m]
            if flash_models:
                return find_best(flash_models)

        # Default for BALANCED or as fallback
        # Prefer Flash for balanced use, then Pro, then anything
        flash_models = [m for m in allowed_models if "flash" in m]
        if flash_models:
            return find_best(flash_models)

        pro_models = [m for m in allowed_models if "pro" in m]
        if pro_models:
            return find_best(pro_models)

        # Ultimate fallback to best available model
        return find_best(allowed_models)


# Load registry data at import time for registry consumers
GeminiModelProvider._ensure_registry()
