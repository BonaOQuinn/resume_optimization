"""Claude through Amazon Bedrock, behind LangChain's ChatAnthropic interface.

Claude in Amazon Bedrock serves the same Messages API as the Claude API at
https://bedrock-mantle.{region}.api.aws/anthropic, so this subclass keeps all of
ChatAnthropic's message formatting, tool calling, and streaming. It swaps only
the transport (the official Anthropic SDK's AnthropicBedrockMantle client, which
signs requests with AWS credentials or a Bedrock API key) and the model ID.

The LangChain model keeps the bare Claude ID (for example claude-opus-5-5) so
ChatAnthropic's per-model rules and profiles still apply, and the Bedrock ID
(anthropic.claude-opus-5-5) is substituted only in the outgoing request.
"""

from functools import cached_property
from typing import Any

from anthropic import AnthropicBedrockMantle, AsyncAnthropicBedrockMantle
from langchain_anthropic import ChatAnthropic

BEDROCK_MODEL_PREFIX = "anthropic."


def bedrock_model_id(model: str) -> str:
    """Return the Bedrock model ID for a bare or already-prefixed Claude model ID."""
    return model if model.startswith(BEDROCK_MODEL_PREFIX) else BEDROCK_MODEL_PREFIX + model


def bare_model_id(model: str) -> str:
    """Return the Claude model ID without Bedrock's provider prefix."""
    return model.removeprefix(BEDROCK_MODEL_PREFIX)


class ChatAnthropicBedrock(ChatAnthropic):
    """ChatAnthropic that sends requests to Claude in Amazon Bedrock."""

    aws_region: str | None = None
    """AWS region, such as us-east-1. Defaults to AWS_REGION / AWS_DEFAULT_REGION."""

    aws_profile: str | None = None
    """Named profile from ~/.aws/credentials. Defaults to the AWS credential chain."""

    @property
    def _llm_type(self) -> str:
        # A non-direct type makes ChatAnthropic attach cache_control kwargs to
        # message blocks instead of sending the direct-API-only top-level field.
        return "anthropic-bedrock-chat"

    def _bedrock_client_params(self) -> dict[str, Any]:
        params = self._client_params
        options: dict[str, Any] = {
            "aws_region": self.aws_region,
            "aws_profile": self.aws_profile,
            "max_retries": params["max_retries"],
            "default_headers": params["default_headers"],
        }
        if "timeout" in params:
            options["timeout"] = params["timeout"]
        return options

    @cached_property
    def _client(self) -> AnthropicBedrockMantle:  # type: ignore[override]
        # Credentials: AWS_BEARER_TOKEN_BEDROCK (a Bedrock API key) if set,
        # otherwise the AWS credential chain (env keys, aws configure, SSO).
        return AnthropicBedrockMantle(**self._bedrock_client_params())

    @cached_property
    def _async_client(self) -> AsyncAnthropicBedrockMantle:  # type: ignore[override]
        return AsyncAnthropicBedrockMantle(**self._bedrock_client_params())

    def _get_request_payload(self, input_: Any, *, stop: list[str] | None = None, **kwargs: Any) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        payload["model"] = bedrock_model_id(payload["model"])
        return payload
