import json
import logging
import asyncio
from typing import Type, Any, Optional
from pydantic import BaseModel, ValidationError
from langchain_openai import ChatOpenAI

from config.settings import settings as config

logger = logging.getLogger(__name__)


class LLMFactory:
    """Centralized LLM factory. All nodes must use this to ensure vendor agility."""

    @staticmethod
    def get_llm(temperature: float = 0.0, fast: bool = False, model: Optional[str] = None) -> ChatOpenAI:
        """
        Build a chat client.

        fast=True selects FAST_MODEL — for classification and query condensation, where a
        smaller model is both cheaper and materially quicker and the output is a small fixed
        schema rather than customer-facing prose.

        Three deliberate settings:
        - `timeout` bounds a SINGLE request. Without it, SDK-level retries on a stalled
          provider consume the whole node budget and the fallback strategies downstream never
          run at all.
        - `max_tokens` bounds the reply. These nodes emit a small JSON object; advertising the
          model's full 65k output window makes the provider pre-authorise a cost far above what
          the call will actually incur, which fails outright on a modest balance.
        - `reasoning.max_tokens: 0` disables extended thinking. Every call here is constrained
          to a Pydantic schema, so thinking tokens buy nothing and cost seconds. Providers that
          don't recognise the field ignore it.
        """
        api_base = config.OPENAI_API_BASE.strip() if config.OPENAI_API_BASE and config.OPENAI_API_BASE.strip() else None
        chosen = model or (config.FAST_MODEL if fast else config.DEFAULT_MODEL)

        return ChatOpenAI(
            model=chosen,
            temperature=temperature,
            max_retries=config.MAX_RETRIES,
            timeout=config.LLM_REQUEST_TIMEOUT_SECONDS,
            max_tokens=config.LLM_MAX_OUTPUT_TOKENS,
            base_url=api_base,
            api_key=config.OPENAI_API_KEY,
            extra_body={"reasoning": {"max_tokens": 0}},
        )


def model_fallback_chain(primary: str) -> list:
    """
    Model ids to try, in order, for one logical call: the primary, then each configured
    fallback that isn't already the primary. Used when a model fails OUTRIGHT (provider
    outage, rate limit, retired id) — not for schema failures, which the strategy chain in
    execute_vendor_agnostic_node handles on the same model.
    """
    chain = [primary]
    for candidate in (config.MODEL_FALLBACKS or "").split(","):
        candidate = candidate.strip()
        if candidate and candidate not in chain:
            chain.append(candidate)
    return chain


def extract_and_parse_json(raw_text: str) -> dict:
    """Bulletproof JSON extraction using strict boundary isolation."""
    try:
        start_idx = raw_text.find('{')
        end_idx = raw_text.rfind('}')

        if start_idx == -1 or end_idx == -1:
            raise ValueError("No JSON object boundaries found in text.")

        clean_json_string = raw_text[start_idx:end_idx + 1]
        return json.loads(clean_json_string)
    except Exception as e:
        logger.error(f"Failed strict JSON extraction: {e}. Raw Text: {raw_text}")
        raise ValueError("JSON Extraction Failed")


async def _attempt_structured(
    llm: Any,
    formatted_prompt: Any,
    schema_class: Type[BaseModel],
    user_identifier: str,
) -> BaseModel:
    """
    Three output strategies against ONE model, most reliable first. Raises if all three fail,
    so the caller can decide whether to try a different model.
    """
    try:
        # Strategy 1: Native Structured Output
        structured_llm = llm.with_structured_output(schema_class)
        return await structured_llm.ainvoke(formatted_prompt)
    except Exception as structured_err:
        logger.warning(f"[{user_identifier}] Native structured output failed. Falling back. Err: {structured_err}")

        # Strategy 2: provider JSON mode. Strategy 3: plain invoke + strict extraction.
        try:
            json_mode_llm = llm.bind(response_format={"type": "json_object"})
            response = await json_mode_llm.ainvoke(formatted_prompt)
        except Exception as json_mode_err:
            logger.warning(
                f"[{user_identifier}] json_object mode unavailable; using plain invoke. "
                f"Err: {json_mode_err}"
            )
            response = await llm.ainvoke(formatted_prompt)

        parsed_json = extract_and_parse_json(response.content)
        return schema_class(**parsed_json)


async def execute_vendor_agnostic_node(
    llm: Any,
    formatted_prompt: Any,
    schema_class: Type[BaseModel],
    user_identifier: str = "unknown_user"
) -> Optional[BaseModel]:
    """
    Run one schema-constrained LLM call, hardened in two dimensions: three output strategies
    per model (structured -> JSON mode -> extract), then the configured model fallback chain
    if a model fails outright. Returns None on total failure; callers treat None as a graceful
    degradation rather than raising, so a provider outage costs quality, not the turn.
    """
    primary = getattr(llm, "model_name", None) or getattr(llm, "model", None) or config.DEFAULT_MODEL
    chain = model_fallback_chain(str(primary))
    last_err: Optional[Exception] = None

    try:
        async with asyncio.timeout(config.LLM_TIMEOUT_SECONDS):
            for index, model_id in enumerate(chain):
                candidate = llm
                if index > 0:
                    logger.warning(f"[{user_identifier}] Retrying on fallback model '{model_id}'.")
                    try:
                        candidate = LLMFactory.get_llm(model=model_id)
                    except Exception as build_err:  # pragma: no cover - config-shaped failure
                        logger.error(f"[{user_identifier}] Could not build '{model_id}': {build_err}")
                        continue
                try:
                    return await _attempt_structured(
                        candidate, formatted_prompt, schema_class, user_identifier
                    )
                except ValidationError as ve:
                    # The model replied but broke the contract. Another model may not.
                    logger.error(f"[{user_identifier}] Schema Validation Error on '{model_id}': {ve}")
                    last_err = ve
                except Exception as e:
                    logger.error(f"[{user_identifier}] Model '{model_id}' failed: {repr(e)}")
                    last_err = e

            logger.error(f"[{user_identifier}] All models exhausted. Last error: {repr(last_err)}")
            return None

    except Exception as e:
        logger.error(f"[{user_identifier}] Node Execution Failure: {repr(e)}")
        return None