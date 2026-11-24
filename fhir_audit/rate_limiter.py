"""
Async Sliding-Window Rate Limiter & Smart 429 Retry Handler for LLM APIs.

Protects Free-Tier endpoints (e.g., Gemini 15 RPM) from HTTP 429 RESOURCE_EXHAUSTED
errors during parallel multi-agent execution and large batch benchmarks.
Measures pure API inference latency separately from rate-limit wait time so
benchmark latency metrics remain scientifically accurate.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

from openai import AsyncOpenAI


_RETRY_SEC_RE = re.compile(r"(?:retry in|retryDelay['\"]?\s*[:=]\s*['\"]?)\s*(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)


class AsyncSlidingWindowLimiter:
    """Per-model async sliding-window RPM limiter with gentle request pacing."""

    def __init__(self) -> None:
        self._locks: Dict[str, asyncio.Lock] = {}
        self._windows: Dict[str, Deque[float]] = {}
        self._last_call: Dict[str, float] = {}

    def _get_limits(self, model_name: str) -> Tuple[int, float]:
        low = model_name.lower()
        if "gemini" in low:
            # Gemini Free Tier: 15 RPM; cap safely at 12 RPM by default (override via GEMINI_MAX_RPM)
            max_rpm = int(os.getenv("GEMINI_MAX_RPM", "12"))
            min_gap = float(os.getenv("GEMINI_MIN_GAP_SEC", "1.2"))
            return max(1, max_rpm), max(0.0, min_gap)
        max_rpm = int(os.getenv("LLM_MAX_RPM", "60"))
        min_gap = float(os.getenv("LLM_MIN_GAP_SEC", "0.1"))
        return max(1, max_rpm), max(0.0, min_gap)

    async def acquire(self, model_name: str) -> float:
        """Wait until a request slot is available. Returns seconds waited."""
        if model_name not in self._locks:
            self._locks[model_name] = asyncio.Lock()
            self._windows[model_name] = deque()
            self._last_call[model_name] = 0.0

        max_rpm, min_gap = self._get_limits(model_name)
        window_sec = 62.0
        waited_total = 0.0

        async with self._locks[model_name]:
            win = self._windows[model_name]
            while True:
                now = time.monotonic()
                while win and (now - win[0]) >= window_sec:
                    win.popleft()

                wait_for = 0.0
                if len(win) >= max_rpm:
                    wait_for = max(wait_for, window_sec - (now - win[0]) + 0.5)
                since_last = now - self._last_call[model_name]
                if since_last < min_gap:
                    wait_for = max(wait_for, min_gap - since_last)

                if wait_for <= 0:
                    now_ok = time.monotonic()
                    win.append(now_ok)
                    self._last_call[model_name] = now_ok
                    return waited_total

                if wait_for >= 3.0:
                    print(f"[RateLimiter] {model_name}: đạt ngưỡng {max_rpm} RPM, chờ {wait_for:.1f}s để tránh lỗi 429...")
                await asyncio.sleep(wait_for)
                waited_total += wait_for


GLOBAL_LIMITER = AsyncSlidingWindowLimiter()


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate" in msg or "quota" in msg or "resource_exhausted" in msg


def _is_server_overloaded_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "503" in msg or "unavailable" in msg or "high demand" in msg or "overloaded" in msg


def _is_non_retryable_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "404" in msg or "not_found" in msg or "401" in msg or "403" in msg or "api_key_invalid" in msg


def _is_json_mode_unsupported(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("400" in msg or "invalid" in msg or "unsupported" in msg) and "response_format" in msg


def _parse_retry_delay(exc: Exception, attempt: int) -> float:
    m = _RETRY_SEC_RE.search(str(exc))
    if m:
        return min(90.0, float(m.group(1)) + 2.0)
    return min(65.0, 10.0 * (attempt + 1))


async def call_llm_with_rate_limit(
    client: AsyncOpenAI,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.15,
    max_retries: int = 4,
) -> Tuple[str, int, Dict[str, int]]:
    """
    Call OpenAI-compatible chat completions API with:
      1. Sliding-window RPM pacing (prevents 429 bursts)
      2. Smart 429 retry honoring server retryDelay (never double-fires on 429)
      3. Pure API inference latency measurement (excludes quota wait time)
    """
    last_err: Optional[Exception] = None
    use_json_format = True

    for attempt in range(max_retries + 1):
        await GLOBAL_LIMITER.acquire(model_name)
        t0 = time.perf_counter()
        try:
            kwargs: Dict[str, Any] = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
            }
            if use_json_format:
                kwargs["response_format"] = {"type": "json_object"}

            resp = await client.chat.completions.create(**kwargs)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            content = resp.choices[0].message.content or ""
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(resp.usage, "total_tokens", 0) or 0,
            }
            return content, latency_ms, usage
        except Exception as exc:
            last_err = exc
            if use_json_format and _is_json_mode_unsupported(exc):
                use_json_format = False
                continue

            if _is_non_retryable_error(exc):
                break

            if attempt < max_retries:
                if _is_rate_limit_error(exc):
                    delay = _parse_retry_delay(exc, attempt)
                    print(
                        f"[RateLimiter] Dính lỗi 429 trên {model_name} (lần {attempt + 1}/{max_retries + 1}). "
                        f"Tự động chờ {delay:.1f}s rồi thử lại..."
                    )
                    await asyncio.sleep(delay)
                elif _is_server_overloaded_error(exc):
                    delay = min(30.0, 5.0 * (attempt + 1))
                    print(
                        f"[RateLimiter] Máy chủ {model_name} đang quá tải 503 (lần {attempt + 1}/{max_retries + 1}). "
                        f"Tự động chờ {delay:.1f}s rồi thử lại..."
                    )
                    await asyncio.sleep(delay)
                else:
                    await asyncio.sleep(2.0 * (attempt + 1))

    raise RuntimeError(f"LLM call failed after {max_retries + 1} attempts: {last_err}")
