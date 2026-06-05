"""LLM client using OpenAI-compatible API (Ollama, LM Studio, vLLM, OpenAI, Groq, xAI, etc)."""

from __future__ import annotations
from typing import Generator, Optional

from openai import OpenAI

from .config import DocQConfig, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


class LLMClient:
    def __init__(self, cfg: DocQConfig):
        self.cfg = cfg
        self.client = OpenAI(
            base_url=cfg.llm_base_url,
            api_key=cfg.llm_api_key,
            timeout=120.0,
        )

    def is_available(self) -> bool:
        """Best-effort check whether the LLM server responds at all."""
        try:
            # Use a very short timeout for the health check
            # Some servers (pure Ollama without OpenAI compat shim) may not have /models
            # so we fall back to trying a tiny chat completion (non-streaming).
            try:
                self.client.models.list(timeout=3.0)
                return True
            except Exception:
                # Fallback: try a minimal completion (many local servers support this)
                self.client.chat.completions.create(
                    model=self.cfg.llm_model,
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=1,
                    timeout=5.0,
                    stream=False,
                )
                return True
        except Exception:
            return False

    def stream_answer(
        self,
        question: str,
        context: str,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> Generator[str, None, None]:
        """Yield answer tokens. Caller prints for perceived speed."""
        model = model or self.cfg.llm_model
        user_content = USER_PROMPT_TEMPLATE.format(context=context, question=question)

        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"\n[LLM error: {e}]\n"
            yield (
                "\nTip: Is Ollama (or your local LLM server) running?\n"
                "  - Try in another terminal:  ollama run llama3.2:3b   (then /bye)\n"
                "  - Or: ollama serve\n"
                "  - Change model / server with env vars: DOCQA_LLM_MODEL, DOCQA_LLM_BASE_URL\n"
                "  - You can still use  python -m docq search \"your question\"  for fast retrieval without an LLM.\n"
            )

    def answer(
        self,
        question: str,
        context: str,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """Non-streaming full answer (useful for API mode later)."""
        model = model or self.cfg.llm_model
        user_content = USER_PROMPT_TEMPLATE.format(context=context, question=question)
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            return f"[LLM error: {e}]"
