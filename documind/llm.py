"""Chat backends. Local mode never calls a remote model."""

from __future__ import annotations

from documind.config import Settings


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.provider = settings.llm_provider

    def complete(self, system: str, user: str) -> str:
        if self.provider == "local":
            raise RuntimeError("The local provider answers from retrieved text, not a chat model.")
        message = self.build_chat().invoke([("system", system), ("human", user)])
        content = message.content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content)

    def build_chat(self):
        if self.provider == "openai":
            if not self.settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is required for the openai provider.")
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=self.settings.openai_model,
                api_key=self.settings.openai_api_key,
                temperature=0,
            )
        if self.provider == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=self.settings.ollama_model,
                base_url=self.settings.ollama_base_url,
                temperature=0,
                validate_model_on_init=False,
            )
        raise RuntimeError(f"Unknown LLM provider: {self.provider}")
