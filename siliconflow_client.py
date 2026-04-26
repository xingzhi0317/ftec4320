import os
import tomllib
from pathlib import Path

from openai import OpenAI


SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-V3"
SECRETS_PATH = Path(__file__).parent / ".streamlit" / "secrets.toml"


def get_api_key() -> str:
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if api_key:
        return api_key

    if SECRETS_PATH.exists():
        with SECRETS_PATH.open("rb") as secrets_file:
            api_key = tomllib.load(secrets_file).get("SILICONFLOW_API_KEY")
        if api_key:
            return api_key

    try:
        import streamlit as st

        api_key = st.secrets.get("SILICONFLOW_API_KEY")
    except Exception:
        api_key = None

    if not api_key:
        raise RuntimeError(
            "Missing SILICONFLOW_API_KEY. Set it in .streamlit/secrets.toml "
            "or as an environment variable."
        )

    return api_key


def create_client() -> OpenAI:
    return OpenAI(
        api_key=get_api_key(),
        base_url=SILICONFLOW_BASE_URL,
    )


def chat_deepseek(
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    response_format: dict[str, str] | None = None,
) -> str:
    client = create_client()
    request = dict(
        model=DEEPSEEK_MODEL,
        messages=messages,
        temperature=temperature,
    )
    if response_format:
        request["response_format"] = response_format

    response = client.chat.completions.create(**request)
    return response.choices[0].message.content or ""


def ask_deepseek(prompt: str) -> str:
    return chat_deepseek(
        messages=[
            {"role": "system", "content": "You are a concise trading assistant."},
            {"role": "user", "content": prompt},
        ],
    )


if __name__ == "__main__":
    print(ask_deepseek("用一句话确认连接成功。"))
