"""FastAPI service wrapping the Ollama LLM server."""

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b"

app = FastAPI(
    title="LLM Service",
    description="FastAPI wrapper around Ollama serving Qwen2.5:0.5B",
)


class ChatRequest(BaseModel):
    prompt: str
    system_prompt: str | None = None


class ChatResponse(BaseModel):
    response: str

def build_ollama_payload(prompt: str, system_prompt: str | None) -> dict:
    """
    Формирует тело запроса для Ollama API (/api/generate).

    Параметры:
        prompt: пользовательский текст запроса к модели.
        system_prompt: системная инструкция, задающая поведение модели.
            Если не передана — поле system в payload не включается.

    Возвращает:
        Словарь, готовый к сериализации в JSON для отправки в Ollama.
    """
    payload: dict = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    }
    if system_prompt:
        payload["system"] = system_prompt
    return payload


def call_ollama(payload: dict) -> str:
    """
    Отправляет подготовленный payload на сервер Ollama и возвращает сгенерированный текст.

    При недоступности сервера бросает HTTPException 503.
    При HTTP-ошибке со стороны Ollama бросает HTTPException 502.

    Параметры:
        payload: словарь с полями model, prompt и опционально system.

    Возвращает:
        Строку с ответом модели из поля response в JSON-ответе Ollama.
    """
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise HTTPException(status_code=503, detail="Ollama server is not reachable") from exc
    except requests.exceptions.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Ollama error: {exc}") from exc

    data = resp.json()
    return data.get("response", "")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Основной эндпоинт сервиса: принимает текстовый запрос, проксирует его в Ollama
    и возвращает ответ модели.

    Параметры:
        request: тело запроса с полями prompt и опциональным system_prompt.

    Возвращает:
        ChatResponse с полем response — текстом ответа языковой модели.
    """
    payload = build_ollama_payload(request.prompt, request.system_prompt)
    response_text = call_ollama(payload)
    return ChatResponse(response=response_text)


@app.get("/health")
def health() -> dict:
    """
    Проверка работоспособности сервиса (liveness probe).

    Возвращает:
        Словарь {"status": "ok"} если сервис запущен и принимает запросы.
    """
    return {"status": "ok"}
