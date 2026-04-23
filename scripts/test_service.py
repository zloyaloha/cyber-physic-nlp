"""Script to test the LLM FastAPI service from outside the Docker container."""

import json
import requests

SERVICE_URL = "http://localhost:8000/chat"

TEST_CASES = [
    {
        "name": "Health ping",
        "prompt": "Say hello in one sentence.",
        "system_prompt": None,
    },
    {
        "name": "Simple question",
        "prompt": "What is the capital of France?",
        "system_prompt": None,
    },
    {
        "name": "Spam classification (obvious spam)",
        "prompt": "WINNER!! You have been selected to receive a £900 prize! Call 09061701461 now.",
        "system_prompt": (
            "You are an SMS spam classifier. Respond ONLY with JSON: "
            '{"reasoning": "<string>", "verdict": 0 or 1}'
        ),
    },
    {
        "name": "Spam classification (ham)",
        "prompt": "Hey, are you free for lunch tomorrow?",
        "system_prompt": (
            "You are an SMS spam classifier. Respond ONLY with JSON: "
            '{"reasoning": "<string>", "verdict": 0 or 1}'
        ),
    },
    {
        "name": "Translation task",
        "prompt": "Translate to Spanish: The weather is nice today.",
        "system_prompt": "You are a helpful translation assistant.",
    },
]


def send_chat_request(prompt: str, system_prompt: str | None) -> str:
    """
    Отправляет один запрос на FastAPI-сервис и возвращает текст ответа модели.

    Параметры:
        prompt: текст запроса к модели.
        system_prompt: системная инструкция или None.

    Возвращает:
        Строку с ответом модели или сообщение об ошибке.
    """
    payload = {"prompt": prompt}
    if system_prompt:
        payload["system_prompt"] = system_prompt

    try:
        resp = requests.post(SERVICE_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.exceptions.ConnectionError:
        return "ERROR: Could not connect to the service. Is the container running?"
    except requests.exceptions.HTTPError as exc:
        return f"ERROR: HTTP {exc.response.status_code} — {exc.response.text}"


def print_result(name: str, prompt: str, response: str) -> None:
    """
    Выводит результат одного тест-кейса в форматированном виде.

    Параметры:
        name: название теста.
        prompt: текст запроса (обрезается до 80 символов при выводе).
        response: ответ модели.
    """
    print(f"\n{'=' * 60}")
    print(f"Test: {name}")
    print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print(f"Response: {response}")


def run_all_tests() -> None:
    """Run all predefined test cases and print results."""
    print(f"Testing LLM service at {SERVICE_URL}")
    print(f"Running {len(TEST_CASES)} test cases...\n")

    for case in TEST_CASES:
        response = send_chat_request(case["prompt"], case["system_prompt"])
        print_result(case["name"], case["prompt"], response)

    print(f"\n{'=' * 60}")
    print("All tests completed.")


if __name__ == "__main__":
    run_all_tests()
