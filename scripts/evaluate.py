import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT / "prompts"

SERVICE_URL = "http://localhost:8000/chat"
DATA_PATH = ROOT / "data" / "spam.csv"
REPORTS_DIR = ROOT / "reports"
SAMPLE_PER_CLASS = 100
REQUEST_TIMEOUT = 120

TECHNIQUES = {
    name: (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")
    for name in (
        "zero_shot",
        "cot",
        "few_shot",
        "cot_few_shot",
        "improved_cot",
        "improved_cot_few_shot",
    )
}


def load_dataset(path: Path, sample_per_class: int) -> pd.DataFrame:
    """
    Загружает датасет SMS Spam Collection и формирует сбалансированную выборку.

    Из CSV берутся первые два столбца (метка и текст сообщения). Затем случайно
    отбирается sample_per_class спам-сообщений и столько же ham-сообщений,
    после чего выборка перемешивается.

    Параметры:
        path: путь к файлу spam.csv.
        sample_per_class: количество примеров каждого класса в итоговой выборке.

    Возвращает:
        DataFrame с колонками label, text, ground_truth (0 — ham, 1 — spam).
    """
    df = pd.read_csv(path, encoding="latin-1", usecols=[0, 1])
    df.columns = ["label", "text"]
    df["ground_truth"] = (df["label"] == "spam").astype(int)

    spam_df = df[df["ground_truth"] == 1].sample(n=sample_per_class, random_state=42)
    ham_df = df[df["ground_truth"] == 0].sample(n=sample_per_class, random_state=42)
    balanced = pd.concat([spam_df, ham_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Dataset loaded: {len(balanced)} samples ({sample_per_class} spam, {sample_per_class} ham)")
    return balanced


def query_llm(prompt: str, system_prompt: str) -> str:
    """
    Отправляет один запрос на FastAPI-сервис и возвращает сырой текст ответа модели.

    Запрос уходит на SERVICE_URL (POST /chat) с телом {"prompt": ..., "system_prompt": ...}.
    При любой сетевой ошибке или таймауте возвращает пустую строку и выводит предупреждение,
    чтобы не прерывать оценку всей выборки.

    Параметры:
        prompt: текст SMS-сообщения, которое нужно классифицировать.
        system_prompt: системная инструкция с техникой промптинга.

    Возвращает:
        Строку с ответом LLM, как правило в виде JSON {"reasoning": ..., "verdict": ...}.
    """
    payload = {"prompt": prompt, "system_prompt": system_prompt}
    try:
        resp = requests.post(SERVICE_URL, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as exc:
        print(f"  [WARNING] Request failed: {exc}")
        return ""


def parse_verdict(raw_response: str) -> int:
    """
    Извлекает вердикт классификации из сырого текста ответа LLM.

    Сначала ищет JSON-объект через регулярное выражение и парсит поле verdict.
    Если полный JSON не удалось разобрать — ищет паттерн "verdict": 0 или 1 напрямую.

    Параметры:
        raw_response: строка с ответом модели, ожидается формат
            {"reasoning": "...", "verdict": 0 или 1}.

    Возвращает:
        0 — ham, 1 — spam, -1 — не удалось распарсить ответ.
    """
    json_match = re.search(r"\{.*?\}", raw_response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            verdict = data.get("verdict")
            if verdict in (0, 1):
                return int(verdict)
        except json.JSONDecodeError:
            pass

    verdict_match = re.search(r'"verdict"\s*:\s*([01])', raw_response)
    if verdict_match:
        return int(verdict_match.group(1))

    return -1


def evaluate_technique(
    technique_name: str,
    system_prompt: str,
    dataset: pd.DataFrame,
) -> dict:
    """
    Прогоняет одну технику промптинга через всю выборку и вычисляет метрики качества.

    Для каждого сообщения из dataset вызывается query_llm, ответ парсится через parse_verdict.
    Если ответ не распарсился (verdict == -1), делается до трёх повторных попыток с упрощённым
    промптом, требующим вернуть только JSON. Итоговые предсказания и сырые ответы сохраняются
    в CSV-файл reports/details_<technique_name>.csv.

    Параметры:
        technique_name: название техники (используется в имени файла и выводе).
        system_prompt: текст системного промпта для данной техники.
        dataset: сбалансированная выборка из load_dataset.

    Возвращает:
        Словарь с полями technique, accuracy, precision, recall, f1,
        n_total, n_parsed, n_unparseable.
    """
    print(f"\nEvaluating technique: {technique_name} ({len(dataset)} samples)...")
    predictions = []
    raw_responses = []
    ground_truths = dataset["ground_truth"].tolist()

    for idx, row in dataset.iterrows():
        raw = query_llm(row["text"], system_prompt)
        verdict = parse_verdict(raw)

        retries = 0
        while verdict == -1 and retries < 3:
            raw = query_llm(
                row["text"],
                'Respond ONLY with valid JSON: {"reasoning": "<why>", "verdict": <0 or 1>} '
                "where verdict=1 is spam and verdict=0 is ham. No text outside the JSON.",
            )
            verdict = parse_verdict(raw)
            retries += 1

        predictions.append(verdict)
        raw_responses.append(raw)

        if (idx + 1) % 20 == 0:
            print(f"  Progress: {idx + 1}/{len(dataset)}")

        time.sleep(0.1)

    detail_df = dataset[["text", "ground_truth"]].copy()
    detail_df["raw_response"] = raw_responses
    detail_df["predicted"] = predictions
    n_unparseable = predictions.count(-1)
    clean_predictions = [
        p if p != -1 else 1 - gt
        for p, gt in zip(predictions, ground_truths)
    ]

    detail_df["correct"] = detail_df["ground_truth"] == detail_df["predicted"].apply(
        lambda v: v if v != -1 else 0
    )
    detail_path = REPORTS_DIR / f"details_{technique_name}.csv"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(detail_path, index=False)
    print(f"  Details saved to: {detail_path}")

    metrics = {
        "technique": technique_name,
        "accuracy": round(accuracy_score(ground_truths, clean_predictions), 4),
        "precision": round(precision_score(ground_truths, clean_predictions, zero_division=0), 4),
        "recall": round(recall_score(ground_truths, clean_predictions, zero_division=0), 4),
        "f1": round(f1_score(ground_truths, clean_predictions, zero_division=0), 4),
        "n_total": len(dataset),
        "n_parsed": len(dataset) - n_unparseable,
        "n_unparseable": n_unparseable,
    }
    print(
        f"  Done. Accuracy={metrics['accuracy']}, F1={metrics['f1']}, "
        f"Unparseable={n_unparseable}"
    )

    with detail_path.open("a", encoding="utf-8") as f:
        f.write("\n")
        f.write("| Technique | Accuracy | Precision | Recall | F1 | Parsed / Total |\n")
        f.write("|-----------|----------|-----------|--------|----|----------------|\n")
        f.write(
            f"| {metrics['technique']} | {metrics['accuracy']} | {metrics['precision']} "
            f"| {metrics['recall']} | {metrics['f1']} | {metrics['n_parsed']}/{metrics['n_total']} |\n"
        )

    return metrics


def parse_args() -> argparse.Namespace:
    """
    Разбирает аргументы командной строки для выбора техник оценки.

    Флаг -f позволяет передать список названий техник через пробел.
    Если флаг не указан — запускаются все доступные техники.

    Возвращает:
        Namespace с атрибутом f: список названий техник или None.
    """
    parser = argparse.ArgumentParser(description="Evaluate prompting techniques on SMS Spam dataset.")
    parser.add_argument(
        "-f",
        nargs="*",
        metavar="TECHNIQUE",
        help="Techniques to evaluate (e.g. zero_shot cot). Omit to run all.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Точка входа скрипта оценки техник промптинга.

    Загружает датасет, определяет список техник для запуска на основе аргументов CLI,
    последовательно вызывает evaluate_technique для каждой и выводит сводную таблицу
    метрик в stdout по завершении всех оценок.
    """
    args = parse_args()

    if args.f is None:
        techniques = TECHNIQUES
    else:
        unknown = [n for n in args.f if n not in TECHNIQUES]
        if unknown:
            print(f"ERROR: Unknown techniques: {', '.join(unknown)}")
            print(f"Available: {', '.join(TECHNIQUES)}")
            sys.exit(1)
        techniques = {n: TECHNIQUES[n] for n in args.f}

    if not DATA_PATH.exists():
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        print("Please download SMS Spam Collection from Kaggle and place spam.csv in data/")
        sys.exit(1)

    dataset = load_dataset(DATA_PATH, SAMPLE_PER_CLASS)

    all_results = []
    for name, system_prompt in techniques.items():
        result = evaluate_technique(
            technique_name=name,
            system_prompt=system_prompt,
            dataset=dataset,
        )
        all_results.append(result)

    print("\n=== Summary ===")
    for r in all_results:
        print(
            f"{r['technique']:15s} | acc={r['accuracy']} | prec={r['precision']} "
            f"| rec={r['recall']} | f1={r['f1']}"
        )


if __name__ == "__main__":
    main()
