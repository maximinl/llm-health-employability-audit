import argparse
import csv
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv


RATING_FIELDS = [
    "communication",
    "harmony",
    "ambition",
    "competence",
    "responsibility",
    "dependability",
    "resilience",
    "independence",
    "recommendation",
    "elimination",
]

PROVIDER_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# "elimination" is negatively keyed (agreement that the applicant should be eliminated),
# so it is reverse-scored on the 1-9 scale before averaging.
REVERSE_SCORED_FIELDS = {"elimination"}
SCALE_MIN, SCALE_MAX = 1, 9


def employability_score(ratings: dict[str, Any]) -> float:
    scored = [
        (SCALE_MIN + SCALE_MAX - ratings[field]) if field in REVERSE_SCORED_FIELDS else ratings[field]
        for field in RATING_FIELDS
    ]
    return sum(scored) / len(scored)


def build_cover_letter(base: str, disclosure_text: str) -> str:
    if not disclosure_text:
        return base
    return f"{base}\n\n{disclosure_text}"


def build_tasks(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    jobs = load_json(root / config["jobs_path"])
    disclosures = load_json(root / config["disclosures_path"])
    prompt_conditions = config["prompt_conditions"]
    enabled_models = [model for model in config["models"] if model.get("enabled", True)]

    tasks = []
    for model in enabled_models:
        for prompt_condition in prompt_conditions:
            template = load_text(root / prompt_condition["template_path"])
            for job in jobs:
                for disclosure in disclosures:
                    cover_letter = build_cover_letter(job["cover_letter_base"], disclosure["text"])
                    prompt = template.format(
                        job_type=job["job_type"],
                        job_advert=job["job_advert"],
                        cv=job["cv"],
                        cover_letter=cover_letter,
                    )
                    stimulus_id = f"{job['job_type']}__{disclosure['disclosure']}__{prompt_condition['id']}"
                    for replication in range(1, int(config["replications"]) + 1):
                        tasks.append(
                            {
                                "experiment_id": config["experiment_id"],
                                "provider": model["provider"],
                                "model": model["model"],
                                "prompt_condition": prompt_condition["id"],
                                "job_type": job["job_type"],
                                "disclosure": disclosure["disclosure"],
                                "stimulus_id": stimulus_id,
                                "replication": replication,
                                "prompt": prompt,
                            }
                        )
    random.shuffle(tasks)
    return tasks


def task_key(task: dict[str, Any]) -> tuple[str, str, str, str, str, int]:
    return (
        task["provider"],
        task["model"],
        task["prompt_condition"],
        task["job_type"],
        task["disclosure"],
        int(task["replication"]),
    )


def load_completed_keys(raw_output_path: Path) -> set[tuple[str, str, str, str, str, int]]:
    if not raw_output_path.exists():
        return set()

    completed = set()
    with raw_output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") == "ok":
                completed.add(task_key(record))
    return completed


def extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def validate_ratings(parsed: dict[str, Any]) -> dict[str, Any]:
    clean = {}
    for field in RATING_FIELDS:
        value = parsed[field]
        if isinstance(value, str):
            value = value.strip()
        value = int(value)
        if value == 0:
            value = 1
        if value < 1 or value > 9:
            raise ValueError(f"{field} rating {value} is outside 1-9")
        clean[field] = value
    clean["short_rationale"] = str(parsed.get("short_rationale", "")).strip()
    return clean


def call_and_parse_with_retries(
    task: dict[str, Any],
    config: dict[str, Any],
    max_attempts: int,
    retry_wait_seconds: float,
) -> tuple[dict[str, Any], str]:
    last_error: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            text = call_model(task, config)
            parsed = extract_json(text)
            return validate_ratings(parsed), text
        except Exception as exc:
            last_error = exc
            if attempt < max_attempts:
                time.sleep(retry_wait_seconds * attempt)

    if last_error is None:
        raise RuntimeError("Model call failed without an exception.")
    raise last_error


def call_openai(task: dict[str, Any], config: dict[str, Any]) -> str:
    from openai import OpenAI

    client = OpenAI()
    uses_new_token_arg = task["model"].startswith(("gpt-5", "o3", "o4"))
    token_arg = "max_completion_tokens" if uses_new_token_arg else "max_tokens"
    request = {
        "model": task["model"],
        "messages": [{"role": "user", "content": task["prompt"]}],
        "response_format": {"type": "json_object"},
        token_arg: int(config["max_output_tokens"]),
    }
    if not uses_new_token_arg:
        request["temperature"] = float(config["temperature"])

    response = client.chat.completions.create(**request)
    return response.choices[0].message.content or ""


def call_anthropic(task: dict[str, Any], config: dict[str, Any]) -> str:
    import anthropic

    client = anthropic.Anthropic()
    request = {
        "model": task["model"],
        "max_tokens": int(config["max_output_tokens"]),
        "messages": [{"role": "user", "content": task["prompt"]}],
    }

    response = client.messages.create(**request)
    return "".join(block.text for block in response.content if getattr(block, "type", "") == "text")


def call_google(task: dict[str, Any], config: dict[str, Any]) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client()
    response = client.models.generate_content(
        model=task["model"],
        contents=task["prompt"],
        config=types.GenerateContentConfig(
            temperature=float(config["temperature"]),
            max_output_tokens=int(config["max_output_tokens"]),
            response_mime_type="application/json",
        ),
    )
    return response.text or ""


def call_model(task: dict[str, Any], config: dict[str, Any]) -> str:
    provider = task["provider"]
    if provider == "openai":
        return call_openai(task, config)
    if provider == "anthropic":
        return call_anthropic(task, config)
    if provider == "google":
        return call_google(task, config)
    raise ValueError(f"Unsupported provider: {provider}")


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def rebuild_processed(raw_output_path: Path, processed_output_path: Path) -> None:
    processed_output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    if raw_output_path.exists():
        with raw_output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("status") != "ok":
                    continue
                ratings = record["ratings"]
                mean_employability = employability_score(ratings)
                rows.append(
                    {
                        "experiment_id": record["experiment_id"],
                        "timestamp_utc": record["timestamp_utc"],
                        "provider": record["provider"],
                        "model": record["model"],
                        "prompt_condition": record["prompt_condition"],
                        "job_type": record["job_type"],
                        "disclosure": record["disclosure"],
                        "stimulus_id": record["stimulus_id"],
                        "replication": record["replication"],
                        **{field: ratings[field] for field in RATING_FIELDS},
                        "mean_employability": round(mean_employability, 4),
                        "short_rationale": ratings.get("short_rationale", ""),
                    }
                )

    fieldnames = [
        "experiment_id",
        "timestamp_utc",
        "provider",
        "model",
        "prompt_condition",
        "job_type",
        "disclosure",
        "stimulus_id",
        "replication",
        *RATING_FIELDS,
        "mean_employability",
        "short_rationale",
    ]
    with processed_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    config = load_json(root / args.config)
    raw_output_path = root / config["raw_output_path"]
    processed_output_path = root / config["processed_output_path"]
    tasks = build_tasks(config, root)

    if args.providers:
        providers = {provider.strip() for provider in args.providers.split(",") if provider.strip()}
        tasks = [task for task in tasks if task["provider"] in providers]

    if args.limit is not None:
        tasks = tasks[: args.limit]

    if args.dry_run:
        print(f"Dry run: {len(tasks)} tasks would be executed.")
        for task in tasks[:5]:
            print(
                f"- {task['provider']}:{task['model']} "
                f"{task['prompt_condition']} {task['job_type']} {task['disclosure']} "
                f"rep={task['replication']}"
            )
        return

    missing_keys = sorted(
        {
            PROVIDER_ENV_KEYS[task["provider"]]
            for task in tasks
            if task["provider"] in PROVIDER_ENV_KEYS and not os.getenv(PROVIDER_ENV_KEYS[task["provider"]])
        }
    )
    if missing_keys:
        joined = ", ".join(missing_keys)
        raise RuntimeError(f"Missing required API key environment variables: {joined}")

    completed = load_completed_keys(raw_output_path)
    wait_seconds = float(args.wait_seconds)
    executed = 0
    skipped = 0

    for task in tasks:
        if task_key(task) in completed:
            skipped += 1
            continue

        record = {key: value for key, value in task.items() if key != "prompt"}
        record["timestamp_utc"] = utc_now()
        try:
            ratings, text = call_and_parse_with_retries(
                task,
                config,
                max_attempts=int(args.max_attempts),
                retry_wait_seconds=float(args.retry_wait_seconds),
            )
            record["ratings"] = ratings
            record["raw_response"] = text
            record["status"] = "ok"
        except Exception as exc:
            record["status"] = "error"
            record["error_type"] = type(exc).__name__
            record["error"] = str(exc)

        write_jsonl(raw_output_path, record)
        executed += 1
        print(
            f"{record['status']}: {record['provider']}:{record['model']} "
            f"{record['prompt_condition']} {record['job_type']} {record['disclosure']} "
            f"rep={record['replication']}"
        )
        if wait_seconds > 0:
            time.sleep(wait_seconds)

    rebuild_processed(raw_output_path, processed_output_path)
    print(f"Executed {executed} tasks, skipped {skipped} completed tasks.")
    print(f"Raw responses: {raw_output_path}")
    print(f"Processed ratings: {processed_output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LLM health disclosure employability audit.")
    parser.add_argument("--config", default="config/experiment.json", help="Path to experiment config.")
    parser.add_argument("--dry-run", action="store_true", help="Print tasks without calling APIs.")
    parser.add_argument("--limit", type=int, default=None, help="Optional task limit for pilot runs.")
    parser.add_argument(
        "--providers",
        default=None,
        help="Comma-separated provider filter, for example: openai or openai,anthropic.",
    )
    parser.add_argument("--max-attempts", type=int, default=3, help="Attempts per task before writing an error.")
    parser.add_argument("--retry-wait-seconds", type=float, default=3.0, help="Base wait between retries.")
    parser.add_argument("--wait-seconds", type=float, default=0.5, help="Delay between API calls.")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
