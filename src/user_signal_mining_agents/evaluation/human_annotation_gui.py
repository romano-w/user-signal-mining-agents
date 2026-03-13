from __future__ import annotations

import json
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from pydantic import ValidationError

from ..schemas import HumanAnnotationResult, HumanAnnotationTask

RUBRIC_DIMENSIONS: list[dict[str, str]] = [
    {
        "key": "relevance",
        "label": "Relevance",
        "description": "Does the output address the founder's specific intent?",
    },
    {
        "key": "groundedness",
        "label": "Groundedness",
        "description": "Are focus points traceable to specific evidence snippets rather than generic claims?",
    },
    {
        "key": "distinctiveness",
        "label": "Distinctiveness",
        "description": "Are focus points distinct rather than repetitive?",
    },
]

_ANNOTATOR_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,63}$")
_HTML_TEMPLATE_PATH = Path(__file__).with_name("human_annotation_gui.html")


class AnnotationWorkspace:
    """Loads human annotation tasks and persists per-annotator results."""

    def __init__(self, tasks_dir: Path):
        self.tasks_dir = tasks_dir
        self.results_dir = tasks_dir / "_results"
        self.tasks = self._load_tasks()

    def _load_tasks(self) -> dict[str, HumanAnnotationTask]:
        if not self.tasks_dir.exists():
            raise FileNotFoundError(f"Task directory not found: {self.tasks_dir}")
        if not self.tasks_dir.is_dir():
            raise NotADirectoryError(f"Task path must be a directory: {self.tasks_dir}")

        tasks: dict[str, HumanAnnotationTask] = {}
        for path in sorted(self.tasks_dir.glob("*.json")):
            try:
                task = HumanAnnotationTask.model_validate_json(path.read_text(encoding="utf-8"))
            except (ValidationError, OSError, UnicodeDecodeError):
                continue

            if task.task_id in tasks:
                raise ValueError(f"Duplicate task_id {task.task_id!r} detected in {self.tasks_dir}")
            tasks[task.task_id] = task
        return tasks

    @staticmethod
    def normalize_annotator_id(annotator_id: str) -> str:
        value = annotator_id.strip()
        if not _ANNOTATOR_ID_PATTERN.fullmatch(value):
            raise ValueError(
                "annotator_id must be 2-64 chars and use letters, numbers, '.', '-', or '_'"
            )
        return value

    def _require_task(self, task_id: str) -> HumanAnnotationTask:
        task = self.tasks.get(task_id)
        if task is None:
            raise KeyError(f"Unknown task_id: {task_id}")
        return task

    def _result_path(self, task_id: str, annotator_id: str) -> Path:
        safe_annotator = self.normalize_annotator_id(annotator_id)
        return self.results_dir / safe_annotator / f"{task_id}.json"

    def list_tasks(self, annotator_id: str | None = None) -> list[dict[str, object]]:
        safe_annotator = None
        if annotator_id:
            safe_annotator = self.normalize_annotator_id(annotator_id)

        rows: list[dict[str, object]] = []
        for task_id in sorted(self.tasks):
            task = self.tasks[task_id]
            completed = False
            if safe_annotator is not None:
                completed = self._result_path(task_id, safe_annotator).exists()

            rows.append(
                {
                    "task_id": task.task_id,
                    "prompt_id": task.prompt.id,
                    "statement": task.prompt.statement,
                    "domain": task.prompt.domain,
                    "evidence_count": len(task.retrieved_evidence),
                    "system_a_focus_points": len(task.system_a_focus_points),
                    "system_b_focus_points": len(task.system_b_focus_points),
                    "is_completed": completed,
                }
            )
        return rows

    def get_public_task(self, task_id: str) -> dict[str, object]:
        task = self._require_task(task_id)
        return task.model_dump(mode="json", exclude={"ground_truth_mapping"})

    def load_result(self, task_id: str, annotator_id: str) -> dict[str, object] | None:
        self._require_task(task_id)
        path = self._result_path(task_id, annotator_id)
        if not path.exists():
            return None

        result = HumanAnnotationResult.model_validate_json(path.read_text(encoding="utf-8"))
        return result.model_dump(mode="json")

    def save_result(
        self,
        *,
        task_id: str,
        annotator_id: str,
        payload: dict[str, object],
    ) -> HumanAnnotationResult:
        self._require_task(task_id)
        safe_annotator = self.normalize_annotator_id(annotator_id)

        validated_payload = dict(payload)
        validated_payload["task_id"] = task_id
        validated_payload["annotator_id"] = safe_annotator
        result = HumanAnnotationResult.model_validate(validated_payload)

        out_path = self._result_path(task_id, safe_annotator)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result

    def export_results(self, annotator_id: str) -> list[dict[str, object]]:
        safe_annotator = self.normalize_annotator_id(annotator_id)
        rows: list[dict[str, object]] = []
        for task_id in sorted(self.tasks):
            result = self.load_result(task_id, safe_annotator)
            if result is not None:
                rows.append(result)
        return rows


def _build_index_html(default_annotator_id: str) -> str:
    html_template = _HTML_TEMPLATE_PATH.read_text(encoding="utf-8")
    return (
        html_template.replace("__DEFAULT_ANNOTATOR__", json.dumps(default_annotator_id))
        .replace("__RUBRIC_DIMENSIONS__", json.dumps(RUBRIC_DIMENSIONS))
    )


def _parse_query(path: str) -> dict[str, str]:
    parsed = urlparse(path)
    raw = parse_qs(parsed.query)
    return {k: v[0] for k, v in raw.items() if v}


def _path_parts(path: str) -> list[str]:
    return [part for part in urlparse(path).path.split("/") if part]


def _make_handler(
    workspace: AnnotationWorkspace,
    *,
    default_annotator_id: str,
) -> type[BaseHTTPRequestHandler]:
    class AnnotationRequestHandler(BaseHTTPRequestHandler):
        server_version = "USMAnnotationServer/1.0"

        def log_message(self, _format: str, *_args: object) -> None:
            return

        def _send_json(self, status: HTTPStatus, body: dict[str, object]) -> None:
            payload = json.dumps(body).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_html(self, html: str) -> None:
            payload = html.encode("utf-8")
            self.send_response(HTTPStatus.OK.value)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _read_json(self) -> dict[str, object]:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                raise ValueError("Request body is required.")
            raw = self.rfile.read(content_length).decode("utf-8")
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("JSON body must be an object.")
            return parsed

        def do_GET(self) -> None:  # noqa: N802
            try:
                parts = _path_parts(self.path)
                query = _parse_query(self.path)

                if parts in ([], ["index.html"]):
                    self._send_html(_build_index_html(default_annotator_id))
                    return

                if parts == ["api", "health"]:
                    self._send_json(HTTPStatus.OK, {"ok": True, "task_count": len(workspace.tasks)})
                    return

                if parts == ["api", "tasks"]:
                    annotator_id = query.get("annotator_id")
                    tasks = workspace.list_tasks(annotator_id=annotator_id)
                    completed = sum(1 for t in tasks if t["is_completed"])
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "tasks": tasks,
                            "completed": completed,
                            "total": len(tasks),
                            "rubric_dimensions": RUBRIC_DIMENSIONS,
                        },
                    )
                    return

                if len(parts) == 3 and parts[:2] == ["api", "tasks"]:
                    task_id = unquote(parts[2])
                    annotator_id = query.get("annotator_id")
                    result = workspace.load_result(task_id, annotator_id) if annotator_id else None
                    self._send_json(
                        HTTPStatus.OK,
                        {"task": workspace.get_public_task(task_id), "existing_result": result},
                    )
                    return

                if parts == ["api", "export"]:
                    annotator_id = query.get("annotator_id", "")
                    rows = workspace.export_results(annotator_id)
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "annotator_id": workspace.normalize_annotator_id(annotator_id),
                            "results": rows,
                            "count": len(rows),
                        },
                    )
                    return

                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Route not found."})
            except KeyError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            except (ValueError, ValidationError, json.JSONDecodeError) as exc:
                detail = exc.errors() if isinstance(exc, ValidationError) else str(exc)
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": detail})
            except Exception as exc:  # noqa: BLE001
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

        def do_POST(self) -> None:  # noqa: N802
            try:
                parts = _path_parts(self.path)
                if len(parts) == 4 and parts[:2] == ["api", "tasks"] and parts[3] == "result":
                    task_id = unquote(parts[2])
                    body = self._read_json()
                    annotator_id = body.get("annotator_id")
                    if not isinstance(annotator_id, str):
                        raise ValueError("annotator_id is required in request body.")
                    saved = workspace.save_result(
                        task_id=task_id,
                        annotator_id=annotator_id,
                        payload=body,
                    )
                    self._send_json(
                        HTTPStatus.OK,
                        {"ok": True, "result": saved.model_dump(mode="json")},
                    )
                    return

                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Route not found."})
            except KeyError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            except (ValueError, ValidationError, json.JSONDecodeError) as exc:
                detail = exc.errors() if isinstance(exc, ValidationError) else str(exc)
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": detail})
            except Exception as exc:  # noqa: BLE001
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    return AnnotationRequestHandler


def run_annotation_server(
    tasks_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    default_annotator_id: str = "",
) -> None:
    workspace = AnnotationWorkspace(tasks_dir)
    if not workspace.tasks:
        raise ValueError(f"No valid HumanAnnotationTask JSON files found under: {tasks_dir}")

    normalized_default = ""
    if default_annotator_id.strip():
        normalized_default = workspace.normalize_annotator_id(default_annotator_id)

    handler = _make_handler(workspace, default_annotator_id=normalized_default)
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}"

    print(f"Human annotation GUI running at {url}")
    print(f"Task directory: {tasks_dir}")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAnnotation server stopped.")
    finally:
        server.server_close()

