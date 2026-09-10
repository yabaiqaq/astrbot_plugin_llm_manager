"""Token 消耗统计存储。

记录每次 LLM 调用的模型、实例和 token 用量，支持按时间范围聚合排行。
只依赖标准库，便于独立单元测试。
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

#: 统计记录默认保留天数
DEFAULT_RETENTION_DAYS = 30

#: 单文件最大记录数（超过时清理最旧的）
MAX_RECORDS = 50000


class Stats:
    """Token 消耗统计存储。

    线程安全（写操作加锁），读写采用原子替换（临时文件 + rename）。
    """

    def __init__(
        self,
        path: str | Path,
        retention_days: int = DEFAULT_RETENTION_DAYS,
    ) -> None:
        self.path = Path(path)
        self.retention_days = int(retention_days)
        self._lock = threading.Lock()
        self._data: dict[str, Any] = self._load()

    # ---------- 基础 ----------

    def _default(self) -> dict[str, Any]:
        return {"version": 1, "records": []}

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "records" in data:
                    return data
            except Exception:
                pass
        return self._default()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    # ---------- 记录 ----------

    def record(
        self,
        model: str,
        instance_id: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
    ) -> None:
        """记录一次调用的 token 消耗。"""
        with self._lock:
            self._data.setdefault("records", []).append({
                "ts": int(time.time()),
                "model": str(model),
                "instance": str(instance_id),
                "prompt_tokens": int(prompt_tokens or 0),
                "completion_tokens": int(completion_tokens or 0),
                "total_tokens": int(total_tokens or 0),
            })
            self._prune_locked()
            self._save()

    def _prune_locked(self) -> None:
        """清理过期和过多的记录（调用方需持有锁）。"""
        records = self._data.get("records", [])
        # 超过最大记录数时，保留最新的 MAX_RECORDS 条
        if len(records) > MAX_RECORDS:
            records = records[-MAX_RECORDS:]
        # 清理超过 retention_days 天的记录
        cutoff = int(time.time()) - self.retention_days * 86400
        self._data["records"] = [r for r in records if r.get("ts", 0) >= cutoff]

    # ---------- 聚合查询 ----------

    def aggregate(self, days: int | None = None) -> list[dict[str, Any]]:
        """按模型+实例聚合统计。

        Args:
            days: 时间范围（天），None 表示全部历史（受 retention_days 限制）。

        Returns:
            按 total_tokens 降序排列的列表，每项包含：
            model, instance, calls, prompt_tokens, completion_tokens, total_tokens
        """
        cutoff = None
        if days is not None:
            cutoff = int(time.time()) - days * 86400

        agg: dict[tuple[str, str], dict[str, Any]] = {}
        for r in self._data.get("records", []):
            if cutoff is not None and r.get("ts", 0) < cutoff:
                continue
            key = (r.get("model", ""), r.get("instance", ""))
            if key not in agg:
                agg[key] = {
                    "model": r.get("model", ""),
                    "instance": r.get("instance", ""),
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            agg[key]["calls"] += 1
            agg[key]["prompt_tokens"] += r.get("prompt_tokens", 0)
            agg[key]["completion_tokens"] += r.get("completion_tokens", 0)
            agg[key]["total_tokens"] += r.get("total_tokens", 0)

        return sorted(agg.values(), key=lambda x: x["total_tokens"], reverse=True)

    def total_calls(self, days: int | None = None) -> int:
        """统计总调用次数。"""
        return sum(item["calls"] for item in self.aggregate(days))

    def total_tokens(self, days: int | None = None) -> int:
        """统计总 token 消耗。"""
        return sum(item["total_tokens"] for item in self.aggregate(days))

    def record_count(self) -> int:
        """当前存储的原始记录数。"""
        return len(self._data.get("records", []))
