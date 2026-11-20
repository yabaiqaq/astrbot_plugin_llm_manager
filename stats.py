"""Token 消耗统计存储（JSONL 追加写）。

每次成功调用只追加一行，避免旧版“每个请求重写整个 stats.json”的
O(n) 写放大；聚合时使用内存副本，跨实例通过文件 mtime+size 自动重载。

旧版 `stats.json`（JSON 对象）会在首次加载时迁移到同目录的
`stats.jsonl`，原文件保留不动，便于回滚与人工核对。
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

#: 单文件最大记录数（超过时压缩，只保留最新的）
MAX_RECORDS = 50000

#: 保留天数清理的最小间隔（秒）；平时只追加，不做全表扫描
PRUNE_INTERVAL_SECONDS = 86400

#: 同一进程内的所有 Stats 实例共用一把文件锁：
#: 热重载后可能同时存在两个实例，串行化文件操作避免追加/压缩交错。
_FILE_LOCK = threading.RLock()


class Stats:
    """Token 消耗统计存储。

    线程安全（读写均加锁），写入采用追加模式；只有迁移、清理或压缩时
    才会原子重写整个文件。
    """

    def __init__(
        self,
        path: str | Path,
        retention_days: int = DEFAULT_RETENTION_DAYS,
    ) -> None:
        given = Path(path)
        # 兼容调用方传入 stats.json：实际数据落在同目录的 stats.jsonl
        self.legacy_path = given if given.suffix.lower() == ".json" else None
        self.path = (
            given.with_suffix(".jsonl")
            if given.suffix.lower() == ".json"
            else given
        )
        self.retention_days = int(retention_days)
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = []
        self._file_state: tuple[int, int] | None = None
        self._last_prune = time.time()
        with self._lock, _FILE_LOCK:
            self._load_locked()

    # ---------- 文件状态 ----------

    def _stat_state(self) -> tuple[int, int] | None:
        """返回统计文件 (mtime_ns, size)；文件不存在时返回 None。"""
        try:
            st = self.path.stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def refresh_if_changed(self) -> bool:
        """其他 Stats 实例写盘后重新载入，返回是否发生重载。"""
        with self._lock, _FILE_LOCK:
            return self._refresh_locked()

    def _refresh_locked(self) -> bool:
        state = self._stat_state()
        if state == self._file_state:
            return False
        if state is None:
            # 文件被外部删除：内存副本一并清空，避免下次聚合显示已删除的数据
            self._records = []
            self._file_state = None
            return True
        self._load_jsonl_locked()
        return True

    # ---------- 载入与迁移 ----------

    def _load_locked(self) -> None:
        if self.path.exists():
            self._load_jsonl_locked()
            return
        if self.legacy_path is not None and self.legacy_path.exists():
            self._records = self._read_legacy_locked()
            # 首次升级：把旧 JSON 里的记录写成 JSONL，旧文件保留
            self._compact_locked()
            return
        self._records = []
        self._file_state = None

    def _load_jsonl_locked(self) -> None:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError:
            # 短暂占用时保留内存副本，下次再试
            return

        records: list[dict[str, Any]] = []
        broken = False
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                broken = True
                continue
            if isinstance(item, dict):
                records.append(item)
            else:
                broken = True
        # 崩溃可能留下半行：修复一次，避免后续追加粘连成无效 JSON
        if raw and not raw.endswith("\n"):
            broken = True
        self._records = records
        if broken:
            self._compact_locked()
        else:
            self._file_state = self._stat_state()

    def _read_legacy_locked(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self.legacy_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if not isinstance(data, dict):
            return []
        records = data.get("records")
        if not isinstance(records, list):
            return []
        return [item for item in records if isinstance(item, dict)]

    # ---------- 记录 ----------

    def record(
        self,
        model: str,
        instance_id: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        fallback_from: str = "",
    ) -> None:
        """记录一次成功调用的 token 消耗；fallback_from 非空表示发生过降级。"""
        rec = {
            "ts": int(time.time()),
            "model": str(model),
            "instance": str(instance_id),
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
            "total_tokens": int(total_tokens or 0),
        }
        if fallback_from:
            rec["fallback_from"] = str(fallback_from)
        with self._lock, _FILE_LOCK:
            # 先吸收其他实例刚写入的记录，避免压缩时把它们丢掉
            self._refresh_locked()
            self._records.append(rec)
            self._append_locked(rec)
            self._maybe_prune_locked()

    def _append_locked(self, rec: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
        self._file_state = self._stat_state()

    def _maybe_prune_locked(self) -> None:
        """仅在超量或跨天时清理一次，避免每个请求都全表扫描。"""
        now = time.time()
        over_limit = len(self._records) > MAX_RECORDS
        if not over_limit and now - self._last_prune < PRUNE_INTERVAL_SECONDS:
            return

        records = self._records
        if over_limit:
            records = records[-MAX_RECORDS:]
        cutoff = int(now) - self.retention_days * 86400
        records = [
            item for item in records
            if int(item.get("ts", 0) or 0) >= cutoff
        ]
        if len(records) != len(self._records):
            self._records = records
            self._compact_locked()
        self._last_prune = now

    def _compact_locked(self) -> None:
        """原子重写整个 JSONL（迁移、修复、清理时使用）。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = "".join(
            json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
            for item in self._records
        )
        tmp = self.path.with_name(
            f"{self.path.name}.{threading.get_ident()}.tmp"
        )
        try:
            tmp.write_text(body, encoding="utf-8")
            tmp.replace(self.path)
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass
        self._file_state = self._stat_state()

    # ---------- 聚合查询 ----------

    def _snapshot(self, days: int | None = None) -> list[dict[str, Any]]:
        with self._lock, _FILE_LOCK:
            self._refresh_locked()
            records = list(self._records)
        if days is None:
            return records
        cutoff = int(time.time()) - days * 86400
        return [r for r in records if int(r.get("ts", 0) or 0) >= cutoff]

    def aggregate(self, days: int | None = None) -> list[dict[str, Any]]:
        """按模型+实例聚合统计。

        Args:
            days: 时间范围（天），None 表示全部历史（受 retention_days 限制）。

        Returns:
            按 total_tokens 降序排列的列表，每项包含：
            model, instance, calls, prompt_tokens, completion_tokens, total_tokens
        """
        agg: dict[tuple[str, str], dict[str, Any]] = {}
        for r in self._snapshot(days):
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
            agg[key]["prompt_tokens"] += int(r.get("prompt_tokens", 0) or 0)
            agg[key]["completion_tokens"] += int(r.get("completion_tokens", 0) or 0)
            agg[key]["total_tokens"] += int(r.get("total_tokens", 0) or 0)

        return sorted(agg.values(), key=lambda x: x["total_tokens"], reverse=True)

    def total_calls(self, days: int | None = None) -> int:
        """统计总调用次数。"""
        return sum(item["calls"] for item in self.aggregate(days))

    def total_tokens(self, days: int | None = None) -> int:
        """统计总 token 消耗。"""
        return sum(item["total_tokens"] for item in self.aggregate(days))

    def fallback_calls(self, days: int | None = None) -> int:
        """统计发生过降级的成功响应次数。"""
        return sum(
            1 for r in self._snapshot(days)
            if r.get("fallback_from")
        )

    def record_count(self) -> int:
        """当前存储的原始记录数。"""
        return len(self._snapshot())
