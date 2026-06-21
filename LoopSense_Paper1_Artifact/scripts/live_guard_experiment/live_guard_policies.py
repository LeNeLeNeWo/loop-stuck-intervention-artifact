from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from live_guard_common import WARN_REPLAN_MESSAGE


FILE_RE = re.compile(r"(?:(?:\.{0,2}/)?[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.(?:py|js|ts|tsx|java|go|rs|rb|php|c|cc|cpp|h|hpp|md|rst|txt|yaml|yml|toml|ini|cfg|json)")


def extract_action_targets(action: dict[str, Any]) -> list[str]:
    command = str(action.get("command") or "")
    matches = FILE_RE.findall(command)
    normalized = []
    for item in matches:
        item = item.strip().strip("'\"")
        if item and item not in normalized:
            normalized.append(item)
    if normalized:
        return normalized
    cleaned = re.sub(r"\s+", " ", command).strip()
    if not cleaned:
        return ["<empty-action>"]
    return [cleaned[:120]]


def action_tool(action: dict[str, Any]) -> str:
    command = str(action.get("command") or "").strip()
    return command.split()[0] if command else "<empty>"


@dataclass
class GuardEvent:
    step: int
    guard_name: str
    arm: str
    target: str
    recent_targets: list[str]
    decision: str


@dataclass
class GuardState:
    arm: str
    window: int = 5
    repeats: int = 3
    max_step_threshold: int = 50
    recent_targets: list[str] = field(default_factory=list)
    recent_tools: list[str] = field(default_factory=list)
    events: list[GuardEvent] = field(default_factory=list)
    warned: bool = False
    hard_stopped: bool = False
    trigger_step: int | None = None

    def observe_actions(self, step: int, actions: list[dict[str, Any]]) -> GuardEvent | None:
        primary_targets: list[str] = []
        for action in actions:
            self.recent_tools.append(action_tool(action))
            primary_targets.extend(extract_action_targets(action))
        if not primary_targets:
            primary_targets = ["<no-target>"]
        for target in primary_targets:
            self.recent_targets.append(target)
        self.recent_targets = self.recent_targets[-self.window :]
        self.recent_tools = self.recent_tools[-self.window :]
        for target in reversed(primary_targets):
            count = sum(1 for item in self.recent_targets if item == target)
            if count >= self.repeats:
                decision = "none"
                if self.arm == "WARN_REPLAN_GUARD" and not self.warned:
                    self.warned = True
                    decision = "warn_replan"
                elif self.arm == "HARD_STOP_GUARD":
                    self.hard_stopped = True
                    decision = "hard_stop"
                else:
                    decision = "guard_trigger_observed"
                event = GuardEvent(
                    step=step,
                    guard_name=f"file_revisit_guard window={self.window},repeats={self.repeats}",
                    arm=self.arm,
                    target=target,
                    recent_targets=list(self.recent_targets),
                    decision=decision,
                )
                self.events.append(event)
                if self.trigger_step is None:
                    self.trigger_step = step
                return event
        return None

    def warning_message(self) -> dict[str, Any]:
        return {"role": "user", "content": WARN_REPLAN_MESSAGE, "extra": {"intervention": "WARN_REPLAN_GUARD"}}


def count_repetitions(actions_by_step: list[list[dict[str, Any]]], window: int = 5) -> dict[str, Any]:
    targets: list[str] = []
    tools: list[str] = []
    repeated_file_count = 0
    repeated_tool_count = 0
    for actions in actions_by_step:
        step_targets: list[str] = []
        for action in actions:
            step_targets.extend(extract_action_targets(action))
            tools.append(action_tool(action))
        for target in step_targets:
            if target in targets[-window:]:
                repeated_file_count += 1
            targets.append(target)
    for idx, tool in enumerate(tools):
        if tool in tools[max(0, idx - window) : idx]:
            repeated_tool_count += 1
    total_actions = sum(len(actions) for actions in actions_by_step)
    return {
        "action_count": total_actions,
        "repeated_file_count": repeated_file_count,
        "repeated_tool_count": repeated_tool_count,
        "repeated_file_rate": repeated_file_count / total_actions if total_actions else 0.0,
        "repeated_tool_rate": repeated_tool_count / total_actions if total_actions else 0.0,
        "unique_targets": len(set(targets)),
    }
