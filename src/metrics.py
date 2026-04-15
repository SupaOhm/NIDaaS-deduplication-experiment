from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class RunMetrics:
    total_input: int = 0
    total_output: int = 0
    total_dropped: int = 0
    batch_times: list[float] = field(default_factory=list)

    def add_batch(self, input_events: int, output_events: int, dropped: int, elapsed_s: float) -> None:
        self.total_input += input_events
        self.total_output += output_events
        self.total_dropped += dropped
        self.batch_times.append(elapsed_s)

    def summary(self) -> dict:
        total_time = sum(self.batch_times)
        return {
            "total_input": self.total_input,
            "total_output": self.total_output,
            "total_dropped": self.total_dropped,
            "total_time_s": total_time,
            "throughput_events_per_s": self.total_input / total_time if total_time > 0 else 0.0,
            "avg_batch_time_s": total_time / len(self.batch_times) if self.batch_times else 0.0,
        }