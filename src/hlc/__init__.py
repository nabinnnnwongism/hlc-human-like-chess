"""Human-Like Chess (HLC) package."""

from hlc.core import BotCore
from hlc.scheduler import Scheduler, SchedulerConfig
from hlc.timing.base import DEFAULT_BUCKET_EDGES, ThinkTimeDistribution, TimingModel
from hlc.timing.baselines import ClockOnlyBaseline, HeuristicBaseline
from hlc.types import CandidateMove, Decision, GameState, MoveDistribution, MoveEngine

# Maia3Engine is a lazy import — requires maia3 + torch which may not be present
# in lightweight / test environments.
def __getattr__(name: str):
    if name == "Maia3Engine":
        from hlc.engines.maia3 import Maia3Engine  # noqa: PLC0415
        return Maia3Engine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_BUCKET_EDGES",
    "BotCore",
    "CandidateMove",
    "ClockOnlyBaseline",
    "Decision",
    "GameState",
    "HeuristicBaseline",
    "Maia3Engine",
    "MoveDistribution",
    "MoveEngine",
    "Scheduler",
    "SchedulerConfig",
    "ThinkTimeDistribution",
    "TimingModel",
]
