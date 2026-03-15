from dataclasses import dataclass
from typing import List


@dataclass
class Agent:
    name: str
    role: str


def sequential_default_agents() -> List[Agent]:
    return [
        Agent(name="Planner", role="planner"),
        Agent(name="Judger", role="judger"),
    ]

def hierarchical_default_agents() -> List[Agent]:
    return [
        Agent(name="Math", role="math"),
        Agent(name="Science", role="science"),
        Agent(name="Code", role="code"),
        Agent(name="Summarizer", role="summarizer"),
    ]


__all__ = ["Agent", "sequential_default_agents", "hierarchical_default_agents"]
