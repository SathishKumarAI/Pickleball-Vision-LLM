"""Domain models for the Pickleball Vision Analytics system."""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

class ShotType(Enum):
    """Types of shots in pickleball."""
    SERVE = "serve"
    RETURN = "return"
    DINK = "dink"
    DRIVE = "drive"
    DROP = "drop"
    SMASH = "smash"
    LOB = "lob"
    BLOCK = "block"

class SkillLevel(Enum):
    """Player skill levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    PROFESSIONAL = "professional"

@dataclass
class Point:
    """Represents a point on the court."""
    x: float
    y: float

@dataclass
class Shot:
    """Represents a shot in the game."""
    id: str
    player_id: str
    rally_id: str
    shot_type: ShotType
    placement: Point
    speed: float
    spin: float
    timestamp: datetime
    effectiveness_score: float
    metadata: Dict[str, Any]

@dataclass
class Player:
    """Represents a player in the system."""
    id: str
    name: str
    skill_level: SkillLevel
    team_id: Optional[str]
    metadata: Dict[str, Any]

@dataclass
class Team:
    """Represents a team of players."""
    id: str
    name: str
    players: List[Player]
    created_at: datetime
    metadata: Dict[str, Any]

@dataclass
class Rally:
    """Represents a rally in the game."""
    id: str
    game_id: str
    shots: List[Shot]
    duration: float
    winner_team: str
    ending_type: str
    start_time: datetime
    end_time: datetime
    metadata: Dict[str, Any]

@dataclass
class Game:
    """Represents a pickleball game."""
    id: str
    user_id: str
    teams: List[Team]
    rallies: List[Rally]
    date: datetime
    duration: float
    score: str
    location: str
    metadata: Dict[str, Any]

@dataclass
class AnalysisResult:
    """Represents the results of game analysis."""
    id: str
    game_id: str
    analysis_type: str
    metrics: Dict[str, float]
    created_at: datetime
    metadata: Dict[str, Any]

@dataclass
class User:
    """Represents a system user."""
    id: str
    username: str
    email: str
    subscription_plan: str
    created_at: datetime
    last_login: datetime
    metadata: Dict[str, Any]

@dataclass
class PlayerPosition:
    """Represents a player's position at a specific time."""
    id: str
    game_id: str
    player_id: str
    position: Point
    timestamp: datetime
    speed: float
    metadata: Dict[str, Any] 