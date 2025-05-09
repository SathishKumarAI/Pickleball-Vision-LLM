"""Game repository implementation."""
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import select
from .base import BaseRepository
from ..domain.models import Game, Team, Rally, Shot, PlayerPosition

class GameRepository(BaseRepository[Game]):
    """Repository for game-related operations."""
    
    def get(self, id: str) -> Optional[Game]:
        """Get game by ID with related entities."""
        query = (
            select(Game)
            .outerjoin(Team)
            .outerjoin(Rally)
            .outerjoin(Shot)
            .outerjoin(PlayerPosition)
            .where(Game.id == id)
        )
        return self.session.execute(query).scalar_one_or_none()
    
    def list(self, filters: Dict[str, Any] = None) -> List[Game]:
        """List games with optional filters."""
        query = select(Game)
        
        if filters:
            if "user_id" in filters:
                query = query.where(Game.user_id == filters["user_id"])
            if "date_from" in filters:
                query = query.where(Game.date >= filters["date_from"])
            if "date_to" in filters:
                query = query.where(Game.date <= filters["date_to"])
            if "location" in filters:
                query = query.where(Game.location == filters["location"])
        
        return list(self.session.execute(query).scalars().all())
    
    def create(self, game: Game) -> Game:
        """Create new game with related entities."""
        self.session.add(game)
        self.session.flush()
        return game
    
    def update(self, id: str, game: Game) -> Game:
        """Update existing game."""
        existing_game = self.get(id)
        if existing_game:
            for key, value in game.__dict__.items():
                if key != "id":
                    setattr(existing_game, key, value)
            self.session.flush()
            return existing_game
        return game
    
    def delete(self, id: str) -> bool:
        """Delete game and related entities."""
        game = self.get(id)
        if game:
            self.session.delete(game)
            return True
        return False
    
    def get_player_games(self, player_id: str) -> List[Game]:
        """Get all games for a specific player."""
        query = (
            select(Game)
            .join(Team)
            .join(Rally)
            .join(Shot)
            .where(Shot.player_id == player_id)
            .distinct()
        )
        return list(self.session.execute(query).scalars().all())
    
    def get_team_games(self, team_id: str) -> List[Game]:
        """Get all games for a specific team."""
        query = (
            select(Game)
            .join(Team)
            .where(Team.id == team_id)
        )
        return list(self.session.execute(query).scalars().all())
    
    def get_game_stats(self, game_id: str) -> Dict[str, Any]:
        """Get game statistics."""
        game = self.get(game_id)
        if not game:
            return {}
        
        return {
            "total_rallies": len(game.rallies),
            "total_shots": sum(len(rally.shots) for rally in game.rallies),
            "average_rally_duration": sum(rally.duration for rally in game.rallies) / len(game.rallies) if game.rallies else 0,
            "team_scores": {
                team.id: len([r for r in game.rallies if r.winner_team == team.id])
                for team in game.teams
            }
        } 