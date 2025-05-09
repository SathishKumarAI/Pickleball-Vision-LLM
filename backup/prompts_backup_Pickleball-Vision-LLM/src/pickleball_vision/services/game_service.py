"""Game service for business logic."""
from typing import List, Optional, Dict, Any
from datetime import datetime
from ..repositories.game_repository import GameRepository
from ..domain.models import Game, Team, Rally, Shot, PlayerPosition, AnalysisResult
from ..analytics.strategy_analyzer import StrategyAnalyzer
from ..visualization.advanced_visualizer import AdvancedVisualizer

class GameService:
    """Service for game-related business logic."""
    
    def __init__(
        self,
        game_repository: GameRepository,
        strategy_analyzer: StrategyAnalyzer,
        advanced_visualizer: AdvancedVisualizer
    ):
        """Initialize service with dependencies."""
        self.game_repository = game_repository
        self.strategy_analyzer = strategy_analyzer
        self.advanced_visualizer = advanced_visualizer
    
    def create_game(self, game_data: Dict[str, Any]) -> Game:
        """Create new game with analysis."""
        # Create game entity
        game = Game(
            id=game_data["id"],
            user_id=game_data["user_id"],
            teams=game_data["teams"],
            rallies=game_data["rallies"],
            date=datetime.fromisoformat(game_data["date"]),
            duration=game_data["duration"],
            score=game_data["score"],
            location=game_data["location"],
            metadata=game_data.get("metadata", {})
        )
        
        # Save game
        game = self.game_repository.create(game)
        
        # Analyze game
        analysis = self.analyze_game(game)
        
        return game
    
    def get_game(self, game_id: str) -> Optional[Game]:
        """Get game by ID."""
        return self.game_repository.get(game_id)
    
    def list_games(self, filters: Dict[str, Any] = None) -> List[Game]:
        """List games with optional filters."""
        return self.game_repository.list(filters)
    
    def update_game(self, game_id: str, game_data: Dict[str, Any]) -> Optional[Game]:
        """Update existing game."""
        game = self.game_repository.get(game_id)
        if not game:
            return None
        
        # Update game fields
        for key, value in game_data.items():
            if hasattr(game, key):
                setattr(game, key, value)
        
        # Save changes
        game = self.game_repository.update(game_id, game)
        
        # Re-analyze game
        self.analyze_game(game)
        
        return game
    
    def delete_game(self, game_id: str) -> bool:
        """Delete game."""
        return self.game_repository.delete(game_id)
    
    def analyze_game(self, game: Game) -> AnalysisResult:
        """Analyze game strategy and performance."""
        # Get game data for analysis
        game_data = {
            "shots": [
                {
                    "player_id": shot.player_id,
                    "shot_type": shot.shot_type.value,
                    "placement_x": shot.placement.x,
                    "placement_y": shot.placement.y,
                    "speed": shot.speed,
                    "spin": shot.spin,
                    "timestamp": shot.timestamp.isoformat(),
                    "effectiveness_score": shot.effectiveness_score
                }
                for rally in game.rallies
                for shot in rally.shots
            ],
            "player_positions": [
                {
                    "player_id": pos.player_id,
                    "x": pos.position.x,
                    "y": pos.position.y,
                    "timestamp": pos.timestamp.isoformat(),
                    "speed": pos.speed
                }
                for pos in game.player_positions
            ],
            "rallies": [
                {
                    "duration": rally.duration,
                    "winner_team": rally.winner_team,
                    "ending_type": rally.ending_type,
                    "start_time": rally.start_time.isoformat(),
                    "end_time": rally.end_time.isoformat()
                }
                for rally in game.rallies
            ]
        }
        
        # Analyze strategy
        analysis = self.strategy_analyzer.analyze_strategy(game_data)
        
        # Create analysis result
        analysis_result = AnalysisResult(
            id=f"analysis_{game.id}",
            game_id=game.id,
            analysis_type="strategy",
            metrics=analysis,
            created_at=datetime.now(),
            metadata={}
        )
        
        return analysis_result
    
    def get_game_visualizations(self, game_id: str) -> Dict[str, Any]:
        """Get visualizations for game analysis."""
        game = self.game_repository.get(game_id)
        if not game:
            return {}
        
        # Prepare game data for visualization
        game_data = {
            "shots": [
                {
                    "player_id": shot.player_id,
                    "shot_type": shot.shot_type.value,
                    "placement_x": shot.placement.x,
                    "placement_y": shot.placement.y,
                    "speed": shot.speed,
                    "spin": shot.spin,
                    "timestamp": shot.timestamp.isoformat(),
                    "effectiveness_score": shot.effectiveness_score
                }
                for rally in game.rallies
                for shot in rally.shots
            ],
            "player_positions": [
                {
                    "player_id": pos.player_id,
                    "x": pos.position.x,
                    "y": pos.position.y,
                    "timestamp": pos.timestamp.isoformat(),
                    "speed": pos.speed
                }
                for pos in game.player_positions
            ],
            "rallies": [
                {
                    "duration": rally.duration,
                    "winner_team": rally.winner_team,
                    "ending_type": rally.ending_type,
                    "start_time": rally.start_time.isoformat(),
                    "end_time": rally.end_time.isoformat()
                }
                for rally in game.rallies
            ],
            "games": [game.__dict__],
            "analysis_results": [self.analyze_game(game).__dict__]
        }
        
        # Create visualizations
        return self.advanced_visualizer.create_advanced_dashboard(game_data)
    
    def get_player_stats(self, player_id: str) -> Dict[str, Any]:
        """Get player statistics across all games."""
        games = self.game_repository.get_player_games(player_id)
        
        # Calculate player statistics
        total_games = len(games)
        total_shots = 0
        total_wins = 0
        shot_types = {}
        
        for game in games:
            # Count shots
            for rally in game.rallies:
                for shot in rally.shots:
                    if shot.player_id == player_id:
                        total_shots += 1
                        shot_types[shot.shot_type.value] = shot_types.get(shot.shot_type.value, 0) + 1
            
            # Count wins
            player_team = next(
                (team for team in game.teams if any(p.id == player_id for p in team.players)),
                None
            )
            if player_team:
                team_wins = len([r for r in game.rallies if r.winner_team == player_team.id])
                total_wins += team_wins
        
        return {
            "total_games": total_games,
            "total_shots": total_shots,
            "total_wins": total_wins,
            "win_rate": total_wins / total_games if total_games > 0 else 0,
            "shot_types": shot_types,
            "average_shots_per_game": total_shots / total_games if total_games > 0 else 0
        } 