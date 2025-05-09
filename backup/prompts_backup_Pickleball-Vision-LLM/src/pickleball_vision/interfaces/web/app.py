"""FastAPI application."""
from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
import uvicorn
from typing import Dict, Any, List, Optional
import json
import os
from datetime import timedelta, datetime

from ..container import Container
from ..domain.models import User, Game, Shot, Rally, PlayerPosition, AnalysisResult
from ..auth.auth import (
    authenticate_user,
    create_access_token,
    get_current_active_user,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from ..utils.logger import setup_logger
from ..analytics.gpu_analyzer import GPUAnalyzer
from ..analytics.ml_analyzer import MLAnalyzer
from ..analytics.stream_analyzer import StreamAnalyzer
from ..config.settings import get_settings

logger = setup_logger(__name__)

# Create container
container = Container()

# Initialize analyzers
gpu_analyzer = GPUAnalyzer()
ml_analyzer = MLAnalyzer()
stream_analyzer = StreamAnalyzer()

app = FastAPI(
    title="Pickleball Vision Analytics",
    description="API for analyzing pickleball games and player performance",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/token")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(container.db_session)
):
    """Login endpoint for user authentication."""
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/users/", response_model=Dict[str, str])
async def create_user(
    username: str,
    email: str,
    password: str,
    db: Session = Depends(container.db_session)
):
    """Create new user."""
    from ..auth.auth import create_user as create_user_func
    user = create_user_func(db, username, email, password)
    return {"message": "User created successfully", "user_id": user.id}

@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {
        "name": "Pickleball Vision Analytics API",
        "version": "1.0.0",
        "endpoints": [
            "/api/analyze",
            "/api/visualize",
            "/api/team-analysis",
            "/api/advanced-visualizations",
            "/api/games/{game_id}/advanced-analysis",
            "/api/players/{player_id}/advanced-stats"
        ]
    }

@app.post("/api/analyze")
async def analyze_strategy(
    data: Dict[str, Any],
    current_user: User = Depends(get_current_active_user),
    game_service = Depends(container.game_service_provider)
):
    """Analyze game strategy from provided data."""
    try:
        # Create game with analysis
        game = game_service.create_game(data)
        return game_service.analyze_game(game).metrics
    except Exception as e:
        logger.error(f"Error analyzing strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/visualize")
async def visualize_strategy(
    data: Dict[str, Any],
    current_user: User = Depends(get_current_active_user),
    game_service = Depends(container.game_service_provider)
):
    """Create visualizations for strategy analysis."""
    try:
        # Create game and get visualizations
        game = game_service.create_game(data)
        visualizations = game_service.get_game_visualizations(game.id)
        
        # Convert Plotly figures to JSON
        return {
            key: fig.to_json() for key, fig in visualizations.items()
        }
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/games/", response_model=List[Dict[str, Any]])
async def get_user_games(
    current_user: User = Depends(get_current_active_user),
    game_service = Depends(container.game_service_provider)
):
    """Get all games for current user."""
    games = game_service.list_games({"user_id": current_user.id})
    return [
        {
            "id": game.id,
            "date": game.date,
            "duration": game.duration,
            "score": game.score,
            "location": game.location
        }
        for game in games
    ]

@app.get("/api/games/{game_id}/analysis")
async def get_game_analysis(
    game_id: str,
    current_user: User = Depends(get_current_active_user),
    game_service = Depends(container.game_service_provider)
):
    """Get analysis results for specific game."""
    game = game_service.get_game(game_id)
    if not game or game.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Game not found")
    
    analysis = game_service.analyze_game(game)
    return analysis.metrics

@app.get("/api/games/{game_id}/advanced-analysis")
async def get_game_advanced_analysis(
    game_id: str,
    current_user: User = Depends(get_current_active_user),
    game_service = Depends(container.game_service_provider)
):
    """Get advanced analysis for specific game."""
    try:
        game = game_service.get_game(game_id)
        if not game or game.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Game not found")
        
        visualizations = game_service.get_game_visualizations(game_id)
        return {
            key: fig.to_json() for key, fig in visualizations.items()
        }
    except Exception as e:
        logger.error(f"Error getting advanced analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/players/{player_id}/advanced-stats")
async def get_player_advanced_stats(
    player_id: str,
    current_user: User = Depends(get_current_active_user),
    game_service = Depends(container.game_service_provider)
):
    """Get advanced statistics for specific player."""
    try:
        stats = game_service.get_player_stats(player_id)
        return stats
    except Exception as e:
        logger.error(f"Error getting player advanced stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/players/compare")
async def compare_players(
    player_ids: List[str],
    current_user: User = Depends(get_current_active_user),
    game_service = Depends(container.game_service_provider)
):
    """Compare statistics between multiple players."""
    try:
        comparison_data = {}
        for player_id in player_ids:
            stats = game_service.get_player_stats(player_id)
            comparison_data[player_id] = stats
        
        # Calculate comparative metrics
        comparison_metrics = {
            "win_rates": {
                player_id: data["win_rate"]
                for player_id, data in comparison_data.items()
            },
            "avg_shots_per_game": {
                player_id: data["average_shots_per_game"]
                for player_id, data in comparison_data.items()
            },
            "shot_type_distribution": {
                player_id: data["shot_types"]
                for player_id, data in comparison_data.items()
            }
        }
        
        return comparison_metrics
    except Exception as e:
        logger.error(f"Error comparing players: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/teams/{team_id}/performance")
async def get_team_performance(
    team_id: str,
    current_user: User = Depends(get_current_active_user),
    game_service = Depends(container.game_service_provider)
):
    """Get team performance statistics."""
    try:
        team_games = game_service.list_games({"team_id": team_id})
        
        if not team_games:
            raise HTTPException(status_code=404, detail="No games found for team")
        
        # Calculate team statistics
        total_games = len(team_games)
        total_wins = sum(1 for game in team_games if game.winner_team == team_id)
        total_rallies = sum(len(game.rallies) for game in team_games)
        total_shots = sum(
            len(shot)
            for game in team_games
            for rally in game.rallies
            for shot in rally.shots
        )
        
        # Calculate shot effectiveness
        shot_effectiveness = {}
        for game in team_games:
            for rally in game.rallies:
                for shot in rally.shots:
                    shot_type = shot.shot_type.value
                    if shot_type not in shot_effectiveness:
                        shot_effectiveness[shot_type] = []
                    shot_effectiveness[shot_type].append(shot.effectiveness_score)
        
        avg_effectiveness = {
            shot_type: sum(scores) / len(scores)
            for shot_type, scores in shot_effectiveness.items()
        }
        
        return {
            "total_games": total_games,
            "win_rate": total_wins / total_games if total_games > 0 else 0,
            "avg_rallies_per_game": total_rallies / total_games if total_games > 0 else 0,
            "avg_shots_per_game": total_shots / total_games if total_games > 0 else 0,
            "shot_effectiveness": avg_effectiveness
        }
    except Exception as e:
        logger.error(f"Error getting team performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis/trends")
async def get_performance_trends(
    player_id: str,
    time_period: str = "month",
    current_user: User = Depends(get_current_active_user),
    game_service = Depends(container.game_service_provider)
):
    """Get performance trends over time for a player."""
    try:
        # Get player's games
        games = game_service.list_games({"player_id": player_id})
        
        if not games:
            raise HTTPException(status_code=404, detail="No games found for player")
        
        # Sort games by date
        games.sort(key=lambda x: x.date)
        
        # Calculate trends
        trends = {
            "win_rate": [],
            "avg_shots_per_game": [],
            "shot_effectiveness": {},
            "dates": []
        }
        
        for game in games:
            trends["dates"].append(game.date.isoformat())
            
            # Calculate win rate
            player_team = next(
                (team for team in game.teams if any(p.id == player_id for p in team.players)),
                None
            )
            if player_team:
                team_wins = len([r for r in game.rallies if r.winner_team == player_team.id])
                trends["win_rate"].append(team_wins / len(game.rallies) if game.rallies else 0)
            
            # Calculate average shots
            player_shots = sum(
                1 for rally in game.rallies
                for shot in rally.shots
                if shot.player_id == player_id
            )
            trends["avg_shots_per_game"].append(player_shots)
            
            # Calculate shot effectiveness
            for rally in game.rallies:
                for shot in rally.shots:
                    if shot.player_id == player_id:
                        shot_type = shot.shot_type.value
                        if shot_type not in trends["shot_effectiveness"]:
                            trends["shot_effectiveness"][shot_type] = []
                        trends["shot_effectiveness"][shot_type].append(shot.effectiveness_score)
        
        return trends
    except Exception as e:
        logger.error(f"Error getting performance trends: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis/shot-effectiveness")
async def get_shot_effectiveness(
    game_id: str,
    player_id: str = None,
    current_user: User = Depends(get_current_active_user),
    game_service = Depends(container.game_service_provider)
):
    """Get shot effectiveness analysis for a game."""
    try:
        game = game_service.get_game(game_id)
        if not game or game.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Game not found")
        
        # Get shots data
        shots_data = {
            "shots": [
                {
                    "player_id": shot.player_id,
                    "shot_type": shot.shot_type.value,
                    "placement_x": shot.placement.x,
                    "placement_y": shot.placement.y,
                    "speed": shot.speed,
                    "spin": shot.spin,
                    "effectiveness_score": shot.effectiveness_score
                }
                for rally in game.rallies
                for shot in rally.shots
            ]
        }
        
        # Get visualizations
        visualizations = game_service.get_game_visualizations(game_id)
        return {
            "shot_effectiveness_heatmap": visualizations.get("shot_effectiveness_heatmap"),
            "shot_speed_analysis": visualizations.get("shot_speed_analysis"),
            "shot_spin_analysis": visualizations.get("shot_spin_analysis")
        }
    except Exception as e:
        logger.error(f"Error getting shot effectiveness: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis/rally-patterns")
async def get_rally_patterns(
    game_id: str,
    current_user: User = Depends(get_current_active_user),
    game_service = Depends(container.game_service_provider)
):
    """Get rally pattern analysis for a game."""
    try:
        game = game_service.get_game(game_id)
        if not game or game.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Game not found")
        
        # Get rallies and shots data
        game_data = {
            "rallies": [
                {
                    "id": rally.id,
                    "duration": rally.duration,
                    "winner_team": rally.winner_team,
                    "ending_type": rally.ending_type
                }
                for rally in game.rallies
            ],
            "shots": [
                {
                    "rally_id": shot.rally_id,
                    "player_id": shot.player_id,
                    "shot_type": shot.shot_type.value,
                    "placement_x": shot.placement.x,
                    "placement_y": shot.placement.y
                }
                for rally in game.rallies
                for shot in rally.shots
            ]
        }
        
        # Get visualizations
        visualizations = game_service.get_game_visualizations(game_id)
        return {
            "rally_pattern_analysis": visualizations.get("rally_pattern_analysis")
        }
    except Exception as e:
        logger.error(f"Error getting rally patterns: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis/player-positions")
async def get_player_positions(
    game_id: str,
    player_id: str = None,
    current_user: User = Depends(get_current_active_user),
    game_service = Depends(container.game_service_provider)
):
    """Get player position analysis for a game."""
    try:
        game = game_service.get_game(game_id)
        if not game or game.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Game not found")
        
        # Get player positions data
        positions_data = {
            "player_positions": [
                {
                    "player_id": pos.player_id,
                    "x": pos.position.x,
                    "y": pos.position.y,
                    "timestamp": pos.timestamp.isoformat(),
                    "speed": pos.speed
                }
                for pos in game.player_positions
            ]
        }
        
        # Get visualizations
        visualizations = game_service.get_game_visualizations(game_id)
        return {
            "player_position_heatmap": visualizations.get("player_position_heatmap")
        }
    except Exception as e:
        logger.error(f"Error getting player positions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis/shot-metrics")
async def get_shot_metrics(
    game_id: str,
    player_id: str = None,
    current_user: User = Depends(get_current_active_user),
    game_service = Depends(container.game_service_provider)
):
    """Get detailed shot metrics analysis for a game."""
    try:
        game = game_service.get_game(game_id)
        if not game or game.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Game not found")
        
        # Calculate shot metrics
        shots = [
            shot
            for rally in game.rallies
            for shot in rally.shots
        ]
        
        if player_id:
            shots = [s for s in shots if s.player_id == player_id]
        
        # Calculate metrics by shot type
        shot_metrics = {}
        for shot in shots:
            shot_type = shot.shot_type.value
            if shot_type not in shot_metrics:
                shot_metrics[shot_type] = {
                    "count": 0,
                    "total_speed": 0,
                    "total_spin": 0,
                    "total_effectiveness": 0,
                    "successful_shots": 0
                }
            
            metrics = shot_metrics[shot_type]
            metrics["count"] += 1
            metrics["total_speed"] += shot.speed
            metrics["total_spin"] += shot.spin
            metrics["total_effectiveness"] += shot.effectiveness_score
            if shot.effectiveness_score > 0.7:  # Consider shots with high effectiveness as successful
                metrics["successful_shots"] += 1
        
        # Calculate averages
        for metrics in shot_metrics.values():
            if metrics["count"] > 0:
                metrics["avg_speed"] = metrics["total_speed"] / metrics["count"]
                metrics["avg_spin"] = metrics["total_spin"] / metrics["count"]
                metrics["avg_effectiveness"] = metrics["total_effectiveness"] / metrics["count"]
                metrics["success_rate"] = metrics["successful_shots"] / metrics["count"]
            
            # Remove total fields
            del metrics["total_speed"]
            del metrics["total_spin"]
            del metrics["total_effectiveness"]
        
        return {
            "shot_metrics": shot_metrics,
            "total_shots": len(shots),
            "player_id": player_id
        }
    except Exception as e:
        logger.error(f"Error getting shot metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analysis/advanced/shot-patterns")
async def analyze_shot_patterns(
    game_id: str,
    window_size: int = 5,
    background_tasks: BackgroundTasks = None
) -> Dict[str, Any]:
    """Analyze shot patterns using GPU acceleration."""
    try:
        # Get game data
        game = container.game_service().get_game(game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        # Extract shots
        shots = [
            {
                "placement_x": shot.placement.x,
                "placement_y": shot.placement.y,
                "speed": shot.speed,
                "spin": shot.spin,
                "effectiveness_score": shot.effectiveness_score
            }
            for rally in game.rallies
            for shot in rally.shots
        ]
        
        # Analyze patterns
        patterns = gpu_analyzer.analyze_shot_patterns(shots, window_size)
        
        return {
            "game_id": game_id,
            "timestamp": datetime.now().isoformat(),
            "patterns": patterns
        }
    except Exception as e:
        logger.error(f"Error analyzing shot patterns: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analysis/advanced/player-movement")
async def analyze_player_movement(
    game_id: str,
    player_id: str,
    time_window: float = 1.0
) -> Dict[str, Any]:
    """Analyze player movement patterns using GPU acceleration."""
    try:
        # Get game data
        game = container.game_service().get_game(game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        # Extract positions
        positions = [
            {
                "x": pos.x,
                "y": pos.y,
                "speed": pos.speed,
                "timestamp": pos.timestamp
            }
            for rally in game.rallies
            for pos in rally.player_positions
            if pos.player_id == player_id
        ]
        
        # Analyze movement
        movement = gpu_analyzer.analyze_player_movement(positions, time_window)
        
        return {
            "game_id": game_id,
            "player_id": player_id,
            "timestamp": datetime.now().isoformat(),
            "movement": movement
        }
    except Exception as e:
        logger.error(f"Error analyzing player movement: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analysis/advanced/rally-dynamics")
async def analyze_rally_dynamics(
    game_id: str
) -> Dict[str, Any]:
    """Analyze rally dynamics using GPU acceleration."""
    try:
        # Get game data
        game = container.game_service().get_game(game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        
        # Extract rallies and shots
        rallies = [
            {
                "id": rally.id,
                "duration": rally.duration,
                "winner_team": rally.winner_team
            }
            for rally in game.rallies
        ]
        
        shots = [
            {
                "rally_id": shot.rally_id,
                "placement_x": shot.placement.x,
                "placement_y": shot.placement.y,
                "speed": shot.speed,
                "spin": shot.spin,
                "effectiveness_score": shot.effectiveness_score
            }
            for rally in game.rallies
            for shot in rally.shots
        ]
        
        # Analyze dynamics
        dynamics = gpu_analyzer.analyze_rally_dynamics(rallies, shots)
        
        return {
            "game_id": game_id,
            "timestamp": datetime.now().isoformat(),
            "dynamics": dynamics
        }
    except Exception as e:
        logger.error(f"Error analyzing rally dynamics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analysis/ml/train-shot-predictor")
async def train_shot_predictor(
    game_ids: List[str],
    epochs: int = 100,
    batch_size: int = 32
) -> Dict[str, Any]:
    """Train shot outcome predictor using ML."""
    try:
        # Get games data
        games = [
            container.game_service().get_game(game_id)
            for game_id in game_ids
        ]
        
        # Extract shots
        shots = []
        for game in games:
            if not game:
                continue
            
            for rally in game.rallies:
                for shot in rally.shots:
                    shots.append({
                        "placement_x": shot.placement.x,
                        "placement_y": shot.placement.y,
                        "speed": shot.speed,
                        "spin": shot.spin,
                        "player_position_x": shot.player_position.x,
                        "player_position_y": shot.player_position.y,
                        "opponent_position_x": shot.opponent_position.x,
                        "success": shot.success
                    })
        
        # Train model
        training_results = ml_analyzer.train_shot_predictor(
            shots,
            epochs=epochs,
            batch_size=batch_size
        )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "training_results": training_results
        }
    except Exception as e:
        logger.error(f"Error training shot predictor: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analysis/ml/predict-shot")
async def predict_shot_outcome(
    shot_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Predict shot outcome using trained ML model."""
    try:
        # Make prediction
        prediction = ml_analyzer.predict_shot_outcome(shot_data)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction
        }
    except Exception as e:
        logger.error(f"Error predicting shot outcome: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analysis/ml/player-style")
async def analyze_player_style(
    player_id: str,
    game_ids: List[str]
) -> Dict[str, Any]:
    """Analyze player's playing style using ML."""
    try:
        # Get games data
        games = [
            container.game_service().get_game(game_id)
            for game_id in game_ids
        ]
        
        # Extract player shots
        player_shots = []
        for game in games:
            if not game:
                continue
            
            for rally in game.rallies:
                for shot in rally.shots:
                    if shot.player_id == player_id:
                        player_shots.append({
                            "placement_x": shot.placement.x,
                            "placement_y": shot.placement.y,
                            "speed": shot.speed,
                            "spin": shot.spin,
                            "player_position_x": shot.player_position.x,
                            "player_position_y": shot.player_position.y,
                            "opponent_position_x": shot.opponent_position.x,
                            "success": shot.success
                        })
        
        # Analyze style
        style = ml_analyzer.analyze_player_style(player_shots)
        
        return {
            "player_id": player_id,
            "timestamp": datetime.now().isoformat(),
            "style": style
        }
    except Exception as e:
        logger.error(f"Error analyzing player style: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stream/start")
async def start_analysis_stream(
    stream_id: str,
    data_source: Dict[str, Any],
    analysis_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Start a new analysis stream."""
    try:
        # Create data source function
        def get_data() -> Dict[str, Any]:
            # Implement data source logic
            return data_source
        
        # Start stream
        await stream_analyzer.start_stream(
            stream_id,
            get_data,
            analysis_config
        )
        
        return {
            "stream_id": stream_id,
            "status": "started",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error starting stream {stream_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stream/stop")
async def stop_analysis_stream(
    stream_id: str
) -> Dict[str, Any]:
    """Stop an analysis stream."""
    try:
        # Stop stream
        await stream_analyzer.stop_stream(stream_id)
        
        return {
            "stream_id": stream_id,
            "status": "stopped",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error stopping stream {stream_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stream/status/{stream_id}")
async def get_stream_status(
    stream_id: str
) -> Dict[str, Any]:
    """Get status of an analysis stream."""
    try:
        # Get status
        status = stream_analyzer.get_stream_status(stream_id)
        
        return {
            "stream_id": stream_id,
            "timestamp": datetime.now().isoformat(),
            "status": status
        }
    except Exception as e:
        logger.error(f"Error getting stream status for {stream_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stream/active")
async def get_active_streams() -> Dict[str, Any]:
    """Get list of active streams."""
    try:
        # Get active streams
        streams = stream_analyzer.get_active_streams()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "active_streams": streams
        }
    except Exception as e:
        logger.error(f"Error getting active streams: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def start():
    """Start the FastAPI server."""
    uvicorn.run(
        "pickleball_vision.web.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

if __name__ == "__main__":
    start() 