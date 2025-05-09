"""Dependency injection container."""
from dependency_injector import containers, providers
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

from .config.settings import DATABASE_URL, DATABASE_POOL_SIZE, DATABASE_MAX_OVERFLOW
from .repositories.game_repository import GameRepository
from .services.game_service import GameService
from .analytics.strategy_analyzer import StrategyAnalyzer
from .visualization.advanced_visualizer import AdvancedVisualizer

class Container(containers.DeclarativeContainer):
    """Application container."""
    
    # Configuration
    config = providers.Configuration()
    
    # Database
    engine = providers.Singleton(
        create_engine,
        DATABASE_URL,
        pool_size=DATABASE_POOL_SIZE,
        max_overflow=DATABASE_MAX_OVERFLOW
    )
    
    session_factory = providers.Singleton(
        sessionmaker,
        bind=engine
    )
    
    # Repositories
    game_repository = providers.Factory(
        GameRepository,
        session=session_factory
    )
    
    # Services
    strategy_analyzer = providers.Singleton(
        StrategyAnalyzer
    )
    
    advanced_visualizer = providers.Singleton(
        AdvancedVisualizer
    )
    
    game_service = providers.Factory(
        GameService,
        game_repository=game_repository,
        strategy_analyzer=strategy_analyzer,
        advanced_visualizer=advanced_visualizer
    )
    
    # API dependencies
    def get_db_session():
        """Get database session."""
        session = session_factory()
        try:
            yield session
        finally:
            session.close()
    
    db_session = providers.Resource(
        get_db_session
    )
    
    def get_game_service():
        """Get game service."""
        return game_service()
    
    game_service_provider = providers.Resource(
        get_game_service
    ) 