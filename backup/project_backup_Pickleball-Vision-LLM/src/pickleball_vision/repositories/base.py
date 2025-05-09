"""Base repository for data access."""
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import select, update, delete
from ..domain.models import BaseModel

T = TypeVar('T', bound=BaseModel)

class BaseRepository(Generic[T], ABC):
    """Base repository class for data access operations."""
    
    def __init__(self, session: Session):
        """Initialize repository with database session."""
        self.session = session
    
    @abstractmethod
    def get(self, id: str) -> Optional[T]:
        """Get entity by ID."""
        pass
    
    @abstractmethod
    def list(self, filters: Dict[str, Any] = None) -> List[T]:
        """List entities with optional filters."""
        pass
    
    @abstractmethod
    def create(self, entity: T) -> T:
        """Create new entity."""
        pass
    
    @abstractmethod
    def update(self, id: str, entity: T) -> T:
        """Update existing entity."""
        pass
    
    @abstractmethod
    def delete(self, id: str) -> bool:
        """Delete entity by ID."""
        pass
    
    def commit(self):
        """Commit changes to database."""
        self.session.commit()
    
    def rollback(self):
        """Rollback changes."""
        self.session.rollback()
    
    def close(self):
        """Close database session."""
        self.session.close() 