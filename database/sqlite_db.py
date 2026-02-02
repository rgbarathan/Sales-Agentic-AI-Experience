"""SQLite database manager."""
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import structlog

from .models import Base, ConversationDB, MessageDB, AgentInvocationDB, ToolCallDB, OrderDB, AnalyticsEventDB
from config.settings import settings

logger = structlog.get_logger()


class DatabaseManager:
    """Manages SQLite database operations."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or settings.sqlite_db_path
        
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create engine
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        logger.info("database_initialized", db_path=self.db_path)
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
    
    # Conversation methods
    def create_conversation(self, conversation_id: str, prospect_id: Optional[str] = None) -> ConversationDB:
        """Create a new conversation using direct SQL to ensure ID is set."""
        from sqlalchemy import text
        
        with self.get_session() as session:
            # Use direct SQL to ensure id is included in INSERT
            sql = text("""
                INSERT INTO conversations (id, prospect_id, started_at, status)
                VALUES (:id, :prospect_id, :started_at, :status)
            """)
            
            session.execute(sql, {
                "id": conversation_id,
                "prospect_id": prospect_id,
                "started_at": datetime.now(),
                "status": "active"
            })
            session.commit()
            
            # Now fetch it back using ORM
            conversation = session.query(ConversationDB).filter_by(id=conversation_id).first()
            logger.info("conversation_created", conversation_id=conversation_id)
            return conversation
    
    def get_conversation(self, conversation_id: str) -> Optional[ConversationDB]:
        """Get a conversation by ID."""
        with self.get_session() as session:
            return session.query(ConversationDB).filter_by(id=conversation_id).first()
    
    def update_conversation(
        self,
        conversation_id: str,
        status: Optional[str] = None,
        outcome: Optional[str] = None,
        ended_at: Optional[datetime] = None
    ):
        """Update a conversation."""
        with self.get_session() as session:
            conversation = session.query(ConversationDB).filter_by(id=conversation_id).first()
            if conversation:
                if status:
                    conversation.status = status
                if outcome:
                    conversation.outcome = outcome
                if ended_at:
                    conversation.ended_at = ended_at
                session.commit()
                logger.info("conversation_updated", conversation_id=conversation_id)
    
    # Message methods
    def add_message(
        self,
        message_id: str,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MessageDB:
        """Add a message to a conversation."""
        with self.get_session() as session:
            message = MessageDB(
                id=message_id,
                conversation_id=conversation_id,
                role=role,
                content=content,
                timestamp=datetime.now(),
                meta_data=json.dumps(metadata) if metadata else None
            )
            session.add(message)
            session.commit()
            session.refresh(message)
            return message
    
    def get_messages(self, conversation_id: str) -> List[MessageDB]:
        """Get all messages for a conversation."""
        with self.get_session() as session:
            return session.query(MessageDB).filter_by(conversation_id=conversation_id).order_by(MessageDB.timestamp).all()
    
    async def log_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Async wrapper for adding a message (for FastAPI compatibility).
        
        Args:
            conversation_id: Conversation ID
            role: Message role ('user' or 'assistant')
            content: Message content
            metadata: Optional metadata
        """
        from shared.utils import generate_id
        
        # Validate conversation_id
        if not conversation_id:
            raise ValueError("conversation_id cannot be None or empty")
        
        message_id = generate_id("MSG")
        
        # Create conversation if it doesn't exist
        if not self.get_conversation(conversation_id):
            logger.info("creating_new_conversation", conversation_id=conversation_id)
            self.create_conversation(conversation_id=conversation_id)
        
        return self.add_message(message_id, conversation_id, role, content, metadata)
    
    async def get_conversation_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Async wrapper for getting messages (for FastAPI compatibility).
        
        Returns messages as dictionaries.
        """
        messages = self.get_messages(conversation_id)
        return [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "meta_data": json.loads(msg.meta_data) if msg.meta_data else None
            }
            for msg in messages
        ]
    
    async def list_conversations(self) -> List[Dict[str, Any]]:
        """
        Async wrapper for listing conversations (for FastAPI compatibility).
        
        Returns conversations as dictionaries.
        """
        with self.get_session() as session:
            conversations = session.query(ConversationDB).all()
            return [
                {
                    "id": conv.id,
                    "prospect_id": conv.prospect_id,
                    "started_at": conv.started_at.isoformat(),
                    "ended_at": conv.ended_at.isoformat() if conv.ended_at else None,
                    "status": conv.status,
                    "outcome": conv.outcome
                }
                for conv in conversations
            ]
    
    async def initialize(self):
        """Async initialize method for compatibility (no-op since init happens in __init__)."""
        pass
    
    async def close(self):
        """Async close method for compatibility."""
        if hasattr(self, 'engine'):
            self.engine.dispose()
    
    # Agent invocation methods
    def create_agent_invocation(
        self,
        invocation_id: str,
        conversation_id: str,
        agent_name: str
    ) -> AgentInvocationDB:
        """Create an agent invocation record."""
        with self.get_session() as session:
            invocation = AgentInvocationDB(
                id=invocation_id,
                conversation_id=conversation_id,
                agent_name=agent_name,
                invoked_at=datetime.now(),
                status="running"
            )
            session.add(invocation)
            session.commit()
            session.refresh(invocation)
            return invocation
    
    def complete_agent_invocation(
        self,
        invocation_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        """Complete an agent invocation."""
        with self.get_session() as session:
            invocation = session.query(AgentInvocationDB).filter_by(id=invocation_id).first()
            if invocation:
                invocation.completed_at = datetime.now()
                invocation.status = status
                invocation.duration_ms = int((invocation.completed_at - invocation.invoked_at).total_seconds() * 1000)
                invocation.result = json.dumps(result) if result else None
                invocation.error = error
                session.commit()
    
    # Tool call methods
    def create_tool_call(
        self,
        call_id: str,
        agent_invocation_id: str,
        tool_name: str,
        tool_type: str,
        input_data: Dict[str, Any]
    ) -> ToolCallDB:
        """Create a tool call record."""
        with self.get_session() as session:
            tool_call = ToolCallDB(
                id=call_id,
                agent_invocation_id=agent_invocation_id,
                tool_name=tool_name,
                tool_type=tool_type,
                input_data=json.dumps(input_data),
                called_at=datetime.now(),
                status="running"
            )
            session.add(tool_call)
            session.commit()
            session.refresh(tool_call)
            return tool_call
    
    def complete_tool_call(
        self,
        call_id: str,
        status: str,
        output_data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        """Complete a tool call."""
        with self.get_session() as session:
            tool_call = session.query(ToolCallDB).filter_by(id=call_id).first()
            if tool_call:
                tool_call.status = status
                tool_call.duration_ms = int((datetime.now() - tool_call.called_at).total_seconds() * 1000)
                tool_call.output_data = json.dumps(output_data) if output_data else None
                tool_call.error = error
                session.commit()
    
    # Order methods
    def create_order(
        self,
        order_id: str,
        conversation_id: str,
        prospect_id: str,
        products: List[Dict[str, Any]],
        total_amount: float,
        status: str = "draft"
    ) -> OrderDB:
        """Create an order record."""
        with self.get_session() as session:
            order = OrderDB(
                id=order_id,
                conversation_id=conversation_id,
                prospect_id=prospect_id,
                products=json.dumps(products),
                total_amount=total_amount,
                status=status,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            session.add(order)
            session.commit()
            session.refresh(order)
            logger.info("order_created", order_id=order_id)
            return order
    
    def update_order_status(self, order_id: str, status: str):
        """Update order status."""
        with self.get_session() as session:
            order = session.query(OrderDB).filter_by(id=order_id).first()
            if order:
                order.status = status
                order.updated_at = datetime.now()
                session.commit()
                logger.info("order_status_updated", order_id=order_id, status=status)
    
    # Analytics methods
    def log_event(self, event_id: str, event_type: str, event_data: Dict[str, Any]):
        """Log an analytics event."""
        with self.get_session() as session:
            event = AnalyticsEventDB(
                id=event_id,
                event_type=event_type,
                event_data=json.dumps(event_data),
                timestamp=datetime.now()
            )
            session.add(event)
            session.commit()


# Global database manager instance
db_manager = DatabaseManager()

# Alias for compatibility
SQLiteDB = DatabaseManager
