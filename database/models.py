"""SQLAlchemy database models."""
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class ConversationDB(Base):
    """Conversation database model."""
    __tablename__ = "conversations"
    
    id = Column(String, primary_key=True)
    prospect_id = Column(String, nullable=True)
    started_at = Column(DateTime, default=datetime.now)
    ended_at = Column(DateTime, nullable=True)
    status = Column(String, default="active")
    outcome = Column(String, nullable=True)
    
    # Relationships
    messages = relationship("MessageDB", back_populates="conversation", cascade="all, delete-orphan")
    agent_invocations = relationship("AgentInvocationDB", back_populates="conversation", cascade="all, delete-orphan")
    orders = relationship("OrderDB", back_populates="conversation", cascade="all, delete-orphan")


class MessageDB(Base):
    """Message database model."""
    __tablename__ = "messages"
    
    id = Column(String, primary_key=True)
    conversation_id = Column(String, ForeignKey("conversations.id"))
    role = Column(String)
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.now)
    meta_data = Column(Text, nullable=True)  # JSON string
    
    # Relationships
    conversation = relationship("ConversationDB", back_populates="messages")


class AgentInvocationDB(Base):
    """Agent invocation database model."""
    __tablename__ = "agent_invocations"
    
    id = Column(String, primary_key=True)
    conversation_id = Column(String, ForeignKey("conversations.id"))
    agent_name = Column(String)
    invoked_at = Column(DateTime, default=datetime.now)
    completed_at = Column(DateTime, nullable=True)
    duration_ms = Column(Integer, nullable=True)
    status = Column(String)
    result = Column(Text, nullable=True)  # JSON string
    error = Column(Text, nullable=True)
    
    # Relationships
    conversation = relationship("ConversationDB", back_populates="agent_invocations")
    tool_calls = relationship("ToolCallDB", back_populates="agent_invocation", cascade="all, delete-orphan")


class ToolCallDB(Base):
    """Tool call database model."""
    __tablename__ = "tool_calls"
    
    id = Column(String, primary_key=True)
    agent_invocation_id = Column(String, ForeignKey("agent_invocations.id"))
    tool_name = Column(String)
    tool_type = Column(String)
    input_data = Column(Text)  # JSON string
    output_data = Column(Text, nullable=True)  # JSON string
    called_at = Column(DateTime, default=datetime.now)
    duration_ms = Column(Integer, nullable=True)
    status = Column(String)
    error = Column(Text, nullable=True)
    
    # Relationships
    agent_invocation = relationship("AgentInvocationDB", back_populates="tool_calls")


class OrderDB(Base):
    """Order database model."""
    __tablename__ = "orders"
    
    id = Column(String, primary_key=True)
    conversation_id = Column(String, ForeignKey("conversations.id"))
    prospect_id = Column(String)
    products = Column(Text)  # JSON string
    total_amount = Column(Float)
    status = Column(String)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    conversation = relationship("ConversationDB", back_populates="orders")


class AnalyticsEventDB(Base):
    """Analytics event database model."""
    __tablename__ = "analytics_events"
    
    id = Column(String, primary_key=True)
    event_type = Column(String)
    event_data = Column(Text)  # JSON string
    timestamp = Column(DateTime, default=datetime.now)
