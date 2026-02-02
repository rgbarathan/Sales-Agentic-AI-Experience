"""A2A (Agent-to-Agent) Protocol implementation."""
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Callable, Awaitable
from .models import A2AMessage
import structlog

logger = structlog.get_logger()


class A2AProtocol:
    """
    Agent-to-Agent communication protocol.
    
    Enables agents to communicate with each other asynchronously
    using a message-based protocol.
    """
    
    def __init__(self):
        """Initialize the A2A protocol."""
        self._agents: Dict[str, Callable[[A2AMessage], Awaitable[Optional[A2AMessage]]]] = {}
        self._pending_responses: Dict[str, asyncio.Future] = {}
        self._message_history: list[A2AMessage] = []
    
    def register_agent(
        self,
        agent_name: str,
        handler: Callable[[A2AMessage], Awaitable[Optional[A2AMessage]]]
    ):
        """
        Register an agent with the protocol.
        
        Args:
            agent_name: Name of the agent
            handler: Async function to handle incoming messages
        """
        self._agents[agent_name] = handler
        logger.info("agent_registered", agent_name=agent_name)
    
    def unregister_agent(self, agent_name: str):
        """Unregister an agent from the protocol."""
        if agent_name in self._agents:
            del self._agents[agent_name]
            logger.info("agent_unregistered", agent_name=agent_name)
    
    @property
    def agents(self) -> Dict[str, Callable]:
        """Get dictionary of registered agents."""
        return self._agents
    
    async def send_message(
        self,
        from_agent: str,
        to_agent: str,
        message_type: str,
        payload: Dict[str, Any],
        conversation_id: Optional[str] = None,
        wait_for_response: bool = False,
        timeout: float = 30.0
    ) -> Optional[A2AMessage]:
        """
        Send a message from one agent to another.
        
        Args:
            from_agent: Name of the sending agent
            to_agent: Name of the receiving agent
            message_type: Type of message (request, response, notification, error)
            payload: Message payload
            conversation_id: Optional conversation ID for tracking
            wait_for_response: Whether to wait for a response
            timeout: Timeout in seconds if waiting for response
        
        Returns:
            Response message if wait_for_response is True, otherwise None
        """
        message_id = str(uuid.uuid4())
        
        message = A2AMessage(
            message_id=message_id,
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=message_type,
            payload=payload,
            conversation_id=conversation_id,
            timestamp=datetime.now()
        )
        
        # Store in history
        self._message_history.append(message)
        
        logger.info(
            "a2a_message_sent",
            message_id=message_id,
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=message_type
        )
        
        # Check if target agent is registered
        if to_agent not in self._agents:
            error_msg = f"Agent '{to_agent}' not registered"
            logger.error("a2a_agent_not_found", to_agent=to_agent)
            raise ValueError(error_msg)
        
        # If waiting for response, create a future
        if wait_for_response and message_type == "request":
            response_future = asyncio.Future()
            self._pending_responses[message_id] = response_future
        
        # Deliver message to target agent
        try:
            handler = self._agents[to_agent]
            response = await handler(message)
            
            # If there's a response, handle it
            if response:
                response.correlation_id = message_id
                self._message_history.append(response)
                
                # If someone is waiting for this response, fulfill the future
                if message_id in self._pending_responses:
                    self._pending_responses[message_id].set_result(response)
                    del self._pending_responses[message_id]
                
                return response
            
            # If waiting for response but handler didn't return one, wait for it
            if wait_for_response and message_type == "request":
                try:
                    response = await asyncio.wait_for(response_future, timeout=timeout)
                    return response
                except asyncio.TimeoutError:
                    logger.error("a2a_response_timeout", message_id=message_id)
                    if message_id in self._pending_responses:
                        del self._pending_responses[message_id]
                    raise TimeoutError(f"No response received within {timeout} seconds")
        
        except Exception as e:
            logger.error("a2a_message_delivery_failed", error=str(e), message_id=message_id)
            raise
        
        return None
    
    async def send_response(
        self,
        original_message: A2AMessage,
        payload: Dict[str, Any],
        success: bool = True
    ):
        """
        Send a response to a previous message.
        
        Args:
            original_message: The message being responded to
            payload: Response payload
            success: Whether the operation was successful
        """
        response = A2AMessage(
            message_id=str(uuid.uuid4()),
            from_agent=original_message.to_agent,
            to_agent=original_message.from_agent,
            message_type="response" if success else "error",
            payload=payload,
            conversation_id=original_message.conversation_id,
            correlation_id=original_message.message_id,
            timestamp=datetime.now()
        )
        
        self._message_history.append(response)
        
        # If there's a pending future for this response, fulfill it
        if original_message.message_id in self._pending_responses:
            self._pending_responses[original_message.message_id].set_result(response)
            del self._pending_responses[original_message.message_id]
        else:
            # Otherwise, deliver it normally
            if response.to_agent in self._agents:
                handler = self._agents[response.to_agent]
                await handler(response)
    
    def get_message_history(
        self,
        conversation_id: Optional[str] = None,
        agent_name: Optional[str] = None
    ) -> list[A2AMessage]:
        """
        Get message history, optionally filtered.
        
        Args:
            conversation_id: Filter by conversation ID
            agent_name: Filter by agent name (sent or received)
        
        Returns:
            List of messages
        """
        messages = self._message_history
        
        if conversation_id:
            messages = [m for m in messages if m.conversation_id == conversation_id]
        
        if agent_name:
            messages = [
                m for m in messages
                if m.from_agent == agent_name or m.to_agent == agent_name
            ]
        
        return messages


# Global A2A protocol instance
a2a_protocol = A2AProtocol()
