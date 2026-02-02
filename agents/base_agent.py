"""Base Agent class for all operational agents."""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import structlog

from shared.models import A2AMessage
from shared.protocols import a2a_protocol
from shared.context_loader import context_loader

logger = structlog.get_logger()


class BaseAgent(ABC):
    """
    Base class for all operational agents.
    
    Provides common functionality:
    - Context loading from YAML
    - A2A protocol registration
    - System prompt generation
    - Logging
    """
    
    def __init__(self, agent_name: str):
        """
        Initialize a base agent.
        
        Args:
            agent_name: Name of the agent (e.g., "prospect_agent")
        """
        self.agent_name = agent_name
        
        # Load agent context
        try:
            self.context = context_loader.load_context(agent_name)
            self.system_prompt = context_loader.build_system_prompt(agent_name)
            logger.info(
                "agent_context_loaded",
                agent_name=agent_name
            )
        except FileNotFoundError:
            logger.warning(
                "agent_context_not_found",
                agent_name=agent_name,
                message="Using default configuration"
            )
            self.context = None
            self.system_prompt = f"You are a {agent_name.replace('_', ' ').title()}."
        
        # Register with A2A protocol
        a2a_protocol.register_agent(agent_name, self.handle_a2a_message)
        
        logger.info(
            "agent_initialized",
            agent_name=agent_name,
            framework=self.get_framework()
        )
    
    @abstractmethod
    def get_framework(self) -> str:
        """
        Get the framework this agent uses.
        
        Returns:
            Framework name (e.g., "ADK", "Strands", "LangGraph")
        """
        pass
    
    @abstractmethod    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input and return output.
        Must be implemented by subclasses.
        
        Args:
            input_data: Input data for processing
        
        Returns:
            Output data with agent_activity metadata
        """
        raise NotImplementedError("Subclasses must implement process method")
    
    def _create_response(
        self,
        message: str,
        sub_agents_invoked: List[str] = None,
        communication_methods: List[str] = None,
        tools_used: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a standardized response with agent activity metadata.
        
        Args:
            message: Response message
            sub_agents_invoked: List of sub-agents that were called
            communication_methods: Communication methods used (A2A, MCP, REST, etc.)
            tools_used: Tools/services used (RAG, ChromaDB, APIs, etc.)
            **kwargs: Additional response fields
        
        Returns:
            Response dictionary with agent_activity metadata
        """
        response = {
            "message": message,
            "agent_activity": {
                "primary_agent": self.agent_name,
                "sub_agents_invoked": sub_agents_invoked or [],
                "communication_methods": communication_methods or [],
                "tools_used": tools_used or []
            }
        }
        
        # Add any additional fields
        response.update(kwargs)
        
        return response
    
    def _add_agent_tag(self, message: str, tag: str = None) -> str:
        """
        Add agent identification tag to message.
        
        Args:
            message: Original message
            tag: Custom tag (if None, uses agent name)
        
        Returns:
            Message with agent tag prepended
        """
        if tag is None:
            # Convert agent_name to readable format
            tag = self.agent_name.replace("_", " ").title()
        
        return f"🤖 [{tag}]\\n\\n{message}"
    
    async def handle_a2a_message(self, message: A2AMessage) -> Optional[A2AMessage]:
        """
        Handle incoming A2A messages.
        
        Args:
            message: Incoming A2A message
        
        Returns:
            Response message or None
        """
        logger.info(
            "agent_message_received",
            agent=self.agent_name,
            from_agent=message.from_agent,
            message_type=message.message_type
        )
        
        if message.message_type == "request":
            try:
                # Process the request
                result = await self.process(message.payload)
                
                # Send response
                await a2a_protocol.send_response(
                    original_message=message,
                    payload=result,
                    success=True
                )
            except Exception as e:
                logger.error(
                    "agent_processing_error",
                    agent=self.agent_name,
                    error=str(e)
                )
                # Send error response
                await a2a_protocol.send_response(
                    original_message=message,
                    payload={"error": str(e)},
                    success=False
                )
        
        return None
    
    def get_connected_agents(self) -> List[str]:
        """
        Get list of connected agent names.
        
        Returns:
            List of agent names this agent can communicate with
        """
        if not self.context:
            return []
        
        return [agent.name for agent in self.context.connected_agents]
    
    def get_tools(self) -> Dict[str, List[str]]:
        """
        Get available tools for this agent.
        
        Returns:
            Dictionary of tool types to tool names
        """
        if not self.context:
            return {}
        
        return self.context.connected_tools
