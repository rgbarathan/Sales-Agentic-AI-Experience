"""Base Policy Agent class for RAG-based knowledge agents."""
import asyncio
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

from shared.models import A2AMessage
from shared.protocols import a2a_protocol
from shared.context_loader import context_loader
from rag.rag_manager import rag_manager
import structlog

logger = structlog.get_logger()


class PolicyAgent(ABC):
    """
    Base class for Policy Agents.
    
    Policy Agents use RAG to retrieve information from policy documents
    and provide knowledge to operational agents via A2A Protocol.
    """
    
    def __init__(self, agent_name: str, document_collection: str):
        """
        Initialize a Policy Agent.
        
        Args:
            agent_name: Name of the agent (e.g., "product_policy_agent")
            document_collection: Name of the RAG collection
        """
        self.agent_name = agent_name
        self.document_collection = document_collection
        
        # Load agent context
        try:
            self.context = context_loader.load_context(agent_name)
            self.system_prompt = context_loader.build_system_prompt(agent_name)
            logger.info(
                "policy_agent_context_loaded",
                agent_name=agent_name
            )
        except FileNotFoundError:
            # Context file doesn't exist yet, use defaults
            logger.warning(
                "policy_agent_context_not_found",
                agent_name=agent_name,
                message="Using default configuration"
            )
            self.context = None
            self.system_prompt = f"You are a {agent_name.replace('_', ' ').title()}."
        
        # Register with A2A protocol
        a2a_protocol.register_agent(agent_name, self.handle_a2a_message)
        
        logger.info(
            "policy_agent_initialized",
            agent_name=agent_name,
            collection=document_collection
        )
    
    async def query_policy(
        self,
        question: str,
        n_results: int = 3
    ) -> str:
        """
        Query policy documents using RAG.
        
        Args:
            question: The policy question
            n_results: Number of results to retrieve
        
        Returns:
            Formatted context string with policy information
        """
        logger.info(
            "policy_query",
            agent=self.agent_name,
            question=question
        )
        
        # Use RAG to get context
        context = rag_manager.get_context(
            agent_name=self.document_collection,
            query_text=question,
            n_results=n_results
        )
        
        if not context:
            logger.warning(
                "policy_query_no_results",
                agent=self.agent_name,
                question=question
            )
            return self._format_no_results_response(question)
        
        logger.info(
            "policy_query_success",
            agent=self.agent_name,
            context_length=len(context)
        )
        
        return context
    
    async def handle_a2a_message(self, message: A2AMessage) -> Optional[A2AMessage]:
        """
        Handle incoming A2A messages.
        
        Args:
            message: Incoming A2A message
        
        Returns:
            Response message or None
        """
        logger.info(
            "policy_agent_message_received",
            agent=self.agent_name,
            from_agent=message.from_agent,
            message_type=message.message_type
        )
        
        if message.message_type == "request":
            # Extract question from payload
            question = message.payload.get("question", "")
            n_results = message.payload.get("n_results", 3)
            
            if not question:
                # Return error response
                await a2a_protocol.send_response(
                    original_message=message,
                    payload={"error": "No question provided"},
                    success=False
                )
                return None
            
            # Query policy documents
            context = await self.query_policy(question, n_results)
            
            # Send response
            await a2a_protocol.send_response(
                original_message=message,
                payload={
                    "question": question,
                    "context": context,
                    "agent": self.agent_name,
                    "source": self.document_collection
                },
                success=True
            )
        
        return None
    
    def _format_no_results_response(self, question: str) -> str:
        """Format a response when no policy information is found."""
        return f"""# No Policy Information Found

The question "{question}" did not match any information in the {self.agent_name.replace('_', ' ').title()} knowledge base.

## Suggestions:
- Try rephrasing the question
- Check if this falls under a different policy area
- Contact the policy team for clarification

## Alternative Sources:
- Review the complete policy document manually
- Consult with a subject matter expert
- Check for recent policy updates
"""
    
    @abstractmethod
    def get_policy_summary(self) -> str:
        """
        Get a summary of what policies this agent covers.
        
        Returns:
            Summary string
        """
        pass


class ProductPolicyAgent(PolicyAgent):
    """
    Product Policy Agent.
    
    Provides information about:
    - Product catalog and specifications
    - Pricing rules and tiers
    - Volume and bundle discounts
    - Promotional offers
    - SLAs and installation timelines
    """
    
    def __init__(self):
        super().__init__(
            agent_name="product_policy_agent",
            document_collection="product_policy_agent"
        )
    
    def get_policy_summary(self) -> str:
        return """
        Product Policy Agent covers:
        - Business Internet services (100 Mbps, 500 Mbps, 1 Gbps)
        - Business Voice services (Basic, Pro)
        - Managed Services (WiFi, Security, Cloud Backup)
        - Pricing rules and discount structures
        - Promotional offers and codes
        - Service Level Agreements
        - Installation fees and timelines
        """


class OrderPolicyAgent(PolicyAgent):
    """
    Order Policy Agent.
    
    Provides information about:
    - Order processing procedures
    - Validation requirements
    - Modification and cancellation policies
    - Terms and conditions
    - Order status definitions
    - Escalation procedures
    """
    
    def __init__(self):
        super().__init__(
            agent_name="order_policy_agent",
            document_collection="order_policy_agent"
        )
    
    def get_policy_summary(self) -> str:
        return """
        Order Policy Agent covers:
        - Pre-order requirements and documentation
        - Order validation checklist
        - Order processing workflow
        - Modification policies and restrictions
        - Cancellation policies and fees
        - Terms and conditions
        - Order status definitions
        - Escalation procedures
        - Quality assurance metrics
        """


class ServicePolicyAgent(PolicyAgent):
    """
    Service Policy Agent.
    
    Provides information about:
    - Network coverage areas
    - Serviceability criteria
    - Service Level Agreements
    - Response time commitments
    - Network performance standards
    - Installation requirements
    """
    
    def __init__(self):
        super().__init__(
            agent_name="service_policy_agent",
            document_collection="service_policy_agent"
        )
    
    def get_policy_summary(self) -> str:
        return """
        Service Policy Agent covers:
        - Fiber network coverage areas
        - Serviceability check process and criteria
        - Uptime guarantees (99.9% and 99.99%)
        - Response time SLAs
        - Network performance standards
        - Installation requirements and fees
        - Service limitations and restrictions
        - Disaster recovery and business continuity
        """


class FulfillmentPolicyAgent(PolicyAgent):
    """
    Fulfillment Policy Agent.
    
    Provides information about:
    - Equipment inventory catalog
    - Installation procedures
    - Scheduling processes
    - Quality assurance standards
    - Troubleshooting guides
    - Technician standards
    """
    
    def __init__(self):
        super().__init__(
            agent_name="fulfillment_policy_agent",
            document_collection="fulfillment_policy_agent"
        )
    
    def get_policy_summary(self) -> str:
        return """
        Fulfillment Policy Agent covers:
        - Equipment catalog (ONTs, routers, phones, WiFi APs, firewalls)
        - Inventory management and availability
        - Installation scheduling process
        - Installation procedures for all services
        - Quality assurance checklists
        - Troubleshooting common issues
        - Equipment return and swap procedures
        - Technician standards and training
        - Performance metrics
        """
