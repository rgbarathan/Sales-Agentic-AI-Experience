"""Context loader for agent configurations."""
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import structlog

logger = structlog.get_logger()


class ToolConfig(BaseModel):
    """Tool configuration model."""
    name: str
    server: Optional[str] = None
    endpoint: Optional[str] = None
    usage: str
    when_to_use: str


class ConnectedAgent(BaseModel):
    """Connected agent configuration."""
    name: str
    communication: str
    purpose: str


class EscalationRule(BaseModel):
    """Escalation rule configuration."""
    condition: str
    action: str


class PersonalityConfig(BaseModel):
    """Personality configuration."""
    tone: str
    style: str


class DomainKnowledge(BaseModel):
    """Domain knowledge configuration."""
    industry: str
    expertise: List[str]
    business_rules: List[str]


class ExampleScenario(BaseModel):
    """Example scenario configuration."""
    scenario: str
    steps: List[str]


class AgentContext(BaseModel):
    """Agent context configuration model."""
    role: str
    personality: PersonalityConfig
    domain_knowledge: DomainKnowledge
    connected_agents: List[ConnectedAgent]
    connected_tools: Dict[str, List[ToolConfig]]
    escalation_rules: List[EscalationRule]
    success_metrics: List[str]
    example_scenarios: List[ExampleScenario]


class ContextLoader:
    """Load and manage agent context configurations."""
    
    def __init__(self, context_dir: str = "config/agent_contexts"):
        """
        Initialize the context loader.
        
        Args:
            context_dir: Directory containing agent context YAML files
        """
        self.context_dir = Path(context_dir)
        self._cache: Dict[str, AgentContext] = {}
    
    def load_context(self, agent_name: str) -> AgentContext:
        """
        Load context for a specific agent.
        
        Args:
            agent_name: Name of the agent (e.g., "prospect_agent")
        
        Returns:
            AgentContext object
        """
        # Check cache first
        if agent_name in self._cache:
            return self._cache[agent_name]
        
        # Load from file
        context_file = self.context_dir / f"{agent_name}_context.yaml"
        
        if not context_file.exists():
            logger.warning(
                "context_file_not_found",
                agent_name=agent_name,
                file=str(context_file)
            )
            raise FileNotFoundError(f"Context file not found: {context_file}")
        
        try:
            with open(context_file, 'r') as f:
                data = yaml.safe_load(f)
            
            # Parse into AgentContext model
            context = AgentContext(**data)
            
            # Cache it
            self._cache[agent_name] = context
            
            logger.info("context_loaded", agent_name=agent_name)
            return context
        
        except Exception as e:
            logger.error(
                "context_load_failed",
                agent_name=agent_name,
                error=str(e)
            )
            raise
    
    def build_system_prompt(self, agent_name: str) -> str:
        """
        Build a system prompt from agent context.
        
        Args:
            agent_name: Name of the agent
        
        Returns:
            System prompt string
        """
        context = self.load_context(agent_name)
        
        # Build comprehensive system prompt
        prompt_parts = [
            f"# Role\nYou are a {context.role}.",
            "",
            f"# Personality",
            f"- Tone: {context.personality.tone}",
            f"- Style: {context.personality.style}",
            "",
            f"# Domain Knowledge",
            f"Industry: {context.domain_knowledge.industry}",
            "",
            "Expertise:",
        ]
        
        for expertise in context.domain_knowledge.expertise:
            prompt_parts.append(f"- {expertise}")
        
        prompt_parts.extend([
            "",
            "Business Rules:",
        ])
        
        for rule in context.domain_knowledge.business_rules:
            prompt_parts.append(f"- {rule}")
        
        # Add connected agents
        if context.connected_agents:
            prompt_parts.extend([
                "",
                "# Connected Agents",
                "You can communicate with the following agents:",
            ])
            for agent in context.connected_agents:
                prompt_parts.append(
                    f"- **{agent.name}** ({agent.communication}): {agent.purpose}"
                )
        
        # Add tools
        if context.connected_tools:
            prompt_parts.extend([
                "",
                "# Available Tools",
            ])
            
            for tool_type, tools in context.connected_tools.items():
                prompt_parts.append(f"\n## {tool_type.replace('_', ' ').title()}")
                for tool in tools:
                    prompt_parts.append(f"\n### {tool.name}")
                    if tool.server:
                        prompt_parts.append(f"Server: {tool.server}")
                    if tool.endpoint:
                        prompt_parts.append(f"Endpoint: {tool.endpoint}")
                    prompt_parts.append(f"Usage: {tool.usage}")
                    prompt_parts.append(f"When to use: {tool.when_to_use}")
        
        # Add escalation rules
        if context.escalation_rules:
            prompt_parts.extend([
                "",
                "# Escalation Rules",
            ])
            for rule in context.escalation_rules:
                prompt_parts.append(f"- **If** {rule.condition}, **then** {rule.action}")
        
        # Add success metrics
        if context.success_metrics:
            prompt_parts.extend([
                "",
                "# Success Metrics",
                "Your performance is measured by:",
            ])
            for metric in context.success_metrics:
                prompt_parts.append(f"- {metric}")
        
        # Add example scenarios
        if context.example_scenarios:
            prompt_parts.extend([
                "",
                "# Example Scenarios",
            ])
            for scenario in context.example_scenarios:
                prompt_parts.append(f"\n## {scenario.scenario}")
                prompt_parts.append("Steps:")
                for step in scenario.steps:
                    prompt_parts.append(f"1. {step}")
        
        return "\n".join(prompt_parts)
    
    def get_tool_list(self, agent_name: str) -> List[Dict[str, Any]]:
        """
        Get list of tools for an agent.
        
        Args:
            agent_name: Name of the agent
        
        Returns:
            List of tool configurations
        """
        context = self.load_context(agent_name)
        tools = []
        
        for tool_type, tool_list in context.connected_tools.items():
            for tool in tool_list:
                tools.append({
                    "name": tool.name,
                    "type": tool_type,
                    "server": tool.server,
                    "endpoint": tool.endpoint,
                    "usage": tool.usage,
                    "when_to_use": tool.when_to_use
                })
        
        return tools
    
    def get_connected_agents(self, agent_name: str) -> List[str]:
        """
        Get list of agents this agent can communicate with.
        
        Args:
            agent_name: Name of the agent
        
        Returns:
            List of connected agent names
        """
        context = self.load_context(agent_name)
        return [agent.name for agent in context.connected_agents]


# Global context loader instance
context_loader = ContextLoader()
