"""Super Agent - Orchestrates all other agents using Gemini 3.0 Flash."""
import asyncio
import os
import structlog
from typing import Dict, Any, List, Optional
from datetime import datetime
from google import genai
from google.genai import types


from agents.base_agent import BaseAgent
from shared.protocols import a2a_protocol
from shared.utils import generate_id, get_timestamp
from config.settings import settings
import structlog

logger = structlog.get_logger()


class SuperAgent(BaseAgent):
    """
    Super Agent - Orchestrates all other agents and manages conversations.
    
    Framework: ADK with Gemini 3.0 Flash LLM
    Communication: A2A Protocol to all sub-agents
    
    Responsibilities:
    - Manage customer conversations
    - Use Gemini to understand intent and generate responses
    - Coordinate multi-agent workflows
    - Maintain conversation context
    - Strictly grounded to only use available agents/tools
    """
    
    def __init__(self):
        super().__init__("super_agent")
        logger.info("agent_context_loaded", agent_name="super_agent")
        
        # Initialize Gemini 3.0 Flash
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model_name = os.getenv("LLM_MODEL", "gemini-3-flash-preview")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        
        if self.api_key:
            try:
                # Use new genai.Client for Gemini 3.0
                self.client = genai.Client(api_key=self.api_key)
                
                # Configure thinking for Flash (optimized for speed)
                self.thinking_config = types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.MINIMAL  # Optimized for speed
                )
                
                logger.info("gemini_initialized", model=self.model_name)
            except Exception as e:
                logger.error("gemini_init_failed", error=str(e))
                self.client = None
        else:
            logger.warning("no_api_key", message="GOOGLE_API_KEY not set, using fallback mode")
            self.client = None
        
        self.conversations: Dict[str, Dict[str, Any]] = {}
        
        # Track conversation context for auto-triggering agents
        self.conversation_context: Dict[str, Dict[str, Any]] = {}
        
        # Define available agents and their capabilities
        self.available_agents = {
            "product_policy_agent": "Product information, pricing, features (uses RAG)",
            "order_policy_agent": "Order policies, discounts, terms (uses RAG)",
            "service_policy_agent": "Service level agreements, support (uses RAG)",
            "fulfillment_policy_agent": "Installation, equipment policies (uses RAG)",
            "prospect_agent": "Qualify business prospects (uses MCP - CRM)",
            "lead_generation_agent": "Score and enrich leads (uses REST API)",
            "serviceability_agent": "Check service availability (uses MCP - Network)",
            "address_validation_agent": "Validate addresses (uses LangGraph + API)",
            "offer_agent": "Generate personalized quotes",
            "order_agent": "Process orders",
            "post_order_communication_agent": "Send confirmations (uses REST - Email/SMS)",
            "fulfillment_agent": "Schedule installation (async, uses MCP)",
            "service_activation_agent": "Activate services (async, uses MCP)",
            "post_activation_agent": "Complete setup (async, uses APIs)"
        }
        
        # System instructions for grounding
        self.system_instruction = f"""You are a B2B sales assistant for a cable/internet company helping businesses find internet and communication solutions.

STRICT RULES - YOU MUST FOLLOW THESE:
1. ONLY answer questions about business internet, voice, and communication services
2. REFUSE to answer questions about: sports, news, general knowledge, entertainment, or anything not related to our services
3. You can ONLY use these agents/tools to help customers:
{chr(10).join(f'   - {name}: {desc}' for name, desc in self.available_agents.items())}

4. When you need information, you MUST route to the appropriate agent
5. DO NOT make up product details - always query the product_policy_agent
6. DO NOT use internet knowledge - only use our internal agents and RAG system

AVAILABLE PRODUCTS (query product_policy_agent for details):
- Business Internet: 100 Mbps, 500 Mbps, 1 Gig
- Business Voice: Basic, Pro, Enterprise
- Managed Services: WiFi, Security, Cloud Backup

If asked about anything outside our services, politely decline and redirect to our offerings.

Your responses should be:
- Professional and helpful
- Concise but informative
- Always indicate which agent you're using (e.g., "Let me check with our Product Policy Agent...")
- Refuse off-topic questions politely
"""
    
    def get_framework(self) -> str:
        return "ADK + Gemini 3.0 Flash"
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a customer message using Gemini 3.0 Flash.
        
        Args:
            input_data: Customer message and context
        
        Returns:
            Response to customer
        """
        conversation_id = input_data.get("conversation_id") or generate_id("CONV")
        user_message = input_data.get("message", "")
        
        logger.info(
            "super_agent_processing",
            conversation_id=conversation_id,
            message=user_message[:100]
        )
        
        # Get or create conversation
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = {
                "id": conversation_id,
                "messages": [],
                "context": {},
                "state": "initial"
            }
        
        conversation = self.conversations[conversation_id]
        
        # Add user message to history
        conversation["messages"].append({
            "role": "user",
            "content": user_message,
            "timestamp": get_timestamp()
        })
        
        # Extract context for auto-triggering agents
        context = await self._extract_context(user_message, conversation_id)
        
        # Auto-trigger Prospect Agent if we have company info but haven't created prospect yet
        if context.get("company_name") and not context.get("prospect_created"):
            try:
                logger.info("auto_triggering_prospect_agent", conversation_id=conversation_id)
                prospect_response = await a2a_protocol.send_message(
                    from_agent="super_agent",
                    to_agent="prospect_agent",
                    message_type="request",
                    payload={
                        "company_name": context.get("company_name"),
                        "contact_name": context.get("contact_name"),
                        "employee_count": context.get("employee_count")
                    }
                )
                context["prospect_created"] = True
                context["prospect_id"] = prospect_response.get("prospect_id")
                logger.info("prospect_created", prospect_id=context.get("prospect_id"))
            except Exception as e:
                logger.error("prospect_creation_failed", error=str(e))
        
        # Auto-trigger Lead Generation Agent if we have prospect + service interest
        if context.get("prospect_id") and context.get("has_service_interest") and not context.get("lead_generated"):
            try:
                logger.info("auto_triggering_lead_generation", conversation_id=conversation_id)
                lead_response = await a2a_protocol.send_message(
                    from_agent="super_agent",
                    to_agent="lead_generation_agent",
                    message_type="request",
                    payload={
                        "prospect_id": context.get("prospect_id"),
                        "company_name": context.get("company_name"),
                        "employee_count": context.get("employee_count")
                    }
                )
                context["lead_generated"] = True
                context["lead_id"] = lead_response.get("lead_id")
                context["lead_score"] = lead_response.get("lead_score")
                logger.info("lead_generated", lead_id=context.get("lead_id"), score=context.get("lead_score"))
            except Exception as e:
                logger.error("lead_generation_failed", error=str(e))
        
        # Use Gemini if available, otherwise fallback
        if self.client:
            response = await self._process_with_gemini(user_message, conversation)
        else:
            response = await self._process_fallback(user_message, conversation)
        
        # Add assistant response to history
        conversation["messages"].append({
            "role": "assistant",
            "content": response["message"],
            "timestamp": get_timestamp()
        })
        
        # Update state
        if conversation["state"] == "initial":
            conversation["state"] = "engaged"
        
        return {
            "conversation_id": conversation_id,
            "message": response["message"],
            "intent": response.get("intent", "unknown"),
            "state": conversation["state"],
            "next_actions": response.get("next_actions", []),
            "agent_activity": {
                "primary_agent": "super_agent",
                "sub_agents_invoked": response.get("sub_agents_invoked", []),
                "communication_methods": response.get("communication_methods", []),
                "tools_used": response.get("tools_used", ["Gemini 3.0 Flash"])
            }
        }
    
    
    async def _extract_context(self, message: str, conversation_id: str) -> Dict[str, Any]:
        """Extract business context from message for auto-triggering agents."""
        context = self.conversation_context.get(conversation_id, {})
        
        message_lower = message.lower()
        
        # Extract company name (simple heuristic)
        if "from" in message_lower and (("llc" in message_lower or "inc" in message_lower or "corp" in message_lower)):
            words = message.split()
            for i, word in enumerate(words):
                if word.lower() == "from" and i + 1 < len(words):
                    company_words = []
                    for j in range(i + 1, min(i + 5, len(words))):
                        company_words.append(words[j])
                        if any(suffix in words[j].lower() for suffix in ["llc", "inc", "corp", "corporation"]):
                            context["company_name"] = " ".join(company_words)
                            break
        
        # Extract name
        if "i'm" in message_lower or "my name is" in message_lower:
            words = message.split()
            for i, word in enumerate(words):
                if word.lower() in ["i'm", "im"] and i + 1 < len(words):
                    context["contact_name"] = words[i + 1].strip(",")
                    break
        
        # Extract address
        if any(word in message_lower for word in ["street", "st", "avenue", "ave", "road", "rd"]):
            context["has_address"] = True
        
        # Extract service interest
        if any(word in message_lower for word in ["internet", "voice", "wifi", "security", "backup", "service"]):
            context["has_service_interest"] = True
        
        # Extract employee count
        if "employee" in message_lower:
            words = message.split()
            for i, word in enumerate(words):
                if word.isdigit():
                    context["employee_count"] = int(word)
                    break
        
        self.conversation_context[conversation_id] = context
        return context
    
    async def _process_with_gemini(
        self,
        message: str,
        conversation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process message using Gemini 2.5 Flash."""
        
        # Build conversation history for context
        history = []
        for msg in conversation["messages"][-10:]:  # Last 10 messages
            history.append(f"{msg['role'].upper()}: {msg['content']}")
        
        context_str = "\n".join(history) if history else "No previous context"
        
        # Create prompt
        prompt = f"""{self.system_instruction}

CONVERSATION HISTORY:
{context_str}

CURRENT USER MESSAGE:
{message}

INSTRUCTIONS:
1. First, determine if this is on-topic (about our services) or off-topic
2. If OFF-TOPIC: Politely decline and redirect to our services
3. If ON-TOPIC: Determine what the customer needs
4. If you need information, indicate which agent to query
5. Format your response beautifully using markdown:
   - Use **bold** for important terms
   - Use tables for comparing products/plans
   - Use bullet points for features
   - Use numbered lists for steps
   - Keep it clean and easy to scan

FORMATTING EXAMPLES:

For product comparisons, use tables:
| Plan | Speed | Price | Best For |
|------|-------|-------|----------|
| Internet 100 | 100 Mbps | $79.99/mo | Small teams (5-20) |
| Internet 500 | 500 Mbps | $149.99/mo | Medium teams (20-50) |

For features, use bullet points:
**Key Features:**
• 99.9% uptime SLA
• 24/7 technical support
• Business-class equipment included

Respond in this format:
TOPIC: [on-topic/off-topic]
AGENT_NEEDED: [agent_name or none]
RESPONSE: [your beautifully formatted response to the customer]
"""
        
        try:
            # Call Gemini 3.0 Flash with new SDK
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    thinking_config=self.thinking_config
                )
            )
            
            response_text = response.text
            
            # Parse response
            topic = "on-topic"
            agent_needed = None
            customer_response = response_text
            
            if "TOPIC:" in response_text:
                lines = response_text.split("\n")
                for i, line in enumerate(lines):
                    if line.startswith("TOPIC:"):
                        topic = line.split(":", 1)[1].strip().lower()
                    elif line.startswith("AGENT_NEEDED:"):
                        agent_str = line.split(":", 1)[1].strip().lower()
                        if agent_str != "none":
                            agent_needed = agent_str
                    elif line.startswith("RESPONSE:"):
                        # Get everything after RESPONSE:
                        customer_response = "\n".join(lines[i+1:]).strip()
                        break
            
            # If off-topic, return refusal
            if "off-topic" in topic:
                return {
                    "message": "🚫 **Off-Topic Request**\n\nI apologize, but I can only assist with questions about our business internet, voice, and communication services. I'm not able to answer questions about other topics.\n\n**How can I help you with:**\n• Business Internet plans\n• Voice services\n• Service availability\n• Pricing and quotes",
                    "intent": "off_topic_refusal",
                    "next_actions": ["Ask about products", "Check availability", "Get pricing"],
                    "sub_agents_invoked": [],
                    "communication_methods": [],
                    "tools_used": ["Gemini 2.5 Flash - Grounding Check"]
                }
            
            # If agent needed, query it
            sub_agents = []
            comm_methods = []
            tools = ["Gemini 2.5 Flash"]
            
            if agent_needed and agent_needed in self.available_agents:
                try:
                    logger.info("querying_agent", agent=agent_needed)
                    agent_response = await a2a_protocol.send_message(
                        from_agent="super_agent",
                        to_agent=agent_needed,
                        message_type="request",
                        payload={"question": message},
                        wait_for_response=True,
                        timeout=5.0
                    )
                    
                    if agent_response and agent_response.payload.get("context"):
                        context = agent_response.payload["context"]
                        
                        # Ask Gemini to format it cleanly - NO meta-commentary
                        format_prompt = f"""Format this product information as clean, professional markdown. 

CRITICAL RULES:
- NO introductory text like "Here is..." or "I've formatted..."
- NO meta-commentary about the formatting
- START IMMEDIATELY with the content
- Use markdown tables for product comparisons
- Use bullet points for features
- Keep it concise and scannable

Content to format:
{context}

Output ONLY the formatted content, nothing else."""
                        
                        formatted_response = await asyncio.to_thread(
                            self.client.models.generate_content,
                            model=self.model_name,
                            contents=format_prompt,
                            config=types.GenerateContentConfig(
                                temperature=0.3,  # Lower temp for formatting
                                thinking_config=self.thinking_config
                            )
                        )
                        
                        # Clean up any remaining meta-commentary
                        response_text = formatted_response.text.strip()
                        
                        # Remove common meta-commentary patterns
                        meta_patterns = [
                            "Here is the beautifully formatted",
                            "Here's the formatted",
                            "I've formatted",
                            "designed for clarity",
                            "for professional presentation",
                            "using Markdown tables and structure",
                        ]
                        
                        for pattern in meta_patterns:
                            if response_text.lower().startswith(pattern.lower()):
                                # Find the first actual content (usually after a newline or colon)
                                lines = response_text.split('\n')
                                # Skip first line if it contains meta-commentary
                                if any(p.lower() in lines[0].lower() for p in meta_patterns):
                                    response_text = '\n'.join(lines[1:]).strip()
                                break
                        
                        customer_response = f"📋 **{agent_needed.replace('_', ' ').title()}**\n\n{response_text}\n\n**Need more details?** Just ask!"
                        
                        sub_agents.append(agent_needed)
                        comm_methods.append("A2A Protocol")
                        if "policy" in agent_needed:
                            tools.extend(["RAG/ChromaDB", "sentence-transformers"])
                except Exception as e:
                    logger.error("agent_query_failed", agent=agent_needed, error=str(e))
            
            return {
                "message": customer_response,
                "intent": "gemini_processed",
                "next_actions": ["Get more details", "Check availability", "Request quote"],
                "sub_agents_invoked": sub_agents,
                "communication_methods": comm_methods,
                "tools_used": tools
            }
            
        except Exception as e:
            logger.error("gemini_error", error=str(e))
            return await self._process_fallback(message, conversation)
    
    def _format_rag_response(self, context: str, agent_name: str) -> str:
        """Format RAG context for better readability."""
        # Extract product information if present
        if "Internet 100" in context or "Internet 500" in context or "Internet 1 Gig" in context:
            return context  # Let Gemini format it
        return context
    
    async def _process_fallback(
        self,
        message: str,
        conversation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback processing when Gemini is not available."""
        return {
            "message": "👋 [Super Agent - Fallback Mode]\n\nHello! I'm here to help you with business internet and communication solutions. How can I assist you today?",
            "intent": "fallback",
            "next_actions": ["Ask about products", "Check availability", "Get pricing"],
            "sub_agents_invoked": [],
            "communication_methods": [],
            "tools_used": []
        }
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation by ID."""
        return self.conversations.get(conversation_id)
    
    def list_conversations(self) -> List[str]:
        """List all conversation IDs."""
        return list(self.conversations.keys())
