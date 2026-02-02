# Super Agent Specification

## Overview

The Super Agent is the orchestrator of the multi-agent B2B sales system. It manages conversations with prospects, routes requests to specialized sub-agents, and coordinates the overall sales workflow.

## Requirements

### Requirement: Conversation Management

The Super Agent SHALL maintain conversation state across multiple interactions with a prospect.

#### Scenario: New Conversation
```
GIVEN a new prospect initiates contact
WHEN the Super Agent receives the first message
THEN it SHALL create a new conversation record
AND assign a unique conversation ID
AND initialize conversation state
```

#### Scenario: Continuing Conversation
```
GIVEN an existing conversation
WHEN the prospect sends a new message
THEN the Super Agent SHALL retrieve conversation history
AND maintain context from previous interactions
AND update conversation state
```

### Requirement: Intent Recognition

The Super Agent SHALL identify prospect intent from natural language input.

#### Scenario: Qualification Intent
```
GIVEN a prospect message like "I'm interested in business internet"
WHEN the Super Agent processes the message
THEN it SHALL recognize qualification intent
AND route to Prospect Agent
```

#### Scenario: Pricing Intent
```
GIVEN a prospect message like "How much does it cost?"
WHEN the Super Agent processes the message
THEN it SHALL recognize pricing intent
AND route to Offer Agent
```

#### Scenario: Order Intent
```
GIVEN a prospect message like "I'd like to place an order"
WHEN the Super Agent processes the message
THEN it SHALL recognize order intent
AND route to Order Agent
```

### Requirement: Agent Orchestration

The Super Agent SHALL coordinate multiple sub-agents to complete complex workflows.

#### Scenario: Complete Sales Flow
```
GIVEN a new prospect inquiry
WHEN the Super Agent orchestrates the sales flow
THEN it SHALL:
1. Invoke Prospect Agent for qualification
2. Invoke Address Validation Agent for address verification
3. Invoke Serviceability Agent for coverage check
4. Invoke Offer Agent for personalized offers
5. Invoke Order Agent when prospect is ready to purchase
AND maintain workflow state throughout
```

#### Scenario: Parallel Agent Invocation
```
GIVEN multiple independent tasks
WHEN the Super Agent needs information from multiple agents
THEN it SHALL invoke agents in parallel
AND aggregate results
AND present unified response to prospect
```

### Requirement: A2A Protocol Communication

The Super Agent SHALL use A2A Protocol for all agent-to-agent communication.

#### Scenario: Send Request to Sub-Agent
```
GIVEN a need to invoke a sub-agent
WHEN the Super Agent sends a request
THEN it SHALL use A2A Protocol message format
AND include conversation_id for tracking
AND wait for response with timeout
```

#### Scenario: Handle Agent Response
```
GIVEN a sub-agent response
WHEN the Super Agent receives the A2A message
THEN it SHALL process the response
AND update conversation state
AND formulate response to prospect
```

### Requirement: Error Handling

The Super Agent SHALL gracefully handle errors from sub-agents.

#### Scenario: Agent Timeout
```
GIVEN a sub-agent request
WHEN the agent does not respond within timeout
THEN the Super Agent SHALL log the timeout
AND provide fallback response to prospect
AND optionally retry with different agent
```

#### Scenario: Agent Error
```
GIVEN a sub-agent returns an error
WHEN the Super Agent receives the error response
THEN it SHALL log the error
AND determine if recovery is possible
AND provide appropriate response to prospect
```

### Requirement: Context Awareness

The Super Agent SHALL load and use its agent context configuration.

#### Scenario: Load Context on Initialization
```
GIVEN Super Agent initialization
WHEN the agent starts
THEN it SHALL load context from super_agent_context.yaml
AND build system prompt from context
AND register with A2A Protocol
```

#### Scenario: Use Context for Decisions
```
GIVEN an agent decision point
WHEN the Super Agent needs to determine next action
THEN it SHALL reference its domain knowledge
AND apply business rules from context
AND follow escalation rules when appropriate
```

### Requirement: Analytics Tracking

The Super Agent SHALL track all interactions for analytics.

#### Scenario: Log Conversation Events
```
GIVEN any conversation interaction
WHEN the Super Agent processes a message
THEN it SHALL log the message to database
AND record agent invocations
AND track conversation outcome
```

#### Scenario: Performance Metrics
```
GIVEN agent invocations
WHEN sub-agents are called
THEN the Super Agent SHALL record timing metrics
AND track success/failure rates
AND log to analytics database
```

## Implementation Details

### Framework
- **Framework**: ADK (Agent Development Kit)
- **LLM**: Google Gemini (gemini-2.0-flash-exp)
- **Context**: Loaded from `config/agent_contexts/super_agent_context.yaml`

### Connected Agents
The Super Agent communicates with:
- Prospect Agent
- Lead Generation Agent
- Address Validation Agent
- Serviceability Agent
- Offer Agent
- Order Agent
- Post Order Communication Agent
- Fulfillment Agent (async)
- Service Activation Agent (async)
- Post Activation Order Completion Agent (async)

### Tools
- **A2A Protocol**: For agent communication
- **Database**: SQLite for conversation storage
- **Analytics**: Event logging for metrics

### Configuration
```yaml
role: "Sales Orchestrator and Conversation Manager"
personality:
  tone: "Professional, helpful, and consultative"
  style: "Clear and efficient"
domain_knowledge:
  industry: "Cable/Telecommunications B2B"
  expertise:
    - "Sales workflow orchestration"
    - "Multi-agent coordination"
    - "Customer conversation management"
```

## Testing Requirements

### Test: Basic Conversation
```
GIVEN a new prospect
WHEN they send "Hello, I need business internet"
THEN Super Agent SHALL create conversation
AND invoke Prospect Agent
AND return friendly greeting with next steps
```

### Test: Complete Sales Flow
```
GIVEN a qualified prospect
WHEN they complete the sales journey
THEN Super Agent SHALL orchestrate all agents
AND track progress through each stage
AND successfully create an order
```

### Test: Error Recovery
```
GIVEN a sub-agent failure
WHEN an agent returns an error
THEN Super Agent SHALL handle gracefully
AND provide helpful message to prospect
AND log error for review
```

## Success Metrics

- **Response Time**: <2 seconds for simple queries
- **Workflow Completion**: >90% of qualified prospects complete flow
- **Error Rate**: <5% agent invocation errors
- **Customer Satisfaction**: >85% positive feedback

## Dependencies

- A2A Protocol implementation
- All sub-agent implementations
- Database layer
- Context loader
- Analytics module
