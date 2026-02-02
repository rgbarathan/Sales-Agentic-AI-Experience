# Project Context

## Purpose
This is a demonstration project showcasing a multi-agent B2B sales system for a Cable MSO (Multiple System Operator). The system demonstrates the power of Agentic AI by orchestrating 15+ specialized agents to handle the complete sales workflow from prospect qualification to service activation.

**Key Goals:**
- Demonstrate multi-agent orchestration with a Super Agent
- Showcase mixed agent frameworks (ADK, Strands SDK, LangGraph/LangChain)
- Implement RAG-based Policy Agents for knowledge management
- Show A2A (Agent-to-Agent) protocol communication
- Demonstrate async/offline agent processing
- Provide production-quality demo with realistic data

## Tech Stack

### Core Technologies
- **Language**: Python 3.11+
- **Web Framework**: FastAPI
- **LLM**: Google Gemini (gemini-2.0-flash-exp)

### Agent Frameworks
- **ADK (Agent Development Kit)**: Super Agent, Prospect Agent, Lead Generation Agent, Order Agent
- **Strands SDK**: Serviceability Agent, Offer Agent, Post Order Communication Agent
- **LangGraph/LangChain**: Address Validation Agent, Fulfillment Agent, Service Activation Agent, Post Activation Agent

### Data & Storage
- **Database**: SQLite (structured data), Elasticsearch (optional, for log analytics)
- **RAG**: ChromaDB with sentence-transformers for vector search
- **Mock Data**: Faker library for realistic business data

### Communication
- **A2A Protocol**: Custom agent-to-agent messaging protocol
- **MCP (Model Context Protocol)**: For tool integration (6 MCP servers)
- **REST APIs**: For external system integration (5 REST endpoints)

### Development Tools
- **OpenSpec**: Spec-driven development workflow
- **Testing**: pytest, pytest-asyncio
- **Logging**: structlog (structured logging)
- **Configuration**: Pydantic Settings

## Project Conventions

### Code Style
- **Type Hints**: All functions must have type hints
- **Docstrings**: Google-style docstrings for all classes and public methods
- **Formatting**: Follow PEP 8 conventions
- **Imports**: Organized (stdlib, third-party, local)
- **Line Length**: Maximum 100 characters

### Architecture Patterns

#### Agent Pattern
- Each agent has a dedicated YAML context file in `config/agent_contexts/`
- Agents register with A2A Protocol on initialization
- Agents load their context and build system prompts dynamically
- Policy Agents use RAG for knowledge retrieval
- Operational Agents use MCP tools or REST APIs

#### Separation of Concerns
- **agents/**: Agent implementations (ADK, Strands, LangGraph, Policy)
- **mcp_servers/**: MCP server implementations
- **rest_api/**: REST API endpoints
- **shared/**: Common utilities (models, protocols, context loader, mock data)
- **database/**: Database layer (SQLite, Elasticsearch)
- **rag/**: RAG system and policy documents
- **config/**: Configuration and agent contexts
- **openspec/**: OpenSpec specifications

#### Communication Architecture
- **A2A Protocol**: Standardized messaging between agents
- **MCP**: For structured data access (CRM, inventory, network data)
- **REST**: For external system integration
- **Async Jobs**: Background processing for long-running tasks (fulfillment, activation)

### Testing Strategy

#### Unit Tests
- Test individual agents in isolation
- Mock A2A protocol and tool calls
- Test context loading and prompt generation
- Test RAG queries

#### Integration Tests
- Test agent-to-agent communication
- Test complete workflows (prospect → order → fulfillment)
- Test MCP server interactions
- Test REST API endpoints

#### Demo Scenarios
- Happy path: Complete sale from prospect to activation
- Edge cases: Address not serviceable, out of stock equipment
- Error handling: Agent timeouts, service failures

### Git Workflow
- **Main Branch**: Production-ready code
- **Feature Branches**: `feature/agent-name` or `feature/component-name`
- **Commits**: Descriptive commit messages following conventional commits
- **Pull Requests**: Required for all changes with spec review

## Domain Context

### Business Domain: Cable MSO B2B Sales
- **Target Customers**: Small and Medium Businesses (SMBs)
- **Products**: Business Internet, Voice, Managed WiFi, Managed Security, Cloud Backup
- **Sales Process**: Prospect qualification → Serviceability check → Offer generation → Order → Fulfillment → Activation
- **Service Tiers**: Internet 100, Internet 500, Internet 1 Gig with varying SLAs

### Agent Roles
- **Super Agent**: Orchestrates conversation and coordinates sub-agents
- **Prospect Agent**: Qualifies business prospects
- **Lead Generation Agent**: Scores and enriches leads
- **Address Validation Agent**: Validates business addresses
- **Serviceability Agent**: Checks network coverage and availability
- **Offer Agent**: Generates personalized offers with pricing
- **Order Agent**: Processes orders and initiates fulfillment
- **Post Order Communication Agent**: Sends confirmations and updates
- **Fulfillment Agent** (async): Handles equipment and installation scheduling
- **Service Activation Agent** (async): Provisions network services
- **Post Activation Agent** (async): Completes billing and CRM updates

### Policy Agents (RAG-Based)
- **Product Policy Agent**: Product catalog, pricing rules, promotions
- **Order Policy Agent**: Order processing SOPs, terms & conditions
- **Service Policy Agent**: Coverage policies, SLAs, network standards
- **Fulfillment Policy Agent**: Equipment specs, installation procedures

## Important Constraints

### Technical Constraints
- **No Real Integrations**: All data is mocked, no actual CRM/billing/network systems
- **Demo Performance**: Configurable delays for realistic demo behavior
- **LLM Rate Limits**: Respect Google Gemini API rate limits
- **Local Development**: Must run entirely on local machine (except LLM API)

### Business Constraints
- **SMB Focus**: Only small/medium business customers (5-250 employees)
- **Service Areas**: Limited to specific zip codes in demo coverage map
- **Product Catalog**: Fixed set of products (no custom offerings in demo)
- **Contract Terms**: 12, 24, or 36-month contracts only

### Regulatory Constraints
- **Data Privacy**: No real customer data, all mock data
- **Compliance**: Demo only, not production-compliant
- **SLA Commitments**: Mock SLAs for demonstration purposes

## External Dependencies

### APIs
- **Google Gemini API**: LLM for agent intelligence (requires API key)
- **ChromaDB**: Embedded vector database (no external service)
- **SQLite**: Embedded database (no external service)
- **Elasticsearch** (optional): For log analytics if enabled

### Mock External Systems
All external systems are mocked with realistic data:
- **CRM System**: Mock prospect and customer data
- **Network Management**: Mock serviceability and coverage data
- **Inventory System**: Mock equipment availability
- **Billing System**: Mock billing and payment processing
- **Provisioning System**: Mock service activation

### Development Dependencies
- **Node.js/npm**: For OpenSpec CLI tool
- **Python 3.11+**: Runtime environment
- **Virtual Environment**: Isolated Python dependencies

## OpenSpec Workflow

This project uses OpenSpec for spec-driven development:

1. **Specifications**: Agent specs in `openspec/specs/agents/`
2. **Changes**: Proposed changes in `openspec/changes/`
3. **Workflows**: Use `/openspec-proposal`, `/openspec-apply`, `/openspec-archive` commands
4. **Agent Instructions**: See `openspec/AGENTS.md` for AI assistant guidance

## Getting Started

See `README.md` for complete setup instructions including:
- Python venv setup
- Dependency installation
- RAG initialization
- Running the demo
