# Agent Instructions for OpenSpec Workflow

This project follows the OpenSpec specification-driven development workflow for AI coding assistants.

## Project Overview

This is a multi-agent B2B sales system demonstrating Agentic AI capabilities. The system includes:
- 1 Super Agent (orchestrator)
- 11 Operational Sub-Agents (prospect qualification, serviceability, offers, orders, fulfillment, etc.)
- 4 Policy Agents (RAG-based knowledge specialists)
- Mixed frameworks: ADK, Strands SDK, LangGraph/LangChain
- Communication: A2A Protocol, MCP servers, REST APIs

## Specifications Location

All specifications are located in `openspec/specs/`:

- `agents/` - Agent specifications (capabilities, requirements, interactions)
- `tools/` - Tool specifications (MCP tools, REST endpoints)
- `workflows/` - Workflow specifications (agent orchestration, sales flows)

## Development Workflow

When working on this project:

1. **Review Specifications First**
   - Check `openspec/specs/` for existing specifications
   - Understand agent capabilities and requirements
   - Review workflow specifications for interaction patterns

2. **Propose Changes**
   - Use `/openspec-proposal` to create change proposals
   - Specify what agents/tools/workflows need to be modified
   - Include clear requirements and scenarios

3. **Implement Changes**
   - Use `/openspec-apply` to implement approved specifications
   - Follow the agent context patterns in `config/agent_contexts/`
   - Maintain separation of concerns (agents, tools, workflows)

4. **Archive Completed Work**
   - Use `/openspec-archive` to merge approved changes
   - Update source-of-truth specifications

## Key Architectural Patterns

### Agent Pattern
- Each agent has a YAML context file in `config/agent_contexts/`
- Agents communicate via A2A Protocol
- Policy Agents use RAG for knowledge retrieval
- Operational Agents use MCP tools or REST APIs

### Tool Pattern
- MCP tools for data access (CRM, inventory, network)
- REST APIs for external system integration
- All tools have OpenSpec specifications

### Workflow Pattern
- Super Agent orchestrates sub-agents
- Async agents handle long-running tasks
- Policy Agents provide knowledge to operational agents

## Current Implementation Status

See `PROGRESS.md` for detailed status.

**Completed**:
- Core infrastructure
- RAG system with policy documents
- Database layer (SQLite)
- Shared modules (A2A protocol, context loader, mock data)

**In Progress**:
- Agent context configurations
- Policy Agent implementations
- MCP server implementations
- Operational agent implementations

## Important Files

- `README.md` - Setup and usage guide
- `PROGRESS.md` - Implementation progress
- `AgenticAIImplementationPlan.txt` - Detailed implementation plan
- `config/agent_contexts/` - Agent configuration files
- `rag/documents/` - Policy documents for RAG
- `openspec/specs/` - OpenSpec specifications

## Testing

- Unit tests in `tests/`
- Integration tests for agent workflows
- End-to-end demo scenarios

## Questions?

Refer to the specifications in `openspec/specs/` or ask the development team.
