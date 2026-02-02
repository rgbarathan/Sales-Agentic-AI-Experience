# B2B Sales Agentic AI System

A production-ready multi-agent AI system for B2B sales automation, featuring intelligent agent orchestration, real-time chat, and automated sales workflows.

## 🚀 Features

- **Multi-Agent Architecture**: 15+ specialized agents working together via Agent-to-Agent (A2A) protocol
- **Intelligent Orchestration**: Super Agent coordinates all sub-agents for seamless workflows
- **Auto-Triggered Workflows**: Automatic prospect creation and lead generation based on conversation context
- **Real-Time Chat**: WebSocket-based chat interface with markdown support
- **RAG Integration**: Retrieval-Augmented Generation for policy and product information
- **Agent Activity Tracking**: Visual tags showing which agents, tools, and protocols were used
- **Telemetry Dashboard**: Real-time monitoring of system health and agent activity

## 🏗️ Architecture

### System Overview

```mermaid
graph LR
    User[👤 User<br/>WebSocket] --> SuperAgent[🧠 Super Agent<br/>Gemini 2.0 Flash]
    
    SuperAgent --> PolicyAgents[📋 Policy Agents<br/>MCP Protocol]
    SuperAgent --> OpAgents[🎯 Operational Agents<br/>A2A Protocol]
    SuperAgent --> ServiceAgents[🌐 Service Agents<br/>REST API]
    SuperAgent --> WorkflowAgents[⚙️ Workflow Agents<br/>LangGraph]
    
    PolicyAgents --> P1[Product Policy]
    PolicyAgents --> P2[Order Policy]
    PolicyAgents --> P3[Service Policy]
    PolicyAgents --> P4[Fulfillment Policy]
    
    OpAgents --> O1[Prospect Agent<br/>Auto-triggered]
    OpAgents --> O2[Lead Gen Agent<br/>Auto-triggered]
    OpAgents --> O3[Order Agent]
    
    ServiceAgents --> S1[Serviceability]
    ServiceAgents --> S2[Offer]
    ServiceAgents --> S3[Post Order Comm]
    
    WorkflowAgents --> W1[Address Validation]
    WorkflowAgents --> W2[Fulfillment]
    WorkflowAgents --> W3[Service Activation]
    WorkflowAgents --> W4[Post Activation]
    
    O3 -.A2A.-> S1
    S1 -.A2A.-> W2
    
    SuperAgent --> RAG[(📚 RAG<br/>ChromaDB)]
    SuperAgent --> DB[(💾 Database<br/>SQLite)]
    SuperAgent --> Gemini[☁️ Gemini API]
    
    classDef super fill:#ef4444,stroke:#dc2626,color:#fff,stroke-width:3px
    classDef group fill:#374151,stroke:#6b7280,color:#fff
    classDef policy fill:#3b82f6,stroke:#1e40af,color:#fff
    classDef operational fill:#10b981,stroke:#059669,color:#fff
    classDef service fill:#f59e0b,stroke:#d97706,color:#fff
    classDef workflow fill:#8b5cf6,stroke:#6d28d9,color:#fff
    classDef support fill:#6b7280,stroke:#4b5563,color:#fff
    
    class SuperAgent super
    class PolicyAgents,OpAgents,ServiceAgents,WorkflowAgents group
    class P1,P2,P3,P4 policy
    class O1,O2,O3 operational
    class S1,S2,S3 service
    class W1,W2,W3,W4 workflow
    class RAG,DB,Gemini support
```

**Key Features:**
- **Super Agent** orchestrates all 15 specialized agents
- **Auto-triggered**: Prospect & Lead Gen agents activate automatically
- **Sub-agent communication**: Order → Serviceability → Fulfillment (A2A)
- **Multiple protocols**: MCP, A2A, REST, LangGraph, WebSocket

### Agent Types

**Policy Agents** (MCP-based)
- Product Policy Agent
- Order Policy Agent
- Service Policy Agent
- Fulfillment Policy Agent

**Operational Agents** (ADK-based)
- Prospect Agent (auto-triggered)
- Lead Generation Agent (auto-triggered)
- Order Agent (with sub-agent communication)

**Service Agents** (Strands-based)
- Serviceability Agent
- Offer Agent
- Post Order Communication Agent

**Workflow Agents** (LangGraph-based)
- Address Validation Agent
- Fulfillment Agent
- Service Activation Agent
- Post Activation Agent

### Communication Flow

```
User (WebSocket)
    ↓
Super Agent (Gemini 2.0 Flash) ← Central Orchestrator
    ├─→ Policy Agents (MCP)
    ├─→ Operational Agents (A2A)
    ├─→ Service Agents (REST)
    ├─→ Workflow Agents (LangGraph)
    ├─→ RAG System (ChromaDB)
    └─→ Database (SQLite)

Sub-Agent to Sub-Agent (A2A):
    Order Agent → Serviceability Agent → Fulfillment Agent
```

## 📋 Prerequisites

- Python 3.13+
- Google Gemini API key
- SQLite (included)

## 🔧 Installation

1. **Clone the repository**
```bash
git clone https://github.com/sudhamanc/SalesAgenticAIDemo.git
cd SalesAgenticAIDemo
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

## 🚀 Quick Start

1. **Start the application**
```bash
python main.py
```

2. **Access the UI**
```
http://localhost:8000
```

3. **Try a sample conversation**
```
"Hi, I'm John from TechCorp LLC at 123 Main St. 
We have 100 employees and need business internet."
```

The system will automatically:
- Create a prospect record
- Generate a qualified lead
- Provide service recommendations
- Invoke relevant agents as needed

## 🎯 Key Capabilities

### Auto-Triggered Workflows

**Prospect Creation** - Automatically triggered when:
- Company name is mentioned
- Contact information provided

**Lead Generation** - Automatically triggered when:
- Prospect exists
- Service interest expressed
- Employee count provided

**Order Processing** - Sub-agent communication flow:
1. Order Agent validates request
2. Calls Serviceability Agent to check address
3. Calls Fulfillment Agent to schedule installation
4. Returns complete order details

### Agent Activity Tracking

Every response shows:
- 🤖 **Blue tags**: Sub-agents invoked
- ⚡ **Green tags**: Tools used (Gemini, RAG, etc.)
- 📡 **Orange tags**: Communication protocols (A2A, MCP, REST)

## 📊 Telemetry Dashboard

Access real-time system metrics at:
```
http://localhost:8000/telemetry
```

Features:
- Active conversations
- Agent invocation counts
- Database statistics
- Conversation history viewer

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python 3.13
- **LLM**: Google Gemini 2.0 Flash
- **Database**: SQLite
- **Vector Store**: ChromaDB
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **Protocols**: WebSocket, A2A, MCP, REST

## 📁 Project Structure

```
├── agents/              # All agent implementations
│   ├── super_agent.py   # Main orchestrator
│   ├── policy_agents/   # MCP-based policy agents
│   ├── adk_agents/      # ADK operational agents
│   ├── strands_agents/  # Strands service agents
│   └── langgraph_agents/# LangGraph workflow agents
├── database/            # SQLite database layer
├── rag/                 # RAG implementation
│   └── documents/       # Policy documents
├── shared/              # Shared utilities
│   └── protocols.py     # A2A protocol implementation
├── web/                 # Frontend UI
│   ├── index.html
│   └── static/
├── telemetry/           # Telemetry dashboard
├── config/              # Configuration
├── main.py              # Application entry point
└── requirements.txt     # Python dependencies
```

## 🔐 Security

- API keys stored in `.env` (gitignored)
- No sensitive data in repository
- SQLite database for local development

## 📝 License

MIT License - See LICENSE file for details

## 👥 Contributing

Contributions welcome! Please open an issue or submit a pull request.

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ using Google Gemini and FastAPI**
