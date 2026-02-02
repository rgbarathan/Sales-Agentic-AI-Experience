# 🔬 Telemetry Dashboard

## Overview

Real-time monitoring dashboard for the Agentic Sales System running on **port 8001**.

## Features

### 📊 Real-Time Metrics

**Database (SQLite)**:
- Total conversations
- Total messages
- Agent invocations
- Orders placed

**RAG System (ChromaDB)**:
- Total documents indexed
- Number of collections
- Documents per collection
- Collection details

**Agents**:
- Total registered agents (15)
- Agent status
- Live agent list

**Live Activity**:
- Real-time event log
- WebSocket connection status
- System health

### 🎨 Features

- **Auto-refresh** - Updates every 2 seconds via WebSocket
- **Beautiful UI** - Dark theme with glassmorphism
- **Responsive** - Works on all screen sizes
- **Live status** - Green dot shows connection status

## Access

- **URL**: http://localhost:8001
- **API Docs**: http://localhost:8001/docs
- **Metrics API**: http://localhost:8001/api/metrics

## API Endpoints

```bash
# Get all metrics
GET /api/metrics

# Get conversations
GET /api/conversations

# Get messages for a conversation
GET /api/messages/{conversation_id}

# Get RAG collections
GET /api/rag/collections

# WebSocket for live updates
WS /ws/metrics
```

## Running

### Start Dashboard

```bash
# Standalone
python telemetry_dashboard.py

# Or in background
python telemetry_dashboard.py > /tmp/telemetry.log 2>&1 &
```

### Stop Dashboard

```bash
# Find and kill process
lsof -ti:8001 | xargs kill -9
```

## Usage During Demo

1. **Open main app**: http://localhost:8000
2. **Open telemetry**: http://localhost:8001
3. **Interact with main app** - chat, ask questions
4. **Watch telemetry** - see data being created, stored, retrieved in real-time

### What You'll See

**When user sends a message**:
- Messages count increases
- Agent invocations increment
- RAG queries show in activity log

**When RAG is queried**:
- Collection stats update
- Document counts visible
- Query activity logged

**When order is placed**:
- Orders count increases
- Conversation updated
- Database metrics change

## Perfect for Demos

- **Side-by-side display** - Main app + Telemetry
- **Show data flow** - See what's happening behind the scenes
- **Prove it works** - Real data, real-time
- **Impress stakeholders** - Professional monitoring

## Technical Details

- **Framework**: FastAPI
- **WebSocket**: Real-time updates every 2s
- **Database**: Direct SQLite queries
- **RAG**: ChromaDB collection stats
- **Frontend**: Vanilla JS, no frameworks
