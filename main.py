"""Main FastAPI application for the Agentic Sales System."""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict, Any, List
from dotenv import load_dotenv
from datetime import datetime
import structlog
import logging
import os
from logging.handlers import TimedRotatingFileHandler

# Load environment variables from .env file
load_dotenv()

from config.settings import settings
from agents import SuperAgent
from agents.policy_agents import (
    ProductPolicyAgent,
    OrderPolicyAgent,
    ServicePolicyAgent,
    FulfillmentPolicyAgent
)
from database.sqlite_db import SQLiteDB
from shared.protocols import a2a_protocol

# Configure logging
def configure_logging():
    """Configure structured logging with file rotation."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "agentic_sales.log")

    # Standard logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),  # Console
            TimedRotatingFileHandler(
                log_file,
                when="H",
                interval=1,
                backupCount=48,  # Keep 48 hours of logs
                encoding="utf-8"
            )
        ]
    )

    # Structlog configuration
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

configure_logging()
logger = structlog.get_logger()


# Global instances
super_agent: SuperAgent = None
db: SQLiteDB = None
active_connections: List[WebSocket] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global super_agent, db
    
    logger.info("application_startup", message="Initializing Agentic Sales System")
    
    # Initialize database (happens in __init__)
    db = SQLiteDB(settings.sqlite_db_path)
    logger.info("database_ready", path=settings.sqlite_db_path)
    
    # Initialize Policy Agents (4)
    logger.info("initializing_policy_agents")
    from agents.policy_agents import (
        ProductPolicyAgent,
        OrderPolicyAgent,
        ServicePolicyAgent,
        FulfillmentPolicyAgent
    )
    product_policy = ProductPolicyAgent()
    order_policy = OrderPolicyAgent()
    service_policy = ServicePolicyAgent()
    fulfillment_policy = FulfillmentPolicyAgent()
    logger.info("policy_agents_initialized", count=4)
    
    # Initialize Operational Agents (11)
    logger.info("initializing_operational_agents")
    from agents import (
        ProspectAgent,
        LeadGenerationAgent,
        ServiceabilityAgent,
        AddressValidationAgent,
        OfferAgent,
        OrderAgent,
        PostOrderCommunicationAgent,
        FulfillmentAgent,
        ServiceActivationAgent,
        PostActivationAgent
    )
    
    # ADK Agents
    prospect_agent = ProspectAgent()
    lead_gen_agent = LeadGenerationAgent()
    order_agent = OrderAgent()
    
    # Strands Agents
    serviceability_agent = ServiceabilityAgent()
    offer_agent = OfferAgent()
    post_order_comm_agent = PostOrderCommunicationAgent()
    
    # LangGraph Agents
    address_validation_agent = AddressValidationAgent()
    fulfillment_agent = FulfillmentAgent()
    service_activation_agent = ServiceActivationAgent()
    post_activation_agent = PostActivationAgent()
    
    logger.info("operational_agents_initialized", count=10)
    
    # Initialize Super Agent (orchestrator)
    logger.info("initializing_super_agent")
    super_agent = SuperAgent()
    logger.info("super_agent_initialized")
    
    logger.info("application_ready", message="Agentic Sales System is ready", total_agents=15)
    
    yield
    
    # Cleanup
    logger.info("application_shutdown", message="Shutting down Agentic Sales System")


# Create FastAPI app
app = FastAPI(
    title="Agentic Sales System",
    description="AI-powered B2B sales agent system for Cable MSO",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Agentic Sales System",
        "version": "1.0.0"
    }


# Chat endpoint (REST)
@app.post("/api/chat")
async def chat(request: Dict[str, Any]):
    """
    Process a chat message.
    
    Request body:
    {
        "conversation_id": "optional-id",
        "message": "user message"
    }
    """
    try:
        message = request.get("message")
        conversation_id = request.get("conversation_id")
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Process with Super Agent
        result = await super_agent.process({
            "conversation_id": conversation_id,
            "message": message
        })
        
        # Log to database
        if db:
            await db.log_message(
                conversation_id=result["conversation_id"],
                role="user",
                content=message
            )
            await db.log_message(
                conversation_id=result["conversation_id"],
                role="assistant",
                content=result["message"]
            )
        
        return result
        
    except Exception as e:
        logger.error("chat_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint for real-time chat
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time chat."""
    await websocket.accept()
    active_connections.append(websocket)
    conversation_id = None
    
    try:
        logger.info("websocket_connected", client=websocket.client)
        
        while True:
            # Receive message
            data = await websocket.receive_json()
            message = data.get("message")
            conversation_id = data.get("conversation_id", conversation_id)
            
            if not message:
                await websocket.send_json({"error": "Message is required"})
                continue
            
            # Process with Super Agent
            result = await super_agent.process({
                "conversation_id": conversation_id,
                "message": message
            })
            
            conversation_id = result["conversation_id"]
            
            # Log to database
            if db:
                await db.log_message(
                    conversation_id=conversation_id,
                    role="user",
                    content=message
                )
                await db.log_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=result["message"]
                )
            
            # Send response
            await websocket.send_json(result)
            
    except WebSocketDisconnect:
        logger.info("websocket_disconnected", conversation_id=conversation_id)
        active_connections.remove(websocket)
    except Exception as e:
        logger.error("websocket_error", error=str(e))
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
        if websocket in active_connections:
            active_connections.remove(websocket)


# Conversation history endpoint
@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history."""
    try:
        if not db:
            raise HTTPException(status_code=503, detail="Database not available")
        
        messages = await db.get_conversation_messages(conversation_id)
        
        return {
            "conversation_id": conversation_id,
            "messages": messages
        }
        
    except Exception as e:
        logger.error("get_conversation_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# List conversations endpoint
@app.get("/api/conversations")
async def list_conversations():
    """List all conversations."""
    try:
        if not db:
            raise HTTPException(status_code=503, detail="Database not available")
        
        conversations = await db.list_conversations()
        
        return {
            "conversations": conversations
        }
        
    except Exception as e:
        logger.error("list_conversations_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Agent status endpoint
@app.get("/api/agents/status")
async def agent_status():
    """Get status of all agents."""
    try:
        registered_agents = list(a2a_protocol.agents.keys())
        
        return {
            "super_agent": "active" if super_agent else "inactive",
            "registered_agents": registered_agents,
            "total_agents": len(registered_agents)
        }
        
    except Exception as e:
        logger.error("agent_status_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Serve static files (web UI)
try:
    app.mount("/static", StaticFiles(directory="web/static"), name="static")
except:
    logger.warning("static_files_not_found", message="web/static directory not found")


# Mount static files for main web UI
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# Mount telemetry static files
app.mount("/telemetry/static", StaticFiles(directory="telemetry/static"), name="telemetry_static")


@app.get("/telemetry", response_class=HTMLResponse)
async def serve_telemetry():
    """Serve the telemetry dashboard."""
    try:
        import os
        telemetry_path = os.path.join(os.path.dirname(__file__), "telemetry", "index.html")
        with open(telemetry_path, 'r') as f:
            content = f.read()
            # Update static paths to use /telemetry/static
            content = content.replace('src="/static/', 'src="/telemetry/static/')
            content = content.replace('href="/static/', 'href="/telemetry/static/')
            return HTMLResponse(content=content)
    except FileNotFoundError as e:
        logger.error("telemetry_html_not_found", error=str(e), path=telemetry_path)
        return HTMLResponse(content="<html><body><h1>Telemetry dashboard not found</h1><p>Check that telemetry/index.html exists</p></body></html>", status_code=404)
    except Exception as e:
        logger.error("telemetry_error", error=str(e))
        return HTMLResponse(content=f"<html><body><h1>Error loading telemetry</h1><p>{str(e)}</p></body></html>", status_code=500)


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the main web UI."""
    try:
        with open("web/index.html") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return """
        <html>
            <head><title>Agentic Sales System</title></head>
            <body>
                <h1>Agentic Sales System API</h1>
                <p>The web UI is not yet available. Use the API endpoints:</p>
                <ul>
                    <li>POST /api/chat - Send a chat message</li>
                    <li>WS /ws/chat - WebSocket chat</li>
                    <li>GET /api/conversations - List conversations</li>
                    <li>GET /api/agents/status - Agent status</li>
                    <li>GET /health - Health check</li>
                </ul>
                <p><a href="/docs">API Documentation</a></p>
            </body>
        </html>
        """


# ============================================================================
# TELEMETRY ENDPOINTS
# ============================================================================

@app.get("/api/telemetry/metrics")
async def get_telemetry_metrics():
    """Get comprehensive system metrics for telemetry dashboard."""
    return {
        "timestamp": datetime.now().isoformat(),
        "database": await get_database_metrics(),
        "rag": get_rag_metrics(),
        "agents": get_agent_metrics(),
        "system": get_system_metrics()
    }


@app.get("/api/telemetry/conversations")
async def get_telemetry_conversations():
    """Get recent conversations for telemetry."""
    try:
        conversations = await db.list_conversations()
        return {
            "total": len(conversations),
            "recent": conversations[:10]
        }
    except Exception as e:
        logger.error("failed_to_get_conversations", error=str(e))
        return {"total": 0, "recent": []}


@app.get("/api/telemetry/rag/collections")
async def get_telemetry_rag_collections():
    """Get RAG collection statistics."""
    try:
        from rag.rag_manager import rag_manager
        
        collections = {}
        if hasattr(rag_manager, 'collections') and rag_manager.collections:
            for name, collection in rag_manager.collections.items():
                try:
                    count = collection.count()
                    collections[name] = {
                        "name": name,
                        "document_count": count,
                        "last_updated": datetime.now().isoformat()
                    }
                except Exception as e:
                    logger.error("failed_to_count_collection", collection=name, error=str(e))
        else:
            logger.warning("rag_collections_empty_in_endpoint")
        return collections
    except Exception as e:
        logger.error("failed_to_get_rag_collections", error=str(e))
        return {}


async def get_database_metrics():
    """Get database metrics."""
    try:
        with db.get_session() as session:
            from database.models import ConversationDB, MessageDB, AgentInvocationDB, OrderDB
            
            return {
                "conversations": session.query(ConversationDB).count(),
                "messages": session.query(MessageDB).count(),
                "agent_invocations": session.query(AgentInvocationDB).count(),
                "orders": session.query(OrderDB).count()
            }
    except Exception as e:
        logger.error("failed_to_get_db_metrics", error=str(e))
        return {"conversations": 0, "messages": 0, "agent_invocations": 0, "orders": 0}


def get_rag_metrics():
    """Get RAG system metrics."""
    try:
        # Import rag_manager from the initialized instance
        from rag.rag_manager import rag_manager
        
        total_docs = 0
        collections_info = {}
        
        if hasattr(rag_manager, 'collections') and rag_manager.collections:
            for name, collection in rag_manager.collections.items():
                try:
                    count = collection.count()
                    total_docs += count
                    collections_info[name] = count
                except Exception as e:
                    logger.error("failed_to_count_rag_collection", collection=name, error=str(e))
        else:
            logger.warning("rag_collections_empty", message="RAG manager has no collections")
        
        return {
            "total_documents": total_docs,
            "collections": len(collections_info),
            "collections_info": collections_info
        }
    except Exception as e:
        logger.error("failed_to_get_rag_metrics", error=str(e))
        return {"total_documents": 0, "collections": 0, "collections_info": {}}


def get_agent_metrics():
    """Get agent metrics."""
    try:
        return {
            "total_agents": len(a2a_protocol.agents),
            "registered_agents": list(a2a_protocol.agents.keys()),
            "active_agents": len(a2a_protocol.agents)
        }
    except Exception as e:
        logger.error("failed_to_get_agent_metrics", error=str(e))
        return {"total_agents": 0, "registered_agents": [], "active_agents": 0}


def get_system_metrics():
    """Get system-level metrics."""
    return {
        "uptime": "running",
        "status": "healthy",
        "active_connections": len(active_connections)
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
