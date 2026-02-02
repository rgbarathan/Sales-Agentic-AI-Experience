"""Pydantic models for the Agentic AI Sales System."""
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from enum import Enum


# Enums
class IndustryType(str, Enum):
    """Industry types for SMB prospects."""
    RETAIL = "retail"
    HEALTHCARE = "healthcare"
    PROFESSIONAL_SERVICES = "professional_services"
    MANUFACTURING = "manufacturing"
    HOSPITALITY = "hospitality"
    EDUCATION = "education"
    FINANCE = "finance"
    TECHNOLOGY = "technology"
    OTHER = "other"


class CompanySize(str, Enum):
    """Company size categories."""
    MICRO = "micro"  # 1-10 employees
    SMALL = "small"  # 11-50 employees
    MEDIUM = "medium"  # 51-250 employees


class ServiceType(str, Enum):
    """Types of services offered."""
    INTERNET = "internet"
    VOICE = "voice"
    MANAGED_WIFI = "managed_wifi"
    MANAGED_SECURITY = "managed_security"
    CLOUD_BACKUP = "cloud_backup"
    UNIFIED_COMMUNICATIONS = "unified_communications"


class OrderStatus(str, Enum):
    """Order status values."""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    PROCESSING = "processing"
    FULFILLMENT = "fulfillment"
    ACTIVATION = "activation"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


# Address Models
class Address(BaseModel):
    """Address model."""
    street: str
    city: str
    state: str
    zip_code: str
    country: str = "USA"
    latitude: Optional[float] = None
    longitude: Optional[float] = None


# Prospect Models
class ContactInfo(BaseModel):
    """Contact information."""
    name: str
    email: EmailStr
    phone: str
    title: Optional[str] = None


class ProspectData(BaseModel):
    """Prospect data model."""
    prospect_id: str
    company_name: str
    industry: IndustryType
    company_size: CompanySize
    employee_count: int
    annual_revenue: Optional[float] = None
    business_address: Address
    contact_info: ContactInfo
    website: Optional[str] = None
    existing_customer: bool = False
    qualification_score: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.now)


# Lead Models
class LeadData(BaseModel):
    """Lead data model."""
    lead_id: str
    prospect_id: str
    lead_score: int
    priority: Literal["low", "medium", "high", "urgent"]
    enriched_data: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.now)


# Serviceability Models
class ServiceAvailability(BaseModel):
    """Service availability model."""
    service_type: ServiceType
    available: bool
    max_bandwidth_mbps: Optional[int] = None
    installation_fee: Optional[float] = None


class ServiceabilityResult(BaseModel):
    """Serviceability check result."""
    address: Address
    is_serviceable: bool
    available_services: List[ServiceAvailability]
    network_type: Optional[str] = None  # fiber, coax, hybrid
    estimated_install_days: Optional[int] = None


# Product & Offer Models
class Product(BaseModel):
    """Product model."""
    product_id: str
    name: str
    service_type: ServiceType
    description: str
    bandwidth_mbps: Optional[int] = None
    base_price_monthly: float
    installation_fee: float = 0.0
    contract_term_months: int = 12
    features: List[str] = []


class Offer(BaseModel):
    """Offer model."""
    offer_id: str
    prospect_id: str
    products: List[Product]
    total_monthly_price: float
    total_installation_fee: float
    discount_percentage: float = 0.0
    promotion_code: Optional[str] = None
    valid_until: datetime
    terms_and_conditions: str


# Order Models
class OrderItem(BaseModel):
    """Order item model."""
    product_id: str
    quantity: int = 1
    monthly_price: float
    installation_fee: float = 0.0


class Order(BaseModel):
    """Order model."""
    order_id: str
    prospect_id: str
    conversation_id: str
    items: List[OrderItem]
    installation_address: Address
    billing_address: Optional[Address] = None
    total_monthly_price: float
    total_installation_fee: float
    status: OrderStatus = OrderStatus.DRAFT
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    fulfillment_job_id: Optional[str] = None
    activation_job_id: Optional[str] = None
    completion_job_id: Optional[str] = None


# Fulfillment Models
class FulfillmentJob(BaseModel):
    """Fulfillment job model."""
    job_id: str
    order_id: str
    equipment_items: List[str]
    installation_date: Optional[datetime] = None
    technician_id: Optional[str] = None
    status: Literal["pending", "scheduled", "in_progress", "completed", "failed"]
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


# Service Activation Models
class ActivationJob(BaseModel):
    """Service activation job model."""
    job_id: str
    order_id: str
    fulfillment_job_id: str
    services_to_activate: List[str]
    status: Literal["pending", "provisioning", "testing", "completed", "failed"]
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    test_results: Optional[Dict[str, Any]] = None


# A2A Protocol Models
class A2AMessage(BaseModel):
    """Agent-to-Agent protocol message."""
    message_id: str
    from_agent: str
    to_agent: str
    message_type: Literal["request", "response", "notification", "error"]
    payload: Dict[str, Any]
    conversation_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    correlation_id: Optional[str] = None  # For request-response matching


# Analytics Models
class AgentInvocation(BaseModel):
    """Agent invocation record."""
    invocation_id: str
    conversation_id: str
    agent_name: str
    invoked_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    status: Literal["running", "completed", "failed", "timeout"]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ToolCall(BaseModel):
    """Tool call record."""
    call_id: str
    agent_invocation_id: str
    tool_name: str
    tool_type: Literal["mcp", "rest"]
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    called_at: datetime
    duration_ms: Optional[int] = None
    status: Literal["running", "completed", "failed"]
    error: Optional[str] = None


# Conversation Models
class Message(BaseModel):
    """Chat message model."""
    message_id: str
    conversation_id: str
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None


class Conversation(BaseModel):
    """Conversation model."""
    conversation_id: str
    prospect_id: Optional[str] = None
    started_at: datetime = Field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    status: Literal["active", "completed", "abandoned"]
    outcome: Optional[Literal["qualified", "not_qualified", "order_placed", "no_action"]] = None
    messages: List[Message] = []
