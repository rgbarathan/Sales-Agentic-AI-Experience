"""ADK-based agents (mocked ADK framework)."""
from typing import Dict, Any
import asyncio

from agents.base_agent import BaseAgent
from shared.mock_data import mock_data
from shared.utils import generate_id, simulate_processing_delay
import structlog

logger = structlog.get_logger()


class ProspectAgent(BaseAgent):
    """
    Prospect Agent - Qualifies business prospects.
    
    Framework: ADK (Agent Development Kit)
    Communication: MCP tools for CRM access
    """
    
    def __init__(self):
        super().__init__("prospect_agent")
    
    def get_framework(self) -> str:
        return "ADK"
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Qualify a business prospect.
        
        Args:
            input_data: Prospect information
        
        Returns:
            Qualification result
        """
        logger.info("prospect_qualification_started", input_data=input_data)
        
        # Simulate processing delay
        await simulate_processing_delay()
        
        # Extract prospect data
        company_name = input_data.get("company_name")
        employee_count = input_data.get("employee_count")
        industry = input_data.get("industry")
        
        # Generate or retrieve prospect
        if "prospect_id" in input_data:
            # Retrieve existing prospect (mock)
            prospect = mock_data.generate_prospect()
            prospect.prospect_id = input_data["prospect_id"]
        else:
            # Generate new prospect
            prospect = mock_data.generate_prospect()
            if company_name:
                prospect.company_name = company_name
            if employee_count:
                prospect.employee_count = employee_count
            if industry:
                prospect.industry = industry
        
        # Qualification logic
        qualification_status = "qualified"
        qualification_notes = []
        
        # Check employee count
        if prospect.employee_count < 5:
            qualification_status = "not_qualified"
            qualification_notes.append("Company too small for SMB (< 5 employees)")
        elif prospect.employee_count > 250:
            qualification_status = "needs_review"
            qualification_notes.append("Enterprise opportunity (> 250 employees)")
        
        # Check industry for special requirements
        if prospect.industry in ["healthcare", "finance"]:
            qualification_notes.append(f"Special compliance requirements for {prospect.industry}")
        
        # Build result
        result = {
            "prospect_id": prospect.prospect_id,
            "company_name": prospect.company_name,
            "industry": str(prospect.industry.value) if hasattr(prospect.industry, 'value') else str(prospect.industry),
            "employee_count": prospect.employee_count,
            "company_size": str(prospect.company_size.value) if hasattr(prospect.company_size, 'value') else str(prospect.company_size),
            "qualification_status": qualification_status,
            "qualification_notes": qualification_notes,
            "contact_info": {
                "name": prospect.contact_info.name,
                "email": prospect.contact_info.email,
                "phone": prospect.contact_info.phone,
                "title": prospect.contact_info.title
            },
            "address": {
                "street": prospect.business_address.street,
                "city": prospect.business_address.city,
                "state": prospect.business_address.state,
                "zip_code": prospect.business_address.zip_code
            }
        }
        
        logger.info(
            "prospect_qualified",
            prospect_id=prospect.prospect_id,
            status=qualification_status
        )
        
        return result


class LeadGenerationAgent(BaseAgent):
    """
    Lead Generation Agent - Scores and enriches leads.
    
    Framework: ADK
    Communication: REST API for lead management
    """
    
    def __init__(self):
        super().__init__("lead_generation_agent")
    
    def get_framework(self) -> str:
        return "ADK"
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score and enrich a lead.
        
        Args:
            input_data: Prospect/lead information
        
        Returns:
            Lead scoring result
        """
        logger.info("lead_scoring_started", input_data=input_data)
        
        await simulate_processing_delay()
        
        # Generate lead from prospect data
        lead = mock_data.generate_prospect()  # Using prospect as base for lead
        
        # Use prospect data if provided
        if "company_name" in input_data:
            lead.company_name = input_data["company_name"]
        if "industry" in input_data:
            lead.industry = input_data["industry"]
        if "employee_count" in input_data:
            lead.employee_count = input_data["employee_count"]
        
        # Calculate lead score
        score = mock_data.calculate_lead_score(lead)
        
        # Determine grade based on score
        if score >= 80:
            grade = "A"
        elif score >= 60:
            grade = "B"
        elif score >= 40:
            grade = "C"
        else:
            grade = "D"
        
        # Estimate value based on company size
        estimated_value = lead.annual_revenue * 0.02 if lead.annual_revenue else 5000.0
        probability = score / 100.0
        
        # Enrich with additional data
        result = {
            "lead_id": generate_id(),
            "prospect_id": input_data.get("prospect_id", generate_id()),
            "company_name": lead.company_name,
            "lead_score": score,
            "lead_grade": grade,
            "estimated_value": estimated_value,
            "probability": probability,
            "enrichment_data": {
                "industry": str(lead.industry.value) if hasattr(lead.industry, 'value') else str(lead.industry),
                "employee_count": lead.employee_count,
                "annual_revenue": lead.annual_revenue,
                "technology_stack": ["Cloud Services", "CRM", "VoIP"],
                "current_providers": ["Generic ISP", "Phone Company"]
            },
            "recommended_products": ["Internet 500", "Business Voice Pro"],
            "next_steps": [
                "Validate address serviceability",
                "Generate personalized offer",
                "Schedule discovery call"
            ]
        }
        
        logger.info(
            "lead_scored",
            lead_id=result["lead_id"],
            score=score,
            grade=grade
        )
        
        return result


class OrderAgent(BaseAgent):
    """
    Order Agent - Processes orders.
    
    Framework: ADK
    Communication: REST API for order management
    """
    
    def __init__(self):
        super().__init__("order_agent")
    
    def get_framework(self) -> str:
        return "ADK"
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an order with sub-agent A2A communication.
        
        Flow: Order Agent → Serviceability Agent → Fulfillment Agent
        
        Args:
            input_data: Order information
        
        Returns:
            Order processing result
        """
        logger.info("order_processing_started", input_data=input_data)
        
        await simulate_processing_delay()
        
        # Create order from input
        order_id = generate_id()
        prospect_id = input_data.get("prospect_id")
        products = input_data.get("products", [])
        total_amount = input_data.get("total_amount", 0.0)
        address = input_data.get("address", {})
        
        # Validate order (mock)
        validation_passed = True
        validation_notes = []
        sub_agents_called = []
        
        # Check for required fields
        if not prospect_id:
            validation_passed = False
            validation_notes.append("Missing prospect ID")
        
        if not products:
            validation_passed = False
            validation_notes.append("No products selected")
        
        # SUB-AGENT A2A COMMUNICATION FLOW
        serviceability_result = None
        fulfillment_result = None
        
        if validation_passed and address:
            # Step 1: Order Agent → Serviceability Agent
            try:
                logger.info("order_agent_calling_serviceability", address=address)
                from shared.protocols import a2a_protocol
                
                serviceability_response = await a2a_protocol.send_message(
                    from_agent="order_agent",
                    to_agent="serviceability_agent",
                    message_type="request",
                    payload={
                        "address": address,
                        "requested_products": products
                    }
                )
                
                serviceability_result = serviceability_response
                sub_agents_called.append("serviceability_agent")
                logger.info("serviceability_check_complete", serviceable=serviceability_response.get("serviceable"))
                
                # Only proceed if serviceable
                if serviceability_response.get("serviceable"):
                    # Step 2: Order Agent → Fulfillment Agent
                    try:
                        logger.info("order_agent_calling_fulfillment", order_id=order_id)
                        
                        fulfillment_response = await a2a_protocol.send_message(
                            from_agent="order_agent",
                            to_agent="fulfillment_agent",
                            message_type="request",
                            payload={
                                "order_id": order_id,
                                "prospect_id": prospect_id,
                                "products": products,
                                "address": address
                            }
                        )
                        
                        fulfillment_result = fulfillment_response
                        sub_agents_called.append("fulfillment_agent")
                        logger.info("fulfillment_scheduled", installation_date=fulfillment_response.get("installation_date"))
                        
                    except Exception as e:
                        logger.error("fulfillment_agent_call_failed", error=str(e))
                        validation_notes.append(f"Fulfillment scheduling failed: {str(e)}")
                else:
                    validation_passed = False
                    validation_notes.append("Address not serviceable")
                    
            except Exception as e:
                logger.error("serviceability_agent_call_failed", error=str(e))
                validation_notes.append(f"Serviceability check failed: {str(e)}")
        
        # Set order status
        status = "submitted" if validation_passed else "draft"
        
        from datetime import datetime
        result = {
            "order_id": order_id,
            "prospect_id": prospect_id,
            "products": products,
            "total_amount": total_amount,
            "status": status,
            "validation_passed": validation_passed,
            "validation_notes": validation_notes,
            "created_at": datetime.now().isoformat(),
            "sub_agents_called": sub_agents_called,
            "serviceability_result": serviceability_result,
            "fulfillment_result": fulfillment_result,
            "next_steps": [
                "Send order confirmation",
                f"Installation scheduled: {fulfillment_result.get('installation_date') if fulfillment_result else 'TBD'}",
                "Await installation"
            ] if validation_passed and fulfillment_result else [
                "Complete missing information",
                "Resubmit order"
            ]
        }
        
        logger.info(
            "order_processed",
            order_id=order_id,
            status=status,
            validation_passed=validation_passed,
            sub_agents_called=sub_agents_called
        )
        
        return result
