# Policy Agent Specification

## Overview

Policy Agents are specialized knowledge agents that use RAG (Retrieval Augmented Generation) to provide policy-based information to operational agents. They act as knowledge specialists, querying policy documents to answer specific questions.

## General Requirements

### Requirement: RAG-Based Knowledge Retrieval

Policy Agents SHALL use RAG to retrieve relevant information from policy documents.

#### Scenario: Query Policy Document
```
GIVEN a policy question from an operational agent
WHEN the Policy Agent receives the query
THEN it SHALL use RAG to search policy documents
AND retrieve top 3 most relevant passages
AND format the information for the requesting agent
```

#### Scenario: No Relevant Information Found
```
GIVEN a policy question
WHEN no relevant information exists in documents
THEN the Policy Agent SHALL return a "not found" response
AND suggest alternative information sources
```

### Requirement: A2A Protocol Communication

Policy Agents SHALL communicate with operational agents via A2A Protocol.

#### Scenario: Receive Policy Query
```
GIVEN an operational agent needs policy information
WHEN it sends an A2A message to a Policy Agent
THEN the Policy Agent SHALL process the query
AND return formatted policy information via A2A response
```

### Requirement: Document Management

Policy Agents SHALL maintain their knowledge base from markdown documents.

#### Scenario: Initialize Knowledge Base
```
GIVEN Policy Agent initialization
WHEN the agent starts
THEN it SHALL load policy documents from rag/documents/
AND create vector embeddings
AND store in ChromaDB collection
```

---

## Product Policy Agent

### Specific Requirements

#### Requirement: Product Catalog Information

The Product Policy Agent SHALL provide accurate product information from the product catalog.

##### Scenario: Product Pricing Query
```
GIVEN a query "What is the pricing for Internet 500?"
WHEN the Product Policy Agent processes the query
THEN it SHALL retrieve pricing information
AND return base price, installation fee, and contract terms
```

##### Scenario: Product Features Query
```
GIVEN a query "What features are included with Internet 1 Gig?"
WHEN the Product Policy Agent processes the query
THEN it SHALL retrieve feature list
AND return all included features and SLA details
```

#### Requirement: Pricing Rules

The Product Policy Agent SHALL provide pricing rules and discount information.

##### Scenario: Volume Discount Query
```
GIVEN a query "What discounts apply for 15 services?"
WHEN the Product Policy Agent processes the query
THEN it SHALL retrieve volume discount rules
AND return applicable discount percentage
```

##### Scenario: Bundle Discount Query
```
GIVEN a query "What discount for Internet + Voice bundle?"
WHEN the Product Policy Agent processes the query
THEN it SHALL retrieve bundle discount rules
AND return discount details
```

#### Requirement: Promotional Offers

The Product Policy Agent SHALL provide current promotional offer information.

##### Scenario: Active Promotions Query
```
GIVEN a query "What promotions are currently available?"
WHEN the Product Policy Agent processes the query
THEN it SHALL retrieve active promotions
AND return promotion codes, discounts, and expiration dates
```

### Connected Agents
- Offer Agent (primary consumer)
- Order Agent (for pricing validation)
- Super Agent (for general product questions)

### Knowledge Base
- Document: `rag/documents/product_policy.md`
- Collection: `product_policy_agent_knowledge`

---

## Order Policy Agent

### Specific Requirements

#### Requirement: Order Processing Rules

The Order Policy Agent SHALL provide order processing procedures and requirements.

##### Scenario: Pre-Order Requirements Query
```
GIVEN a query "What information is required to create an order?"
WHEN the Order Policy Agent processes the query
THEN it SHALL retrieve pre-order requirements
AND return required fields and documentation
```

##### Scenario: Order Validation Query
```
GIVEN a query "What validations are performed on orders?"
WHEN the Order Policy Agent processes the query
THEN it SHALL retrieve validation checklist
AND return all validation steps
```

#### Requirement: Modification and Cancellation Policies

The Order Policy Agent SHALL provide order modification and cancellation rules.

##### Scenario: Modification Policy Query
```
GIVEN a query "Can I modify an order after submission?"
WHEN the Order Policy Agent processes the query
THEN it SHALL retrieve modification policies
AND return allowed modifications and restrictions
```

##### Scenario: Cancellation Fee Query
```
GIVEN a query "What is the cancellation fee?"
WHEN the Order Policy Agent processes the query
THEN it SHALL retrieve cancellation policy
AND return fee structure based on timing
```

#### Requirement: Terms and Conditions

The Order Policy Agent SHALL provide terms and conditions information.

##### Scenario: Contract Terms Query
```
GIVEN a query "What are the contract terms for 24-month agreements?"
WHEN the Order Policy Agent processes the query
THEN it SHALL retrieve contract terms
AND return payment terms, SLAs, and termination clauses
```

### Connected Agents
- Order Agent (primary consumer)
- Post Order Communication Agent
- Super Agent (for order policy questions)

### Knowledge Base
- Document: `rag/documents/order_policy.md`
- Collection: `order_policy_agent_knowledge`

---

## Service Policy Agent

### Specific Requirements

#### Requirement: Coverage and Serviceability

The Service Policy Agent SHALL provide network coverage and serviceability information.

##### Scenario: Coverage Area Query
```
GIVEN a query "Is fiber available in Chicago?"
WHEN the Service Policy Agent processes the query
THEN it SHALL retrieve coverage information
AND return network type and availability
```

##### Scenario: Serviceability Criteria Query
```
GIVEN a query "What makes an address serviceable?"
WHEN the Service Policy Agent processes the query
THEN it SHALL retrieve serviceability criteria
AND return requirements for service availability
```

#### Requirement: SLA Information

The Service Policy Agent SHALL provide Service Level Agreement details.

##### Scenario: Uptime SLA Query
```
GIVEN a query "What is the uptime guarantee for Premium SLA?"
WHEN the Service Policy Agent processes the query
THEN it SHALL retrieve SLA information
AND return uptime percentage and credit structure
```

##### Scenario: Response Time Query
```
GIVEN a query "What is the response time for critical support?"
WHEN the Service Policy Agent processes the query
THEN it SHALL retrieve response time SLAs
AND return response and resolution targets
```

#### Requirement: Network Performance Standards

The Service Policy Agent SHALL provide network performance standards.

##### Scenario: Bandwidth Delivery Query
```
GIVEN a query "Is bandwidth guaranteed?"
WHEN the Service Policy Agent processes the query
THEN it SHALL retrieve bandwidth delivery standards
AND return guarantee details and measurement methods
```

### Connected Agents
- Serviceability Agent (primary consumer)
- Offer Agent (for SLA information)
- Super Agent (for service policy questions)

### Knowledge Base
- Document: `rag/documents/service_policy.md`
- Collection: `service_policy_agent_knowledge`

---

## Fulfillment Policy Agent

### Specific Requirements

#### Requirement: Equipment Information

The Fulfillment Policy Agent SHALL provide equipment catalog and inventory information.

##### Scenario: Equipment Requirements Query
```
GIVEN a query "What equipment is needed for Internet 500?"
WHEN the Fulfillment Policy Agent processes the query
THEN it SHALL retrieve equipment requirements
AND return ONT, router model, and specifications
```

##### Scenario: Equipment Availability Query
```
GIVEN a query "Is the Cisco ISR 4331 router in stock?"
WHEN the Fulfillment Policy Agent processes the query
THEN it SHALL retrieve inventory information
AND return stock status and lead time
```

#### Requirement: Installation Procedures

The Fulfillment Policy Agent SHALL provide installation procedures and timelines.

##### Scenario: Installation Steps Query
```
GIVEN a query "What are the steps for WiFi installation?"
WHEN the Fulfillment Policy Agent processes the query
THEN it SHALL retrieve installation procedures
AND return step-by-step process and duration
```

##### Scenario: Installation Timeline Query
```
GIVEN a query "How long does standard installation take?"
WHEN the Fulfillment Policy Agent processes the query
THEN it SHALL retrieve timeline information
AND return estimated days and scheduling process
```

#### Requirement: Quality Assurance Standards

The Fulfillment Policy Agent SHALL provide quality assurance and testing procedures.

##### Scenario: QA Checklist Query
```
GIVEN a query "What quality checks are performed after installation?"
WHEN the Fulfillment Policy Agent processes the query
THEN it SHALL retrieve QA checklist
AND return all required validation steps
```

### Connected Agents
- Fulfillment Agent (primary consumer)
- Service Activation Agent
- Super Agent (for fulfillment policy questions)

### Knowledge Base
- Document: `rag/documents/fulfillment_policy.md`
- Collection: `fulfillment_policy_agent_knowledge`

---

## Implementation Details

### Framework
- **Base Class**: `PolicyAgent` (to be implemented)
- **RAG System**: ChromaDB with sentence-transformers
- **Communication**: A2A Protocol

### Common Methods
```python
async def query_policy(self, question: str, n_results: int = 3) -> str:
    """Query policy documents and return formatted context."""
    
async def handle_a2a_message(self, message: A2AMessage) -> A2AMessage:
    """Handle incoming A2A policy query messages."""
```

### Configuration Pattern
```yaml
role: "[Policy Type] Policy Specialist"
personality:
  tone: "Precise and authoritative"
  style: "Fact-based and detailed"
domain_knowledge:
  industry: "Cable/Telecommunications B2B"
  expertise:
    - "Policy interpretation"
    - "Document retrieval"
    - "Information synthesis"
connected_agents:
  - name: "[Primary Consumer Agent]"
    communication: "A2A Protocol"
    purpose: "Provide policy information"
connected_tools:
  rag_tools:
    - name: "query_policy_documents"
      usage: "Retrieve relevant policy information"
      when_to_use: "When policy question is received"
```

## Testing Requirements

### Test: Basic Policy Query
```
GIVEN a Policy Agent
WHEN it receives a policy question
THEN it SHALL return relevant information
AND cite source document
```

### Test: Multiple Results
```
GIVEN a broad policy question
WHEN the Policy Agent searches
THEN it SHALL return top 3 most relevant passages
AND rank by relevance
```

### Test: No Results
```
GIVEN a question outside policy scope
WHEN the Policy Agent searches
THEN it SHALL return "not found" response
AND suggest where to find information
```

## Success Metrics

- **Retrieval Accuracy**: >90% relevant results
- **Response Time**: <1 second for policy queries
- **Coverage**: >95% of policy questions answerable
- **Agent Satisfaction**: >90% of operational agents find responses helpful

## Dependencies

- RAG Manager implementation
- Policy documents in markdown format
- A2A Protocol
- Context loader
