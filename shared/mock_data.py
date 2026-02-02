"""Mock data generators for the Agentic AI Sales System."""
import random
from datetime import datetime, timedelta
from typing import List
from faker import Faker
from .models import (
    Address, ContactInfo, ProspectData, Product, ServiceAvailability,
    ServiceabilityResult, IndustryType, CompanySize, ServiceType
)

# Initialize Faker
fake = Faker()
Faker.seed(42)
random.seed(42)


class MockDataGenerator:
    """Generate realistic mock data for demo purposes."""
    
    # Sample data pools
    COMPANY_SUFFIXES = ["Inc", "LLC", "Corp", "Group", "Solutions", "Services", "Technologies"]
    
    STREET_TYPES = ["St", "Ave", "Blvd", "Dr", "Ln", "Rd", "Way", "Ct"]
    
    CITIES = [
        ("New York", "NY"), ("Los Angeles", "CA"), ("Chicago", "IL"),
        ("Houston", "TX"), ("Phoenix", "AZ"), ("Philadelphia", "PA"),
        ("San Antonio", "TX"), ("San Diego", "CA"), ("Dallas", "TX"),
        ("Austin", "TX"), ("Boston", "MA"), ("Seattle", "WA"),
        ("Denver", "CO"), ("Atlanta", "GA"), ("Miami", "FL")
    ]
    
    BUSINESS_TITLES = [
        "Owner", "CEO", "President", "VP Operations", "IT Manager",
        "Office Manager", "General Manager", "Director of IT"
    ]
    
    @staticmethod
    def generate_address(serviceable: bool = True) -> Address:
        """Generate a random business address."""
        city, state = random.choice(MockDataGenerator.CITIES)
        
        # If serviceable, use specific zip codes in our "coverage area"
        if serviceable:
            zip_code = random.choice([
                "10001", "10002", "90001", "60601", "77001",
                "85001", "19101", "78201", "92101", "75201"
            ])
        else:
            zip_code = fake.zipcode()
        
        return Address(
            street=f"{fake.building_number()} {fake.street_name()} {random.choice(MockDataGenerator.STREET_TYPES)}",
            city=city,
            state=state,
            zip_code=zip_code,
            country="USA",
            latitude=fake.latitude(),
            longitude=fake.longitude()
        )
    
    @staticmethod
    def generate_contact_info() -> ContactInfo:
        """Generate random contact information."""
        first_name = fake.first_name()
        last_name = fake.last_name()
        
        return ContactInfo(
            name=f"{first_name} {last_name}",
            email=f"{first_name.lower()}.{last_name.lower()}@{fake.domain_name()}",
            phone=fake.phone_number(),
            title=random.choice(MockDataGenerator.BUSINESS_TITLES)
        )
    
    @staticmethod
    def generate_prospect(prospect_id: str = None) -> ProspectData:
        """Generate a random prospect."""
        if prospect_id is None:
            prospect_id = f"PROS-{fake.unique.random_number(digits=6)}"
        
        industry = random.choice(list(IndustryType))
        company_size = random.choice(list(CompanySize))
        
        # Determine employee count based on size
        if company_size == CompanySize.MICRO:
            employee_count = random.randint(1, 10)
        elif company_size == CompanySize.SMALL:
            employee_count = random.randint(11, 50)
        else:  # MEDIUM
            employee_count = random.randint(51, 250)
        
        # Generate revenue based on size
        revenue_multiplier = {
            CompanySize.MICRO: (100_000, 500_000),
            CompanySize.SMALL: (500_000, 5_000_000),
            CompanySize.MEDIUM: (5_000_000, 50_000_000)
        }
        min_rev, max_rev = revenue_multiplier[company_size]
        annual_revenue = random.uniform(min_rev, max_rev)
        
        company_name = f"{fake.company().split(',')[0]} {random.choice(MockDataGenerator.COMPANY_SUFFIXES)}"
        
        return ProspectData(
            prospect_id=prospect_id,
            company_name=company_name,
            industry=industry,
            company_size=company_size,
            employee_count=employee_count,
            annual_revenue=annual_revenue,
            business_address=MockDataGenerator.generate_address(serviceable=True),
            contact_info=MockDataGenerator.generate_contact_info(),
            website=f"https://www.{company_name.lower().replace(' ', '')}.com",
            existing_customer=random.random() < 0.1  # 10% chance of existing customer
        )
    
    @staticmethod
    def generate_products() -> List[Product]:
        """Generate product catalog."""
        products = [
            # Internet Products
            Product(
                product_id="INET-100",
                name="Business Internet 100",
                service_type=ServiceType.INTERNET,
                description="100 Mbps symmetric fiber internet for small businesses",
                bandwidth_mbps=100,
                base_price_monthly=79.99,
                installation_fee=99.00,
                contract_term_months=12,
                features=["99.9% uptime SLA", "Static IP included", "24/7 support"]
            ),
            Product(
                product_id="INET-500",
                name="Business Internet 500",
                service_type=ServiceType.INTERNET,
                description="500 Mbps symmetric fiber internet for growing businesses",
                bandwidth_mbps=500,
                base_price_monthly=149.99,
                installation_fee=99.00,
                contract_term_months=12,
                features=["99.9% uptime SLA", "5 static IPs", "24/7 priority support", "Free router"]
            ),
            Product(
                product_id="INET-1000",
                name="Business Internet 1 Gig",
                service_type=ServiceType.INTERNET,
                description="1 Gbps symmetric fiber internet for bandwidth-intensive businesses",
                bandwidth_mbps=1000,
                base_price_monthly=249.99,
                installation_fee=149.00,
                contract_term_months=12,
                features=["99.99% uptime SLA", "13 static IPs", "24/7 priority support", "Enterprise router", "Dedicated account manager"]
            ),
            
            # Voice Products
            Product(
                product_id="VOICE-BASIC",
                name="Business Voice Basic",
                service_type=ServiceType.VOICE,
                description="Cloud-based business phone system with essential features",
                base_price_monthly=29.99,
                installation_fee=0.00,
                contract_term_months=12,
                features=["Unlimited calling", "Voicemail", "Call forwarding", "Mobile app"]
            ),
            Product(
                product_id="VOICE-PRO",
                name="Business Voice Pro",
                service_type=ServiceType.VOICE,
                description="Advanced business phone system with collaboration tools",
                base_price_monthly=49.99,
                installation_fee=0.00,
                contract_term_months=12,
                features=["All Basic features", "Auto attendant", "Call analytics", "Video conferencing", "Team messaging"]
            ),
            
            # Managed Services
            Product(
                product_id="WIFI-MANAGED",
                name="Managed WiFi",
                service_type=ServiceType.MANAGED_WIFI,
                description="Enterprise-grade WiFi with professional installation and management",
                base_price_monthly=99.99,
                installation_fee=299.00,
                contract_term_months=24,
                features=["Professional installation", "Enterprise access points", "Guest network", "Usage analytics", "Remote management"]
            ),
            Product(
                product_id="SEC-MANAGED",
                name="Managed Security",
                service_type=ServiceType.MANAGED_SECURITY,
                description="Comprehensive network security with firewall and threat protection",
                base_price_monthly=149.99,
                installation_fee=199.00,
                contract_term_months=24,
                features=["Next-gen firewall", "Intrusion prevention", "Content filtering", "24/7 monitoring", "Monthly security reports"]
            ),
            Product(
                product_id="BACKUP-CLOUD",
                name="Cloud Backup",
                service_type=ServiceType.CLOUD_BACKUP,
                description="Automated cloud backup for business-critical data",
                base_price_monthly=79.99,
                installation_fee=0.00,
                contract_term_months=12,
                features=["1TB storage", "Automated daily backups", "Encryption", "Quick restore", "Unlimited devices"]
            ),
        ]
        
        return products
    
    @staticmethod
    def check_serviceability(address: Address) -> ServiceabilityResult:
        """Check if an address is serviceable (mock)."""
        # Addresses in our coverage area zip codes are serviceable
        serviceable_zips = [
            "10001", "10002", "90001", "60601", "77001",
            "85001", "19101", "78201", "92101", "75201"
        ]
        
        is_serviceable = address.zip_code in serviceable_zips
        
        if is_serviceable:
            # Determine network type randomly
            network_type = random.choice(["fiber", "fiber", "hybrid"])  # Favor fiber
            
            # Generate available services
            available_services = [
                ServiceAvailability(
                    service_type=ServiceType.INTERNET,
                    available=True,
                    max_bandwidth_mbps=1000 if network_type == "fiber" else 500,
                    installation_fee=99.00
                ),
                ServiceAvailability(
                    service_type=ServiceType.VOICE,
                    available=True,
                    installation_fee=0.00
                ),
                ServiceAvailability(
                    service_type=ServiceType.MANAGED_WIFI,
                    available=True,
                    installation_fee=299.00
                ),
                ServiceAvailability(
                    service_type=ServiceType.MANAGED_SECURITY,
                    available=True,
                    installation_fee=199.00
                ),
                ServiceAvailability(
                    service_type=ServiceType.CLOUD_BACKUP,
                    available=True,
                    installation_fee=0.00
                ),
            ]
            
            estimated_install_days = random.randint(7, 14)
        else:
            available_services = []
            network_type = None
            estimated_install_days = None
        
        return ServiceabilityResult(
            address=address,
            is_serviceable=is_serviceable,
            available_services=available_services,
            network_type=network_type,
            estimated_install_days=estimated_install_days
        )
    
    @staticmethod
    def calculate_lead_score(prospect: ProspectData) -> int:
        """Calculate a lead score based on prospect attributes."""
        score = 50  # Base score
        
        # Size scoring
        if prospect.company_size == CompanySize.MEDIUM:
            score += 20
        elif prospect.company_size == CompanySize.SMALL:
            score += 10
        
        # Industry scoring (some industries are better targets)
        high_value_industries = [
            IndustryType.TECHNOLOGY,
            IndustryType.FINANCE,
            IndustryType.HEALTHCARE,
            IndustryType.PROFESSIONAL_SERVICES
        ]
        if prospect.industry in high_value_industries:
            score += 15
        
        # Revenue scoring
        if prospect.annual_revenue:
            if prospect.annual_revenue > 10_000_000:
                score += 15
            elif prospect.annual_revenue > 1_000_000:
                score += 10
        
        # Existing customer penalty
        if prospect.existing_customer:
            score -= 30
        
        return max(0, min(100, score))  # Clamp between 0-100


# Create a global instance
mock_data = MockDataGenerator()
