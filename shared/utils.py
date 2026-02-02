"""Shared utilities and common functions."""
import uuid
from datetime import datetime
from typing import Optional
import asyncio
from config.settings import settings
import random
import string


def generate_id(prefix: str = "ID") -> str:
    """
    Generate a unique ID with optional prefix.
    
    Args:
        prefix: Prefix for the ID (default: "ID")
    
    Returns:
        Unique ID string
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_part = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"{prefix}-{timestamp}-{random_part}"


def get_timestamp() -> datetime:
    """Get current timestamp."""
    return datetime.now()


async def simulate_processing_delay():
    """Simulate processing delay for realistic demo behavior."""
    if settings.enable_mock_delays:
        delay_seconds = settings.mock_delay_ms / 1000.0
        await asyncio.sleep(delay_seconds)


def format_currency(amount: float) -> str:
    """
    Format amount as currency.
    
    Args:
        amount: Amount to format
    
    Returns:
        Formatted currency string
    """
    return f"${amount:,.2f}"


def calculate_discount(
    base_price: float,
    discount_percentage: float
) -> tuple[float, float]:
    """
    Calculate discounted price.
    
    Args:
        base_price: Original price
        discount_percentage: Discount percentage (0-100)
    
    Returns:
        Tuple of (discounted_price, discount_amount)
    """
    discount_amount = base_price * (discount_percentage / 100)
    discounted_price = base_price - discount_amount
    return discounted_price, discount_amount


def determine_priority(score: int) -> str:
    """
    Determine priority level based on score.
    
    Args:
        score: Score value (0-100)
    
    Returns:
        Priority level: "low", "medium", "high", or "urgent"
    """
    if score >= 80:
        return "urgent"
    elif score >= 60:
        return "high"
    elif score >= 40:
        return "medium"
    else:
        return "low"
