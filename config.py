# config.py
"""
Configuration constants for the e-commerce data generator
"""

from datetime import date
from typing import Dict, List, Tuple

# ------------------------------ DEFAULT SETTINGS ----------------------------- #
DEFAULT_CONFIG = {
    'num_customers': 100000,
    'num_products': 50000,
    'num_merchants': 100,
    'start_date': date(2024, 5, 19),
    'end_date': date(2025, 5, 18),
}

# ------------------------------ BUSINESS CONSTANTS --------------------------- #

# Promotional windows with multipliers
PROMO_WINDOWS: Dict[Tuple[int, int], float] = {
    # Republic Day Sale (week centered on 26 Jan)
    (date(1900, 1, 23).timetuple().tm_yday, date(1900, 1, 29).timetuple().tm_yday): 3.0,
    # Holi (approx 18 Mar, ±2 days)
    (date(1900, 3, 16).timetuple().tm_yday, date(1900, 3, 20).timetuple().tm_yday): 2.0,
    # Independence Day Sale
    (date(1900, 8, 13).timetuple().tm_yday, date(1900, 8, 17).timetuple().tm_yday): 2.5,
    # Diwali (around early Nov)
    (date(1900, 10, 28).timetuple().tm_yday, date(1900, 11, 3).timetuple().tm_yday): 4.0,
    # Christmas week
    (date(1900, 12, 23).timetuple().tm_yday, date(1900, 12, 29).timetuple().tm_yday): 1.8,
}

# Out-of-stock rates by category (max 20%)
OOS_RATE_BY_CATEGORY = {
    "Electronics": 0.10,
    "Home": 0.15,
    "Fashion": 0.12,
    "Beauty": 0.18,
    "Sports": 0.14,
}

# Out-of-stock reason hierarchy
OOS_REASON_TREE = {
    "Fulfillment Issues": [
        "PO Not Confirmed by Vendor",
        "PO Confirmed but not fulfilled in full by Vendor",
    ],
    "Ordering Process Issues": [
        "Quantity not forecast",
        "Vendor not available",
        "Quantity not ordered",
        "Quantity < MOQ",
    ],
    "Inventory Issues": [
        "Inventory not moved to Warehouse",
        "Inventory Damaged",
    ],
    "Product Issues": [
        "HAZMAT Block",
        "Poor Quality Block",
        "Vendor Negotiation Block",
    ],
    "Unable to classify": ["Unable to classify"],
}

FLAT_OOS_REASONS = [r for sub in OOS_REASON_TREE.values() for r in sub]

# Fulfillment and order statuses
FULFILLMENT_STATUSES = ["Shipped", "Delivered", "Returned", "Cancelled"]
FULFILLMENT_WEIGHTS = [0.1, 0.7, 0.1, 0.1]

CART_STATUSES = ["active", "abandoned", "converted", "expired"]

# Product categories and subcategories
CATEGORIES = {
    "Electronics": ["Mobiles", "Laptops", "Headphones", "Wearables"],
    "Home": ["Kitchen", "Furniture", "Decor", "Tools"],
    "Fashion": ["Men Topwear", "Women Topwear", "Footwear", "Accessories"],
    "Beauty": ["Skincare", "Haircare", "Fragrances"],
    "Sports": ["Fitness", "Outdoor", "Team Sports"],
}

# Category-specific product attributes
CATEGORY_ATTRIBUTES = {
    "Mobiles": ["Brand", "Screen Size (in)", "Battery (mAh)", "RAM (GB)", "Storage (GB)"],
    "Laptops": ["Brand", "Processor", "RAM (GB)", "SSD (GB)", "Screen Size (in)"],
    "Headphones": ["Brand", "Type", "Battery (hrs)", "Noise Cancelling"],
    "Wearables": ["Brand", "Type", "Battery (days)", "Waterproof"],
    "Kitchen": ["Material", "Brand", "Capacity"],
    "Furniture": ["Material", "Color", "Dimensions"],
    "Decor": ["Material", "Theme", "Color"],
    "Tools": ["Type", "Material", "Brand"],
    "Men Topwear": ["Brand", "Fabric", "Sleeve"],
    "Women Topwear": ["Brand", "Fabric", "Sleeve"],
    "Footwear": ["Brand", "Size", "Material"],
    "Accessories": ["Brand", "Type", "Color"],
    "Skincare": ["Brand", "Skin Type", "Volume (ml)"],
    "Haircare": ["Brand", "Hair Type", "Volume (ml)"],
    "Fragrances": ["Brand", "Volume (ml)", "Fragrance Family"],
    "Fitness": ["Brand", "Type", "Material"],
    "Outdoor": ["Brand", "Type", "Capacity"],
    "Team Sports": ["Sport", "Brand", "Material"],
}

# ------------------------------ CUSTOMER BEHAVIOR CONFIG -------------------- #

# Customer segments with behavior characteristics
CUSTOMER_SEGMENTS = {
    "price_sensitive": {
        "weight": 0.35,
        "price_sensitivity": 0.8,  # Higher = more price sensitive
        "cart_abandonment_rate": 0.75,
        "conversion_base": 0.015,
        "avg_items_per_cart": 2.1,
        "preferred_categories": ["Fashion", "Home"],
    },
    "premium": {
        "weight": 0.20,
        "price_sensitivity": 0.2,
        "cart_abandonment_rate": 0.45,
        "conversion_base": 0.035,
        "avg_items_per_cart": 3.8,
        "preferred_categories": ["Electronics", "Beauty"],
    },
    "mainstream": {
        "weight": 0.30,
        "price_sensitivity": 0.5,
        "cart_abandonment_rate": 0.60,
        "conversion_base": 0.025,
        "avg_items_per_cart": 2.8,
        "preferred_categories": ["Electronics", "Fashion", "Home"],
    },
    "impulse": {
        "weight": 0.15,
        "price_sensitivity": 0.6,
        "cart_abandonment_rate": 0.30,  # Low abandonment, high impulse
        "conversion_base": 0.040,
        "avg_items_per_cart": 1.5,
        "preferred_categories": ["Beauty", "Sports"],
    }
}

# Demographics affecting behavior
AGE_BEHAVIOR_MAPPING = {
    (18, 25): {
        "tech_affinity": 0.9,
        "price_sensitivity": 0.8,
        "social_influence": 0.7,
        "preferred_categories": ["Electronics", "Fashion", "Beauty"],
    },
    (26, 35): {
        "tech_affinity": 0.8,
        "price_sensitivity": 0.6,
        "social_influence": 0.5,
        "preferred_categories": ["Electronics", "Home", "Fashion"],
    },
    (36, 50): {
        "tech_affinity": 0.6,
        "price_sensitivity": 0.5,
        "social_influence": 0.3,
        "preferred_categories": ["Home", "Fashion", "Sports"],
    },
    (51, 65): {
        "tech_affinity": 0.4,
        "price_sensitivity": 0.4,
        "social_influence": 0.2,
        "preferred_categories": ["Home", "Beauty"],
    }
}

GENDER_PREFERENCES = {
    "M": {
        "Electronics": 1.3,
        "Sports": 1.4,
        "Fashion": 0.8,
        "Beauty": 0.3,
        "Home": 0.9,
    },
    "F": {
        "Electronics": 0.8,
        "Sports": 0.7,
        "Fashion": 1.4,
        "Beauty": 1.6,
        "Home": 1.2,
    },
    "O": {
        "Electronics": 1.0,
        "Sports": 1.0,
        "Fashion": 1.0,
        "Beauty": 1.0,
        "Home": 1.0,
    }
}

# ------------------------------ INVENTORY CONFIG ---------------------------- #

# Inventory management settings
INVENTORY_CONFIG = {
    "initial_stock_range": (10, 1000),
    "restock_probability": 0.15,  # Daily probability of restocking
    "restock_multiplier_range": (0.3, 2.0),  # How much to restock relative to recent sales
    "damage_rate": 0.002,  # Daily probability of inventory damage
    "supplier_delay_range": (1, 7),  # Days for supplier to deliver
}

# Merchant behavior patterns
MERCHANT_TYPES = {
    "premium": {
        "weight": 0.2,
        "price_strategy": "premium",  # 10-30% above base
        "stock_levels": "high",
        "fulfillment_speed": "fast",
        "quality_score": 0.95,
    },
    "discount": {
        "weight": 0.3,
        "price_strategy": "discount",  # 10-30% below base
        "stock_levels": "variable",
        "fulfillment_speed": "slow",
        "quality_score": 0.75,
    },
    "mainstream": {
        "weight": 0.5,
        "price_strategy": "competitive",  # ±10% of base
        "stock_levels": "medium",
        "fulfillment_speed": "medium",
        "quality_score": 0.85,
    }
}

# ------------------------------ PERFORMANCE CONFIG -------------------------- #

# GPU optimization settings
GPU_CONFIG = {
    "batch_size_views": 50_000,
    "batch_size_customers": 10_000,
    "batch_size_products": 5_000,
    "memory_cleanup_frequency": 5,  # Clean memory every N batches
}

# Data validation thresholds
VALIDATION_THRESHOLDS = {
    "max_negative_inventory": 0,
    "max_orphaned_records": 0,
    "min_conversion_rate": 0.005,
    "max_conversion_rate": 0.10,
    "min_cart_abandonment": 0.20,
    "max_cart_abandonment": 0.85,  # Updated to match new realistic cap
}