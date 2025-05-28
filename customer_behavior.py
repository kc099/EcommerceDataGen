# customer_behavior.py
"""
Advanced customer behavior modeling system that simulates realistic
customer interactions, preferences, purchase history, and cart abandonment.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, namedtuple
import random
import math

from config import (
    CUSTOMER_SEGMENTS, AGE_BEHAVIOR_MAPPING, GENDER_PREFERENCES,
    CATEGORIES, CART_STATUSES
)
from gpu_utils import GPUDataGenerator

# Named tuples for structured data
CustomerProfile = namedtuple('CustomerProfile', [
    'customer_id', 'segment', 'price_sensitivity', 'category_preferences',
    'cart_abandonment_rate', 'conversion_base', 'avg_items_per_cart'
])

BrowsingSession = namedtuple('BrowsingSession', [
    'session_id', 'customer_id', 'start_time', 'end_time', 'pages_viewed',
    'categories_browsed', 'session_type'
])

CartEvent = namedtuple('CartEvent', [
    'cart_id', 'customer_id', 'sku', 'quantity', 'price', 'timestamp',
    'event_type', 'session_id'
])


class CustomerBehaviorEngine:
    """
    Comprehensive customer behavior modeling system
    """
    
    def __init__(self, gpu_gen: GPUDataGenerator):
        self.gpu_gen = gpu_gen
        
        # Customer profiles and segments
        self.customer_profiles: Dict[int, CustomerProfile] = {}
        self.purchase_history: Dict[int, List[Dict]] = defaultdict(list)
        self.browsing_sessions: List[BrowsingSession] = []
        self.cart_events: List[CartEvent] = []
        
        # Behavior tracking
        self.category_affinities: Dict[int, Dict[str, float]] = defaultdict(dict)
        self.price_history: Dict[int, List[float]] = defaultdict(list)
        self.session_counter = 1
        self.cart_counter = 1
        
        print("✓ Customer Behavior Engine initialized")
    
    def generate_customer_profiles(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate detailed customer profiles with segments and preferences
        """
        print("Generating customer behavioral profiles...")
        
        n_customers = len(customers_df)
        
        # Assign customer segments
        segment_names = list(CUSTOMER_SEGMENTS.keys())
        segment_weights = [CUSTOMER_SEGMENTS[s]['weight'] for s in segment_names]
        
        customer_segments = self.gpu_gen.random_choice_vectorized(
            segment_names, n_customers, weights=segment_weights
        )
        
        enhanced_customers = []
        
        for i, (_, customer) in enumerate(customers_df.iterrows()):
            customer_id = customer['customer_id']
            age = customer['age']
            gender = customer['gender']
            segment = customer_segments[i]
            
            # Get segment configuration
            segment_config = CUSTOMER_SEGMENTS[segment]
            
            # Calculate age-based behavior modifiers
            age_behavior = self._get_age_behavior(age)
            
            # Generate price sensitivity (segment base + age modifier + randomness)
            base_sensitivity = segment_config['price_sensitivity']
            age_modifier = age_behavior['price_sensitivity']
            price_sensitivity = np.clip(
                base_sensitivity * age_modifier * random.uniform(0.8, 1.2),
                0.1, 1.0
            )
            
            # Generate category preferences
            category_preferences = self._generate_category_preferences(
                segment, gender, age_behavior
            )
            
            # Create customer profile
            profile = CustomerProfile(
                customer_id=customer_id,
                segment=segment,
                price_sensitivity=price_sensitivity,
                category_preferences=category_preferences,
                cart_abandonment_rate=segment_config['cart_abandonment_rate'],
                conversion_base=segment_config['conversion_base'],
                avg_items_per_cart=segment_config['avg_items_per_cart']
            )
            
            self.customer_profiles[customer_id] = profile
            self.category_affinities[customer_id] = category_preferences
            
            # Create enhanced customer record
            enhanced_customer = customer.copy()
            enhanced_customer['segment'] = segment
            enhanced_customer['price_sensitivity'] = price_sensitivity
            enhanced_customer['cart_abandonment_rate'] = segment_config['cart_abandonment_rate']
            enhanced_customer['conversion_base'] = segment_config['conversion_base']
            
            enhanced_customers.append(enhanced_customer)
        
        print(f"✓ Generated profiles for {n_customers:,} customers")
        return pd.DataFrame(enhanced_customers)
    
    def _get_age_behavior(self, age: int) -> Dict[str, float]:
        """Get age-based behavior modifiers"""
        for (min_age, max_age), behavior in AGE_BEHAVIOR_MAPPING.items():
            if min_age <= age <= max_age:
                return behavior
        
        # Default for ages outside defined ranges
        return {
            'tech_affinity': 0.5,
            'price_sensitivity': 0.5,
            'social_influence': 0.3,
            'preferred_categories': ["Electronics", "Home"]
        }
    
    def _generate_category_preferences(self, segment: str, gender: str, 
                                     age_behavior: Dict) -> Dict[str, float]:
        """Generate category preference scores"""
        preferences = {}
        
        # Start with segment preferences
        segment_config = CUSTOMER_SEGMENTS[segment]
        preferred_cats = segment_config['preferred_categories']
        
        # Base preferences for all categories
        for category in CATEGORIES.keys():
            if category in preferred_cats:
                base_score = random.uniform(0.7, 1.0)
            else:
                base_score = random.uniform(0.3, 0.7)
            
            # Apply gender modifier
            gender_modifier = GENDER_PREFERENCES.get(gender, {}).get(category, 1.0)
            
            # Apply age-based tech affinity for Electronics
            if category == "Electronics":
                tech_modifier = age_behavior['tech_affinity']
            else:
                tech_modifier = 1.0
            
            # Calculate final preference
            final_score = base_score * gender_modifier * tech_modifier
            preferences[category] = np.clip(final_score, 0.1, 2.0)
        
        return preferences
    
    def simulate_browsing_session(self, customer_id: int, current_date: date,
                                current_time: datetime) -> Optional[BrowsingSession]:
        """
        Simulate a customer browsing session
        """
        if customer_id not in self.customer_profiles:
            return None
        
        profile = self.customer_profiles[customer_id]
        
        # Determine session type
        session_types = ['casual_browse', 'targeted_search', 'comparison_shop', 'impulse_buy']
        session_weights = [0.4, 0.3, 0.2, 0.1]
        
        session_type = random.choices(session_types, weights=session_weights)[0]
        
        # Generate session duration based on type
        if session_type == 'casual_browse':
            duration_minutes = random.randint(5, 30)
            pages_viewed = random.randint(3, 15)
        elif session_type == 'targeted_search':
            duration_minutes = random.randint(10, 45)
            pages_viewed = random.randint(5, 25)
        elif session_type == 'comparison_shop':
            duration_minutes = random.randint(15, 60)
            pages_viewed = random.randint(8, 40)
        else:  # impulse_buy
            duration_minutes = random.randint(2, 10)
            pages_viewed = random.randint(1, 5)
        
        # Determine categories browsed based on preferences
        num_categories = min(len(CATEGORIES), random.randint(1, 3))
        category_prefs = profile.category_preferences
        
        # Weight categories by preference scores
        categories = list(category_prefs.keys())
        weights = list(category_prefs.values())
        
        categories_browsed = self.gpu_gen.random_choice_vectorized(
            categories, num_categories, weights=weights, replace=False
        )
        
        # Create session
        session = BrowsingSession(
            session_id=self.session_counter,
            customer_id=customer_id,
            start_time=current_time,
            end_time=current_time + timedelta(minutes=duration_minutes),
            pages_viewed=pages_viewed,
            categories_browsed=categories_browsed,
            session_type=session_type
        )
        
        self.session_counter += 1
        self.browsing_sessions.append(session)  # Store in instance
        
        return session
    
    def calculate_view_probability(self, customer_id: int, sku: str, 
                                 category: str, price: float) -> float:
        """
        Calculate probability that customer will view this product
        """
        if customer_id not in self.customer_profiles:
            return 0.5  # Default probability for anonymous users
        
        profile = self.customer_profiles[customer_id]
        
        # Base probability from category preference
        category_pref = profile.category_preferences.get(category, 0.5)
        base_prob = min(0.8, category_pref * 0.4)  # Scale to reasonable view probability
        
        # Price sensitivity adjustment
        if self.price_history[customer_id]:
            avg_price_history = np.mean(self.price_history[customer_id])
            price_ratio = price / avg_price_history
            
            # Higher price sensitivity = lower probability for expensive items
            price_penalty = (price_ratio ** profile.price_sensitivity) if price_ratio > 1 else 1.0
            base_prob = base_prob / price_penalty
        
        # Purchase history boost for similar categories
        history_boost = self._get_category_familiarity_boost(customer_id, category)
        
        final_prob = base_prob * history_boost
        return np.clip(final_prob, 0.01, 0.9)
    
    def simulate_cart_behavior(self, customer_id: int, sku: str, price: float,
                             session: BrowsingSession, view_timestamp: datetime) -> List[CartEvent]:
        """
        Simulate add-to-cart and abandonment behavior
        """
        if customer_id not in self.customer_profiles:
            return []
        
        profile = self.customer_profiles[customer_id]
        cart_events = []
        
        # Probability of adding to cart
        add_to_cart_prob = self._calculate_add_to_cart_probability(
            customer_id, sku, price, session
        )
        
        if random.random() < add_to_cart_prob:
            # Add to cart
            quantity = self._determine_cart_quantity(profile, session.session_type)
            
            cart_event = CartEvent(
                cart_id=self.cart_counter,
                customer_id=customer_id,
                sku=sku,
                quantity=quantity,
                price=price,
                timestamp=view_timestamp + timedelta(seconds=random.randint(10, 300)),
                event_type='add_to_cart',
                session_id=session.session_id
            )
            
            cart_events.append(cart_event)
            self.cart_events.append(cart_event)  # Store in instance
            
            # Determine if cart is abandoned or converted
            abandon_prob = self._calculate_abandonment_probability(
                customer_id, price, session
            )
            
            if random.random() < abandon_prob:
                # Cart abandoned
                abandon_event = CartEvent(
                    cart_id=self.cart_counter,
                    customer_id=customer_id,
                    sku=sku,
                    quantity=quantity,
                    price=price,
                    timestamp=cart_event.timestamp + timedelta(minutes=random.randint(1, 60)),
                    event_type='cart_abandoned',
                    session_id=session.session_id
                )
                cart_events.append(abandon_event)
                self.cart_events.append(abandon_event)  # Store in instance
            else:
                # Potential conversion (will be handled in purchase flow)
                conversion_event = CartEvent(
                    cart_id=self.cart_counter,
                    customer_id=customer_id,
                    sku=sku,
                    quantity=quantity,
                    price=price,
                    timestamp=cart_event.timestamp + timedelta(minutes=random.randint(1, 30)),
                    event_type='ready_to_purchase',
                    session_id=session.session_id
                )
                cart_events.append(conversion_event)
                self.cart_events.append(conversion_event)  # Store in instance
            
            self.cart_counter += 1
        
        return cart_events
    
    def _calculate_add_to_cart_probability(self, customer_id: int, sku: str,
                                         price: float, session: BrowsingSession) -> float:
        """Calculate probability of adding item to cart"""
        profile = self.customer_profiles[customer_id]
        
        # Base probability varies by session type
        session_type_probs = {
            'casual_browse': 0.08,      # Increased from 0.05
            'targeted_search': 0.20,    # Increased from 0.15
            'comparison_shop': 0.12,    # Increased from 0.08
            'impulse_buy': 0.30         # Increased from 0.25
        }
        
        base_prob = session_type_probs.get(session.session_type, 0.12)
        
        # Price sensitivity effect - more reasonable scaling
        if self.price_history[customer_id]:
            avg_price = np.mean(self.price_history[customer_id])
            price_ratio = price / avg_price
            # Gentler price sensitivity effect
            price_factor = math.exp(-profile.price_sensitivity * 0.5 * (price_ratio - 1))
        else:
            # For new customers, use segment-based expectations
            if profile.segment == 'premium':
                expected_price = 15000
            elif profile.segment == 'price_sensitive':
                expected_price = 2000
            elif profile.segment == 'impulse':
                expected_price = 3000
            else:  # mainstream
                expected_price = 8000
            
            price_ratio = price / expected_price
            # Reasonable price sensitivity for new customers
            price_sensitivity_effect = profile.price_sensitivity * 0.3 * (price_ratio - 1)
            price_factor = math.exp(-max(0, price_sensitivity_effect))
        
        # Purchase history boost
        history_boost = 1 + (len(self.purchase_history[customer_id]) * 0.05)
        
        final_prob = base_prob * price_factor * history_boost
        return np.clip(final_prob, 0.01, 0.4)  # Reasonable range
    
    def _calculate_abandonment_probability(self, customer_id: int, price: float,
                                         session: BrowsingSession) -> float:
        """Calculate probability of cart abandonment"""
        profile = self.customer_profiles[customer_id]
        
        # Base abandonment rate from customer profile
        base_rate = profile.cart_abandonment_rate
        
        # Price effect - higher prices increase abandonment
        if self.price_history[customer_id]:
            # Existing customer - compare to their price history
            avg_price = np.mean(self.price_history[customer_id])
            price_ratio = price / avg_price
            # Reasonable price sensitivity: 20% increase in abandonment per 100% price increase
            price_multiplier = 1 + (price_ratio - 1) * profile.price_sensitivity * 0.2
        else:
            # New customer - use segment-based price expectations
            if profile.segment == 'premium':
                expected_price = 15000  # Premium customers expect higher prices
            elif profile.segment == 'price_sensitive':
                expected_price = 2000   # Price-sensitive expect lower prices
            elif profile.segment == 'impulse':
                expected_price = 3000   # Impulse buyers expect moderate prices
            else:  # mainstream
                expected_price = 8000   # Mainstream segment
            
            price_ratio = price / expected_price
            # More reasonable scaling: max 3x multiplier for extremely high prices
            price_effect = (price_ratio - 1) * profile.price_sensitivity * 0.3
            price_multiplier = 1 + max(0, min(price_effect, 2.0))  # Cap at 3x total
        
        # Session type effect
        session_multipliers = {
            'casual_browse': 1.15,     # Slightly higher abandonment
            'targeted_search': 0.85,   # Lower abandonment (more intent)
            'comparison_shop': 1.05,   # Slightly higher (comparison paralysis)
            'impulse_buy': 0.7         # Much lower abandonment
        }
        
        session_multiplier = session_multipliers.get(session.session_type, 1.0)
        
        final_rate = base_rate * price_multiplier * session_multiplier
        
        # More reasonable clipping: 10% minimum, 85% maximum
        return np.clip(final_rate, 0.10, 0.85)
    
    def _determine_cart_quantity(self, profile: CustomerProfile, session_type: str) -> int:
        """Determine quantity to add to cart"""
        if session_type == 'impulse_buy':
            return 1
        
        # Base on average items per cart for customer segment
        avg_items = profile.avg_items_per_cart
        
        # Add some randomness
        quantity = max(1, int(np.random.poisson(avg_items)))
        return min(quantity, 5)  # Cap at 5 items per add
    
    def calculate_conversion_probability(self, customer_id: int, sku: str, 
                                       price: float, is_promotional: bool = False) -> float:
        """
        Calculate final purchase conversion probability
        """
        if customer_id not in self.customer_profiles:
            return 0.02  # Default conversion for anonymous users
        
        profile = self.customer_profiles[customer_id]
        
        # Base conversion rate from customer profile
        base_rate = profile.conversion_base
        
        # Price sensitivity adjustment
        price_factor = self._calculate_price_factor(customer_id, price)
        
        # Promotional boost
        promo_boost = 1.8 if is_promotional else 1.0
        
        # Loyalty boost based on purchase history
        loyalty_boost = 1 + (len(self.purchase_history[customer_id]) * 0.05)
        
        # Category preference boost
        # (This would need category info - simplified for now)
        category_boost = 1.0
        
        final_rate = base_rate * price_factor * promo_boost * loyalty_boost * category_boost
        return np.clip(final_rate, 0.001, 0.15)
    
    def _calculate_price_factor(self, customer_id: int, price: float) -> float:
        """Calculate price factor for conversion probability"""
        profile = self.customer_profiles[customer_id]
        
        if not self.price_history[customer_id]:
            # New customer - use segment-based price expectations
            if profile.segment == 'premium':
                expected_price = 5000
            elif profile.segment == 'price_sensitive':
                expected_price = 1000
            else:
                expected_price = 2500
        else:
            expected_price = np.mean(self.price_history[customer_id])
        
        price_ratio = price / expected_price
        
        # Price sensitivity determines how much price affects conversion
        if price_ratio <= 1:
            # Prices at or below expectation boost conversion
            return 1 + (1 - price_ratio) * (1 - profile.price_sensitivity)
        else:
            # Prices above expectation hurt conversion
            penalty = (price_ratio - 1) * profile.price_sensitivity
            return math.exp(-penalty)
    
    def _get_category_familiarity_boost(self, customer_id: int, category: str) -> float:
        """Get boost based on customer's familiarity with category"""
        category_purchases = sum(
            1 for purchase in self.purchase_history[customer_id]
            if purchase.get('category') == category
        )
        
        # More purchases in category = higher familiarity = higher view probability
        return 1 + (category_purchases * 0.1)
    
    def record_purchase(self, customer_id: int, sku: str, category: str, 
                       price: float, quantity: int, timestamp: datetime):
        """Record a purchase in customer's history"""
        purchase_record = {
            'sku': sku,
            'category': category,
            'price': price,
            'quantity': quantity,
            'timestamp': timestamp
        }
        
        self.purchase_history[customer_id].append(purchase_record)
        self.price_history[customer_id].append(price)
        
        # Update category affinity based on purchase
        if category in self.category_affinities[customer_id]:
            self.category_affinities[customer_id][category] *= 1.1  # Boost preference
    
    def get_customer_segments_summary(self) -> pd.DataFrame:
        """Get summary of customer segments"""
        segment_data = []
        
        for segment, customers in defaultdict(list).items():
            for profile in self.customer_profiles.values():
                if profile.segment == segment:
                    customers.append(profile)
        
        for segment_name, config in CUSTOMER_SEGMENTS.items():
            matching_customers = [p for p in self.customer_profiles.values() 
                                if p.segment == segment_name]
            
            segment_data.append({
                'segment': segment_name,
                'customer_count': len(matching_customers),
                'avg_price_sensitivity': np.mean([c.price_sensitivity for c in matching_customers]) if matching_customers else 0,
                'avg_cart_abandonment': np.mean([c.cart_abandonment_rate for c in matching_customers]) if matching_customers else 0,
                'avg_conversion_base': np.mean([c.conversion_base for c in matching_customers]) if matching_customers else 0,
            })
        
        return pd.DataFrame(segment_data)
    
    def get_browsing_sessions_df(self) -> pd.DataFrame:
        """Get browsing sessions as DataFrame"""
        if not self.browsing_sessions:
            return pd.DataFrame(columns=['session_id', 'customer_id', 'start_time', 
                                       'end_time', 'pages_viewed', 'categories_browsed', 
                                       'session_type'])
        
        sessions_data = []
        for session in self.browsing_sessions:
            session_dict = session._asdict()
            session_dict['categories_browsed'] = ','.join(session.categories_browsed)
            sessions_data.append(session_dict)
        
        return pd.DataFrame(sessions_data)
    
    def get_cart_events_df(self) -> pd.DataFrame:
        """Get cart events as DataFrame"""
        if not self.cart_events:
            return pd.DataFrame(columns=['cart_id', 'customer_id', 'sku', 'quantity', 
                                       'price', 'timestamp', 'event_type', 'session_id'])
        
        return pd.DataFrame([event._asdict() for event in self.cart_events])
    
    def get_customer_lifetime_value(self, customer_id: int) -> Dict[str, Any]:
        """Calculate customer lifetime value metrics"""
        if customer_id not in self.purchase_history:
            return {'total_spent': 0, 'total_orders': 0, 'avg_order_value': 0, 'ltv': 0}
        
        purchases = self.purchase_history[customer_id]
        
        total_spent = sum(p['price'] * p['quantity'] for p in purchases)
        total_orders = len(purchases)
        avg_order_value = total_spent / total_orders if total_orders > 0 else 0
        
        # Simple LTV calculation (total spent * estimated future multiplier)
        if total_orders >= 3:
            ltv_multiplier = 2.5  # Loyal customer
        elif total_orders >= 1:
            ltv_multiplier = 1.5  # Repeat customer
        else:
            ltv_multiplier = 1.0  # New customer
        
        ltv = total_spent * ltv_multiplier
        
        return {
            'total_spent': total_spent,
            'total_orders': total_orders,
            'avg_order_value': avg_order_value,
            'ltv': ltv
        }