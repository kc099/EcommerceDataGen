# inventory_manager.py
"""
Comprehensive inventory management system that tracks stock levels,
validates availability, handles depletion, and simulates restocking events.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, namedtuple
import random

from config import INVENTORY_CONFIG, MERCHANT_TYPES, OOS_RATE_BY_CATEGORY, FLAT_OOS_REASONS
from gpu_utils import GPUDataGenerator

# Named tuples for structured data
InventoryEvent = namedtuple('InventoryEvent', 
                          ['timestamp', 'merchant_id', 'sku', 'event_type', 
                           'quantity', 'reason', 'new_balance'])

StockCheck = namedtuple('StockCheck', 
                       ['merchant_id', 'sku', 'requested_qty', 'available_qty', 
                        'can_fulfill', 'stock_status'])


class InventoryManager:
    """
    Advanced inventory management system with real-time tracking,
    validation, and realistic supply chain simulation.
    """
    
    def __init__(self, gpu_gen: GPUDataGenerator):
        self.gpu_gen = gpu_gen
        
        # Core inventory tracking
        self.current_inventory: Dict[Tuple[str, str], int] = {}  # (merchant_id, sku) -> quantity
        self.reserved_inventory: Dict[Tuple[str, str], int] = {}  # Reserved for pending orders
        
        # Event tracking
        self.inventory_events: List[InventoryEvent] = []
        self.stock_checks: List[StockCheck] = []
        
        # Merchant profiles
        self.merchant_profiles: Dict[str, Dict] = {}
        
        # Supply chain simulation
        self.pending_restocks: List[Dict] = []  # Future restock events
        self.supplier_performance: Dict[str, Dict] = {}  # Supplier reliability metrics
        
        print("✓ Inventory Manager initialized")
    
    def initialize_inventory(self, offers_df: pd.DataFrame, 
                           start_date: date) -> pd.DataFrame:
        """
        Initialize inventory levels and merchant profiles from offers
        """
        print("Initializing inventory system...")
        
        # Generate merchant profiles
        self._generate_merchant_profiles(offers_df)
        
        # Set initial inventory levels
        enhanced_offers = []
        
        for _, offer in offers_df.iterrows():
            merchant_id = offer['merchant_id']
            sku = offer['sku']
            
            # Get merchant type for inventory strategy
            merchant_type = self.merchant_profiles[merchant_id]['type']
            type_config = MERCHANT_TYPES[merchant_type]
            
            # Adjust initial inventory based on merchant type
            base_inventory = offer['inventory']
            if type_config['stock_levels'] == 'high':
                initial_qty = int(base_inventory * random.uniform(1.2, 2.0))
            elif type_config['stock_levels'] == 'low':
                initial_qty = int(base_inventory * random.uniform(0.3, 0.8))
            else:  # medium
                initial_qty = int(base_inventory * random.uniform(0.8, 1.2))
            
            # Store in inventory tracking
            key = (merchant_id, sku)
            self.current_inventory[key] = initial_qty
            self.reserved_inventory[key] = 0
            
            # Record initial stock event
            self.inventory_events.append(InventoryEvent(
                timestamp=datetime.combine(start_date, datetime.min.time()),
                merchant_id=merchant_id,
                sku=sku,
                event_type='initial_stock',
                quantity=initial_qty,
                reason='Initial listing',
                new_balance=initial_qty
            ))
            
            # Create enhanced offer record
            enhanced_offer = offer.copy()
            enhanced_offer['initial_inventory'] = initial_qty
            enhanced_offer['merchant_type'] = merchant_type
            enhanced_offer['quality_score'] = type_config['quality_score']
            enhanced_offers.append(enhanced_offer)
        
        print(f"✓ Initialized inventory for {len(enhanced_offers)} offers")
        return pd.DataFrame(enhanced_offers)
    
    def _generate_merchant_profiles(self, offers_df: pd.DataFrame):
        """Generate realistic merchant profiles"""
        merchants = offers_df['merchant_id'].unique()
        
        # Generate merchant types based on weights
        type_names = list(MERCHANT_TYPES.keys())
        type_weights = [MERCHANT_TYPES[t]['weight'] for t in type_names]
        
        merchant_types = self.gpu_gen.random_choice_vectorized(
            type_names, len(merchants), weights=type_weights
        )
        
        for merchant_id, merchant_type in zip(merchants, merchant_types):
            type_config = MERCHANT_TYPES[merchant_type]
            
            self.merchant_profiles[merchant_id] = {
                'type': merchant_type,
                'price_strategy': type_config['price_strategy'],
                'fulfillment_speed': type_config['fulfillment_speed'],
                'quality_score': type_config['quality_score'],
                'reliability': random.uniform(0.7, 0.98),  # Supplier reliability
                'avg_restock_days': random.randint(3, 14),
            }
    
    def check_stock_availability(self, merchant_id: str, sku: str, 
                               requested_qty: int = 1) -> StockCheck:
        """
        Check if requested quantity is available from specific merchant
        """
        key = (merchant_id, sku)
        available_qty = self.current_inventory.get(key, 0)
        reserved_qty = self.reserved_inventory.get(key, 0)
        
        # Available = current stock - already reserved stock
        truly_available = max(0, available_qty - reserved_qty)
        can_fulfill = truly_available >= requested_qty
        
        # Determine stock status
        if available_qty == 0:
            stock_status = "out_of_stock"
        elif truly_available < requested_qty:
            stock_status = "insufficient_stock"
        elif available_qty < 10:  # Low stock threshold
            stock_status = "low_stock"
        else:
            stock_status = "in_stock"
        
        stock_check = StockCheck(
            merchant_id=merchant_id,
            sku=sku,
            requested_qty=requested_qty,
            available_qty=truly_available,
            can_fulfill=can_fulfill,
            stock_status=stock_status
        )
        
        self.stock_checks.append(stock_check)
        return stock_check
    
    def check_sku_availability(self, sku: str, category: str) -> Tuple[bool, str]:
        """
        Check if SKU is available from any merchant, considering OOS rates
        """
        # Get all merchants carrying this SKU
        merchants_with_sku = [
            (merchant_id, self.current_inventory.get((merchant_id, sku), 0))
            for merchant_id in self.merchant_profiles.keys()
            if (merchant_id, sku) in self.current_inventory
        ]
        
        if not merchants_with_sku:
            return False, "No merchants carry this SKU"
        
        # Check if any merchant has stock
        total_available = sum(qty for _, qty in merchants_with_sku)
        
        if total_available == 0:
            # True out of stock - pick appropriate reason
            return False, random.choice(FLAT_OOS_REASONS)
        
        # Apply category-specific OOS simulation
        # Even if inventory exists, simulate supply chain issues
        oos_rate = OOS_RATE_BY_CATEGORY.get(category, 0.1)
        if random.random() < oos_rate:
            return False, random.choice(FLAT_OOS_REASONS)
        
        return True, "Instock"
    
    def reserve_inventory(self, merchant_id: str, sku: str, 
                         quantity: int, reason: str = "order_placed") -> bool:
        """
        Reserve inventory for a pending order
        """
        stock_check = self.check_stock_availability(merchant_id, sku, quantity)
        
        if not stock_check.can_fulfill:
            return False
        
        key = (merchant_id, sku)
        self.reserved_inventory[key] = self.reserved_inventory.get(key, 0) + quantity
        
        return True
    
    def fulfill_order(self, merchant_id: str, sku: str, quantity: int, 
                     timestamp: datetime, order_id: int) -> bool:
        """
        Fulfill an order by depleting inventory and clearing reservations
        """
        key = (merchant_id, sku)
        
        # Check if we have enough inventory (including reserved)
        current_stock = self.current_inventory.get(key, 0)
        if current_stock < quantity:
            return False
        
        # Deplete inventory
        self.current_inventory[key] = current_stock - quantity
        
        # Clear reservations
        reserved = self.reserved_inventory.get(key, 0)
        self.reserved_inventory[key] = max(0, reserved - quantity)
        
        # Record inventory event
        self.inventory_events.append(InventoryEvent(
            timestamp=timestamp,
            merchant_id=merchant_id,
            sku=sku,
            event_type='sale',
            quantity=-quantity,
            reason=f'Order {order_id}',
            new_balance=self.current_inventory[key]
        ))
        
        return True
    
    def simulate_restocking(self, current_date: date, 
                          sales_velocity: Dict[Tuple[str, str], float]):
        """
        Simulate realistic restocking events based on sales velocity
        """
        restock_events = []
        
        for (merchant_id, sku), current_qty in self.current_inventory.items():
            # Skip if already have enough stock
            if current_qty > 50:  # High stock threshold
                continue
            
            # Get merchant profile
            merchant_profile = self.merchant_profiles.get(merchant_id, {})
            reliability = merchant_profile.get('reliability', 0.8)
            
            # Calculate restock probability based on stock level and sales velocity
            velocity = sales_velocity.get((merchant_id, sku), 0)
            
            # Higher velocity + lower stock = higher restock probability
            base_restock_prob = INVENTORY_CONFIG['restock_probability']
            velocity_factor = min(2.0, velocity * 10)  # Scale velocity impact
            stock_factor = max(0.1, (50 - current_qty) / 50)  # Lower stock = higher probability
            
            restock_prob = base_restock_prob * velocity_factor * stock_factor * reliability
            
            if random.random() < restock_prob:
                # Calculate restock quantity
                recent_sales = max(1, velocity * 7)  # Weekly sales estimate
                min_mult, max_mult = INVENTORY_CONFIG['restock_multiplier_range']
                restock_qty = int(recent_sales * random.uniform(min_mult, max_mult))
                restock_qty = max(10, restock_qty)  # Minimum restock
                
                # Add delivery delay
                delay_days = random.randint(*INVENTORY_CONFIG['supplier_delay_range'])
                delivery_date = current_date + timedelta(days=delay_days)
                
                # Schedule restock
                self.pending_restocks.append({
                    'delivery_date': delivery_date,
                    'merchant_id': merchant_id,
                    'sku': sku,
                    'quantity': restock_qty,
                    'ordered_date': current_date,
                })
                
                restock_events.append((merchant_id, sku, restock_qty, delivery_date))
        
        return restock_events
    
    def process_pending_restocks(self, current_date: date) -> List[InventoryEvent]:
        """
        Process restocks that are due for delivery
        """
        delivered_restocks = []
        remaining_restocks = []
        
        for restock in self.pending_restocks:
            if restock['delivery_date'] <= current_date:
                # Deliver the restock
                merchant_id = restock['merchant_id']
                sku = restock['sku']
                quantity = restock['quantity']
                
                key = (merchant_id, sku)
                old_balance = self.current_inventory.get(key, 0)
                new_balance = old_balance + quantity
                self.current_inventory[key] = new_balance
                
                # Record event
                event = InventoryEvent(
                    timestamp=datetime.combine(current_date, datetime.min.time()),
                    merchant_id=merchant_id,
                    sku=sku,
                    event_type='restock',
                    quantity=quantity,
                    reason='Scheduled delivery',
                    new_balance=new_balance
                )
                
                self.inventory_events.append(event)
                delivered_restocks.append(event)
            else:
                remaining_restocks.append(restock)
        
        self.pending_restocks = remaining_restocks
        return delivered_restocks
    
    def simulate_inventory_damage(self, current_date: date) -> List[InventoryEvent]:
        """
        Simulate random inventory damage/loss events
        """
        damage_events = []
        damage_rate = INVENTORY_CONFIG['damage_rate']
        
        for (merchant_id, sku), current_qty in self.current_inventory.items():
            if current_qty > 0 and random.random() < damage_rate:
                # Calculate damage quantity (1-10% of current stock)
                damage_qty = max(1, int(current_qty * random.uniform(0.01, 0.10)))
                damage_qty = min(damage_qty, current_qty)  # Can't damage more than available
                
                # Apply damage
                new_balance = current_qty - damage_qty
                self.current_inventory[(merchant_id, sku)] = new_balance
                
                # Record event
                damage_reasons = ["Water damage", "Handling damage", "Expiry", "Quality issue", "Theft"]
                event = InventoryEvent(
                    timestamp=datetime.combine(current_date, datetime.min.time()),
                    merchant_id=merchant_id,
                    sku=sku,
                    event_type='damage',
                    quantity=-damage_qty,
                    reason=random.choice(damage_reasons),
                    new_balance=new_balance
                )
                
                self.inventory_events.append(event)
                damage_events.append(event)
        
        return damage_events
    
    def get_inventory_snapshot(self) -> pd.DataFrame:
        """
        Get current inventory levels for all merchant-SKU combinations
        """
        snapshot_data = []
        
        for (merchant_id, sku), current_qty in self.current_inventory.items():
            reserved_qty = self.reserved_inventory.get((merchant_id, sku), 0)
            available_qty = max(0, current_qty - reserved_qty)
            
            snapshot_data.append({
                'merchant_id': merchant_id,
                'sku': sku,
                'current_inventory': current_qty,
                'reserved_inventory': reserved_qty,
                'available_inventory': available_qty,
                'stock_status': 'in_stock' if available_qty > 0 else 'out_of_stock'
            })
        
        return pd.DataFrame(snapshot_data)
    
    def get_inventory_events_df(self) -> pd.DataFrame:
        """
        Get all inventory events as a DataFrame
        """
        if not self.inventory_events:
            return pd.DataFrame(columns=['timestamp', 'merchant_id', 'sku', 'event_type', 
                                       'quantity', 'reason', 'new_balance'])
        
        return pd.DataFrame([event._asdict() for event in self.inventory_events])
    
    def get_stock_checks_df(self) -> pd.DataFrame:
        """
        Get all stock check events as a DataFrame
        """
        if not self.stock_checks:
            return pd.DataFrame(columns=['merchant_id', 'sku', 'requested_qty', 
                                       'available_qty', 'can_fulfill', 'stock_status'])
        
        return pd.DataFrame([check._asdict() for check in self.stock_checks])
    
    def calculate_sales_velocity(self, sales_df: pd.DataFrame, 
                               window_days: int = 7) -> Dict[Tuple[str, str], float]:
        """
        Calculate sales velocity for inventory planning
        """
        if sales_df.empty:
            return {}
        
        # Calculate daily sales rate per merchant-SKU
        sales_velocity = {}
        
        for (merchant_id, sku), group in sales_df.groupby(['merchant_id', 'sku']):
            total_qty = group['quantity'].sum()
            date_range = (group['order_timestamp'].max() - group['order_timestamp'].min()).days
            date_range = max(1, date_range)  # Avoid division by zero
            
            daily_velocity = total_qty / date_range
            sales_velocity[(merchant_id, sku)] = daily_velocity
        
        return sales_velocity
    
    def get_low_stock_alerts(self, threshold: int = 10) -> List[Dict]:
        """
        Get alerts for low stock items
        """
        alerts = []
        
        for (merchant_id, sku), current_qty in self.current_inventory.items():
            if current_qty <= threshold:
                alerts.append({
                    'merchant_id': merchant_id,
                    'sku': sku,
                    'current_qty': current_qty,
                    'alert_level': 'critical' if current_qty == 0 else 'warning'
                })
        
        return alerts
    
    def cleanup_old_events(self, cutoff_date: date):
        """
        Clean up old events to manage memory
        """
        cutoff_datetime = datetime.combine(cutoff_date, datetime.min.time())
        
        # Keep only recent events
        self.inventory_events = [
            event for event in self.inventory_events 
            if event.timestamp >= cutoff_datetime
        ]
        
        # Keep only recent stock checks (last 1000)
        if len(self.stock_checks) > 1000:
            self.stock_checks = self.stock_checks[-1000:]