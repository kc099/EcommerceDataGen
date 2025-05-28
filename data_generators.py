# data_generators.py
"""
Core data generation pipeline with GPU optimization, inventory management,
and customer behavior modeling integration.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional
import random
import gc
from tqdm import tqdm
from faker import Faker

from config import *
from gpu_utils import GPUDataGenerator
from inventory_manager import InventoryManager
from customer_behavior import CustomerBehaviorEngine

fake = Faker("en_IN")


class EcommerceDataPipeline:
    """
    Main data generation pipeline that orchestrates all components
    """
    
    def __init__(self, gpu_gen: GPUDataGenerator):
        self.gpu_gen = gpu_gen
        self.inventory_manager = InventoryManager(gpu_gen)
        self.behavior_engine = CustomerBehaviorEngine(gpu_gen)
        
        # Data storage
        self.customers_df = None
        self.products_df = None
        self.offers_df = None
        
        print("✓ E-commerce Data Pipeline initialized")
    
    def generate_full_dataset(self, num_customers: int, num_products: int,
                            num_merchants: int, start_date: date, end_date: date,
                            avg_views_per_day: int = 5000) -> Dict[str, pd.DataFrame]:
        """
        Generate complete e-commerce dataset with all tables
        """
        print(f"\n{'='*50}")
        print("STARTING FULL DATASET GENERATION")
        print(f"{'='*50}")
        
        # Step 1: Generate basic entities
        self.customers_df = self.generate_customers(num_customers)
        self.products_df = self.generate_products(num_products)
        product_descriptions_df = self.generate_product_descriptions(self.products_df)
        self.offers_df = self.generate_offers(self.products_df, num_merchants)
        
        # Step 2: Initialize systems
        enhanced_offers_df = self.inventory_manager.initialize_inventory(self.offers_df, start_date)
        enhanced_customers_df = self.behavior_engine.generate_customer_profiles(self.customers_df)
        
        # Step 3: Generate interaction data
        views_df = self.generate_views_with_behavior(
            enhanced_customers_df, self.products_df, enhanced_offers_df,
            start_date, end_date, avg_views_per_day
        )
        
        # Step 4: Generate transaction and inventory events
        sales_df = self.generate_sales_with_inventory(views_df, enhanced_offers_df)
        instock_events_df = self.generate_instock_events(views_df)
        
        # Step 5: Generate secondary data
        ratings_df = self.generate_ratings(sales_df)
        
        # Step 6: Get system-generated data
        inventory_events_df = self.inventory_manager.get_inventory_events_df()
        browsing_sessions_df = self.behavior_engine.get_browsing_sessions_df()
        cart_events_df = self.behavior_engine.get_cart_events_df()
        customer_segments_df = self.behavior_engine.get_customer_segments_summary()
        
        # Final dataset
        datasets = {
            'customers.csv': enhanced_customers_df,
            'products.csv': self.products_df,
            'product_descriptions.csv': product_descriptions_df,
            'offers.csv': enhanced_offers_df,
            'views.csv': views_df,
            'instock_events.csv': instock_events_df,
            'sales.csv': sales_df,
            'ratings.csv': ratings_df,
            'inventory_events.csv': inventory_events_df,
            'browsing_sessions.csv': browsing_sessions_df,
            'cart_events.csv': cart_events_df,
            'customer_segments.csv': customer_segments_df,
        }
        
        print(f"\n{'='*50}")
        print("DATASET GENERATION COMPLETED")
        print(f"{'='*50}")
        
        return datasets
    
    def generate_customers(self, n: int) -> pd.DataFrame:
        """Generate customer base data"""
        print(f"Generating {n:,} customers...")
        
        batch_size = min(GPU_CONFIG['batch_size_customers'], n)
        all_customers = []
        
        for start_idx in tqdm(range(0, n, batch_size), desc="Customer batches"):
            end_idx = min(start_idx + batch_size, n)
            batch_size_actual = end_idx - start_idx
            
            # GPU-accelerated random generation
            ages = self.gpu_gen.random_int_vectorized(18, 65, batch_size_actual)
            genders = self.gpu_gen.random_choice_vectorized(["M", "F", "O"], batch_size_actual)
            
            # Generate customer records
            batch_customers = []
            for i in range(batch_size_actual):
                customer_id = start_idx + i + 1
                name = fake.name()
                email = fake.email()
                city = fake.city()
                
                batch_customers.append({
                    'customer_id': customer_id,
                    'name': name,
                    'email': email,
                    'city': city,
                    'age': ages[i],
                    'gender': genders[i]
                })
            
            all_customers.extend(batch_customers)
            
            # Memory management
            if start_idx % (batch_size * 5) == 0:
                gc.collect()
        
        return pd.DataFrame(all_customers)
    
    def generate_products(self, n: int) -> pd.DataFrame:
        """Generate product catalog"""
        print(f"Generating {n:,} products...")
        
        # Vectorized category and subcategory selection
        categories = list(CATEGORIES.keys())
        all_subcategories = []
        category_to_subcats = {}
        
        for cat, subcats in CATEGORIES.items():
            category_to_subcats[cat] = subcats
            all_subcategories.extend(subcats)
        
        # GPU-accelerated generation
        category_choices = self.gpu_gen.random_choice_vectorized(categories, n)
        base_prices = self.gpu_gen.random_uniform_vectorized(100, 100_000, n)
        
        products = []
        for i in tqdm(range(n), desc="Generating products"):
            product_id = i + 1
            sku = f"SKU{product_id:06d}"
            category = category_choices[i]
            subcategory = random.choice(CATEGORIES[category])
            price = round(base_prices[i], 2)
            
            products.append({
                'sku': sku,
                'product_id': product_id,
                'category': category,
                'subcategory': subcategory,
                'base_price': price
            })
        
        return pd.DataFrame(products)
    
    def generate_product_descriptions(self, products_df: pd.DataFrame) -> pd.DataFrame:
        """Generate product attribute descriptions"""
        print("Generating product descriptions...")
        
        descriptions = []
        for _, product in tqdm(products_df.iterrows(), total=len(products_df), 
                              desc="Product descriptions"):
            attributes = CATEGORY_ATTRIBUTES.get(product['subcategory'], ["Brand"])
            
            for attribute in attributes:
                value = fake.word().title()
                descriptions.append({
                    'sku': product['sku'],
                    'description': attribute,
                    'value': value
                })
        
        return pd.DataFrame(descriptions)
    
    def generate_offers(self, products_df: pd.DataFrame, num_merchants: int) -> pd.DataFrame:
        """Generate merchant offers with competitive pricing"""
        print(f"Generating offers for {len(products_df):,} products and {num_merchants} merchants...")
        
        merchants = [f"MER{m:04d}" for m in range(1, num_merchants + 1)]
        offers = []
        
        # Pre-generate random data for efficiency
        total_offers_estimate = len(products_df) * 2.5  # Average 2.5 offers per product
        price_multipliers = self.gpu_gen.random_uniform_vectorized(0.7, 1.2, int(total_offers_estimate))
        inventory_levels = self.gpu_gen.random_int_vectorized(0, 1000, int(total_offers_estimate))
        
        offer_idx = 0
        
        for _, product in tqdm(products_df.iterrows(), total=len(products_df), desc="Generating offers"):
            # Each product offered by 1-4 merchants
            num_merchants_for_product = random.randint(1, min(4, num_merchants))
            selected_merchants = random.sample(merchants, num_merchants_for_product)
            
            for merchant_id in selected_merchants:
                if offer_idx < len(price_multipliers):
                    price = round(product['base_price'] * price_multipliers[offer_idx], 2)
                    inventory = inventory_levels[offer_idx]
                else:
                    price = round(product['base_price'] * random.uniform(0.7, 1.2), 2)
                    inventory = random.randint(0, 1000)
                
                listing_date = fake.date_between(start_date="-2y", end_date="today")
                
                offers.append({
                    'merchant_id': merchant_id,
                    'sku': product['sku'],
                    'inventory': inventory,
                    'offer_price': price,
                    'listing_date': listing_date
                })
                
                offer_idx += 1
        
        return pd.DataFrame(offers)
    
    def generate_views_with_behavior(self, customers_df: pd.DataFrame, 
                                   products_df: pd.DataFrame, offers_df: pd.DataFrame,
                                   start_date: date, end_date: date, 
                                   avg_views_per_day: int) -> pd.DataFrame:
        """
        Generate views with customer behavior modeling and inventory validation
        """
        print("Generating views with customer behavior modeling...")
        
        # Calculate total views with promotional effects
        date_list = self.daterange(start_date, end_date)
        total_views = sum(int(avg_views_per_day * self.promo_multiplier(d)) for d in date_list)
        
        print(f"Total views to generate: {total_views:,}")
        
        # Pre-compute mappings for efficiency
        sku_to_offers = offers_df.groupby('sku')['offer_price'].apply(list).to_dict()
        sku_to_category = products_df.set_index('sku')['category'].to_dict()
        customer_list = customers_df['customer_id'].tolist()
        product_list = products_df['sku'].tolist()
        
        # Generate views in batches
        batch_size = GPU_CONFIG['batch_size_views']
        all_views = []
        
        for batch_start in tqdm(range(0, total_views, batch_size), desc="Processing view batches"):
            batch_end = min(batch_start + batch_size, total_views)
            batch_size_actual = batch_end - batch_start
            
            # Generate batch of views
            batch_views = self._generate_view_batch(
                batch_start, batch_size_actual, date_list, customers_df,
                product_list, sku_to_offers, sku_to_category, customer_list
            )
            
            all_views.extend(batch_views)
            
            # Memory cleanup
            if batch_start % (batch_size * 3) == 0:
                gc.collect()
        
        return pd.DataFrame(all_views)
    
    def _generate_view_batch(self, batch_start: int, batch_size: int, date_list: List[date],
                           customers_df: pd.DataFrame, product_list: List[str],
                           sku_to_offers: Dict, sku_to_category: Dict,
                           customer_list: List[int]) -> List[Dict]:
        """Generate a batch of view records with sessions and cart behavior"""
        
        # Generate weighted dates
        weights = np.array([self.promo_multiplier(d) for d in date_list], dtype=float)
        weights /= weights.sum()
        
        date_indices = self.gpu_gen.xp.random.choice(len(date_list), size=batch_size, p=weights)
        if self.gpu_gen.use_gpu:
            date_indices = self.gpu_gen.to_numpy(date_indices)
        
        selected_dates = [date_list[i] for i in date_indices]
        
        # Generate other random data
        sku_indices = self.gpu_gen.random_int_vectorized(0, len(product_list) - 1, batch_size)
        seconds_in_day = self.gpu_gen.random_int_vectorized(0, 86399, batch_size)
        customer_assignment_probs = self.gpu_gen.random_uniform_vectorized(0, 1, batch_size)
        
        batch_views = []
        active_sessions = {}  # Track active sessions for customers
        
        for i in range(batch_size):
            view_id = batch_start + i + 1
            view_date = selected_dates[i]
            timestamp = datetime.combine(view_date, datetime.min.time()) + timedelta(seconds=int(seconds_in_day[i]))
            
            sku = product_list[sku_indices[i]]
            category = sku_to_category[sku]
            
            # Assign customer (80% probability, but influenced by behavior model)
            if customer_assignment_probs[i] < 0.8:
                customer_id = random.choice(customer_list)
                
                # Create or continue browsing session
                session = self._get_or_create_session(customer_id, view_date, timestamp, active_sessions)
                
                # Use behavior model to influence view probability
                if hasattr(self.behavior_engine, 'calculate_view_probability'):
                    offer_prices = sku_to_offers.get(sku, [])
                    avg_price = np.mean(offer_prices) if offer_prices else 1000
                    
                    view_prob = self.behavior_engine.calculate_view_probability(
                        customer_id, sku, category, avg_price
                    )
                    
                    # Skip this view if behavior model says unlikely
                    if random.random() > view_prob:
                        continue
            else:
                customer_id = None
                session = None
            
            # Check stock availability using inventory manager
            instock, stock_reason = self.inventory_manager.check_sku_availability(sku, category)
            
            # Select price from available offers
            offer_prices = sku_to_offers.get(sku, [])
            if offer_prices and instock:
                price_viewed = random.choice(offer_prices)
            else:
                price_viewed = None
            
            # Generate cart behavior for customer views
            if customer_id is not None and session is not None and price_viewed is not None:
                cart_events = self.behavior_engine.simulate_cart_behavior(
                    customer_id, sku, price_viewed, session, timestamp
                )
                # Cart events are automatically stored in behavior_engine.cart_events
            
            view_record = {
                'http_request_id': view_id,
                'timestamp': timestamp,
                'sku': sku,
                'num_offers': len(offer_prices),
                'instock': instock,
                'price_viewed': price_viewed,
                'customer_id': customer_id
            }
            
            batch_views.append(view_record)
        
        return batch_views
    
    def _get_or_create_session(self, customer_id: int, view_date: date, 
                             timestamp: datetime, active_sessions: Dict) -> Optional[object]:
        """Get existing session or create new one for customer"""
        
        # Check if customer has active session within reasonable time window
        if customer_id in active_sessions:
            last_session = active_sessions[customer_id]
            time_since_last = (timestamp - last_session.end_time).total_seconds()
            
            # If less than 30 minutes since last activity, continue session
            if time_since_last < 1800:  # 30 minutes
                # Update session end time
                last_session = last_session._replace(end_time=timestamp)
                active_sessions[customer_id] = last_session
                return last_session
        
        # Create new session
        session = self.behavior_engine.simulate_browsing_session(
            customer_id, view_date, timestamp
        )
        
        if session:
            active_sessions[customer_id] = session
        
        return session
    
    def generate_sales_with_inventory(self, views_df: pd.DataFrame, 
                                    offers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate sales with proper inventory validation and depletion
        """
        print("Generating sales with inventory validation...")
        
        # Filter views with customers
        customer_views = views_df.dropna(subset=['customer_id']).copy()
        
        if customer_views.empty:
            return pd.DataFrame(columns=[
                'order_id', 'customer_id', 'merchant_id', 'sku', 
                'quantity', 'unit_price', 'order_timestamp', 'fulfilment_status'
            ])
        
        n_customer_views = len(customer_views)
        print(f"Processing {n_customer_views:,} customer views for sales conversion...")
        
        # Create merchant mapping for efficient lookup
        sku_to_merchants = offers_df.groupby('sku')['merchant_id'].apply(list).to_dict()
        
        sales = []
        order_id = 1
        
        # Process views in batches for memory efficiency
        batch_size = min(10000, n_customer_views)
        
        for start_idx in tqdm(range(0, n_customer_views, batch_size), desc="Processing sales batches"):
            end_idx = min(start_idx + batch_size, n_customer_views)
            batch_views = customer_views.iloc[start_idx:end_idx]
            
            batch_sales = self._process_sales_batch(
                batch_views, sku_to_merchants, order_id
            )
            
            sales.extend(batch_sales)
            order_id += len(batch_sales)
            
            # Memory cleanup
            if start_idx % (batch_size * 3) == 0:
                gc.collect()
        
        return pd.DataFrame(sales)
    
    def _process_sales_batch(self, views_batch: pd.DataFrame, 
                           sku_to_merchants: Dict, starting_order_id: int) -> List[Dict]:
        """Process a batch of views for sales conversion"""
        
        batch_size = len(views_batch)
        
        # Generate conversion probabilities using behavior engine
        conversion_probs = []
        for _, view in views_batch.iterrows():
            if pd.notna(view['customer_id']):
                is_promotional = self.promo_multiplier(view['timestamp'].date()) > 1.0
                prob = self.behavior_engine.calculate_conversion_probability(
                    int(view['customer_id']), view['sku'], 
                    view['price_viewed'] or 0, is_promotional
                )
            else:
                prob = 0.02  # Default rate for edge cases
            
            conversion_probs.append(prob)
        
        # Generate random values for conversion decisions
        conversion_randoms = self.gpu_gen.random_uniform_vectorized(0, 1, batch_size)
        
        # Generate quantities and fulfillment statuses
        quantities = self.gpu_gen.random_int_vectorized(1, 3, batch_size)
        status_indices = self.gpu_gen.random_choice_vectorized(
            range(len(FULFILLMENT_STATUSES)), batch_size, weights=FULFILLMENT_WEIGHTS
        )
        
        batch_sales = []
        order_counter = 0
        
        for i, (_, view) in enumerate(views_batch.iterrows()):
            if conversion_randoms[i] < conversion_probs[i]:
                sku = view['sku']
                customer_id = int(view['customer_id'])
                quantity = quantities[i]
                
                # Find merchant with available inventory
                available_merchants = sku_to_merchants.get(sku, [])
                if not available_merchants:
                    continue
                
                # Check inventory availability and select merchant
                merchant_selected = None
                for merchant_id in available_merchants:
                    stock_check = self.inventory_manager.check_stock_availability(
                        merchant_id, sku, quantity
                    )
                    
                    if stock_check.can_fulfill:
                        # Reserve inventory
                        if self.inventory_manager.reserve_inventory(
                            merchant_id, sku, quantity, "order_placed"
                        ):
                            merchant_selected = merchant_id
                            break
                
                if merchant_selected:
                    # Fulfill the order (deplete inventory)
                    if self.inventory_manager.fulfill_order(
                        merchant_selected, sku, quantity, 
                        view['timestamp'], starting_order_id + order_counter
                    ):
                        # Record the sale
                        status = FULFILLMENT_STATUSES[status_indices[i]]
                        
                        sale_record = {
                            'order_id': starting_order_id + order_counter,
                            'customer_id': customer_id,
                            'merchant_id': merchant_selected,
                            'sku': sku,
                            'quantity': quantity,
                            'unit_price': view['price_viewed'],
                            'order_timestamp': view['timestamp'],
                            'fulfilment_status': status
                        }
                        
                        batch_sales.append(sale_record)
                        
                        # Record purchase in customer behavior engine
                        category = self.products_df[self.products_df['sku'] == sku]['category'].iloc[0] if not self.products_df[self.products_df['sku'] == sku].empty else 'Unknown'
                        self.behavior_engine.record_purchase(
                            customer_id, sku, category, view['price_viewed'], 
                            quantity, view['timestamp']
                        )
                        
                        order_counter += 1
        
        return batch_sales
    
    def generate_instock_events(self, views_df: pd.DataFrame) -> pd.DataFrame:
        """Generate stock status events for each view"""
        print("Generating instock events...")
        
        instock_events = []
        
        # Process in batches
        batch_size = min(50000, len(views_df))
        
        for start_idx in tqdm(range(0, len(views_df), batch_size), desc="Instock events"):
            end_idx = min(start_idx + batch_size, len(views_df))
            batch_views = views_df.iloc[start_idx:end_idx]
            
            # Generate batch of OOS reasons
            oos_views = batch_views[~batch_views['instock']]
            if len(oos_views) > 0:
                oos_reasons = self.gpu_gen.random_choice_vectorized(
                    FLAT_OOS_REASONS, len(oos_views)
                )
            
            reason_idx = 0
            for _, view in batch_views.iterrows():
                if view['instock']:
                    reason = "Instock"
                else:
                    reason = oos_reasons[reason_idx] if reason_idx < len(oos_reasons) else "Unable to classify"
                    reason_idx += 1
                
                instock_events.append({
                    'http_request_id': view['http_request_id'],
                    'sku': view['sku'],
                    'reason': reason
                })
        
        return pd.DataFrame(instock_events)
    
    def generate_ratings(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        """Generate customer ratings with GPU optimization"""
        print("Generating ratings...")
        
        if sales_df.empty:
            return pd.DataFrame(columns=[
                'rating_id', 'customer_id', 'sku', 'rating', 'review', 'rating_timestamp'
            ])
        
        n_sales = len(sales_df)
        
        # GPU-accelerated rating generation
        rating_probs = self.gpu_gen.random_uniform_vectorized(0, 1, n_sales)
        ratings = self.gpu_gen.random_int_vectorized(1, 5, n_sales)
        review_delays = self.gpu_gen.random_int_vectorized(1, 30, n_sales)
        
        rating_records = []
        
        for i, (_, sale) in enumerate(tqdm(sales_df.iterrows(), total=n_sales, desc="Processing ratings")):
            if rating_probs[i] < 0.4:  # 40% rating probability
                review_text = fake.sentence(nb_words=random.randint(5, 15))
                review_date = sale['order_timestamp'] + timedelta(days=int(review_delays[i]))
                
                rating_records.append({
                    'rating_id': sale['order_id'],
                    'customer_id': sale['customer_id'],
                    'sku': sale['sku'],
                    'rating': ratings[i],
                    'review': review_text,
                    'rating_timestamp': review_date
                })
        
        return pd.DataFrame(rating_records)
    
    def daterange(self, start: date, end: date) -> List[date]:
        """Generate list of dates in range"""
        from dateutil.rrule import rrule, DAILY
        return [dt.date() for dt in rrule(DAILY, dtstart=start, until=end)]
    
    def promo_multiplier(self, d: date) -> float:
        """Return traffic multiplier for promotional periods"""
        doy = date(1900, d.month, d.day).timetuple().tm_yday
        for (start_doy, end_doy), mult in PROMO_WINDOWS.items():
            if start_doy <= doy <= end_doy:
                return mult
        return 1.4 if d.weekday() >= 5 else 1.0