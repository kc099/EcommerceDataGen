# data_validation.py
"""
Comprehensive data validation system to ensure data quality,
referential integrity, and business logic compliance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import warnings

from config import VALIDATION_THRESHOLDS, CATEGORIES, FULFILLMENT_STATUSES


class DataValidator:
    """
    Comprehensive data validation for e-commerce datasets
    """
    
    def __init__(self):
        self.validation_results = {}
        self.issues_found = []
        
    def validate_full_dataset(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run comprehensive validation on all datasets
        """
        print("Running comprehensive data validation...")
        
        self.issues_found = []
        validation_results = {}
        
        # Individual table validations
        for table_name, df in datasets.items():
            table_issues = self.validate_table_structure(table_name, df)
            if table_issues:
                self.issues_found.extend([f"{table_name}: {issue}" for issue in table_issues])
        
        # Cross-table validations
        referential_issues = self.validate_referential_integrity(datasets)
        self.issues_found.extend(referential_issues)
        
        # Business logic validations
        business_issues = self.validate_business_logic(datasets)
        self.issues_found.extend(business_issues)
        
        # Inventory consistency validations
        inventory_issues = self.validate_inventory_consistency(datasets)
        self.issues_found.extend(inventory_issues)
        
        # Customer behavior validations
        behavior_issues = self.validate_customer_behavior(datasets)
        self.issues_found.extend(behavior_issues)
        
        # Statistical validations
        stats_issues = self.validate_statistical_distributions(datasets)
        self.issues_found.extend(stats_issues)
        
        # Compile results
        validation_results = {
            'is_valid': len(self.issues_found) == 0,
            'total_issues': len(self.issues_found),
            'issues': self.issues_found,
            'dataset_summary': self._generate_dataset_summary(datasets),
            'validation_metrics': self._calculate_validation_metrics(datasets)
        }
        
        return validation_results
    
    def validate_table_structure(self, table_name: str, df: pd.DataFrame) -> List[str]:
        """Validate individual table structure and data quality"""
        issues = []
        
        if df.empty:
            issues.append(f"Table is empty")
            return issues
        
        # Table-specific validations
        if table_name == 'customers.csv':
            issues.extend(self._validate_customers(df))
        elif table_name == 'products.csv':
            issues.extend(self._validate_products(df))
        elif table_name == 'offers.csv':
            issues.extend(self._validate_offers(df))
        elif table_name == 'views.csv':
            issues.extend(self._validate_views(df))
        elif table_name == 'sales.csv':
            issues.extend(self._validate_sales(df))
        elif table_name == 'inventory_events.csv':
            issues.extend(self._validate_inventory_events(df))
        elif table_name == 'cart_events.csv':
            issues.extend(self._validate_cart_events(df))
        
        return issues
    
    def _validate_customers(self, df: pd.DataFrame) -> List[str]:
        """Validate customers table"""
        issues = []
        
        # Required columns
        required_cols = ['customer_id', 'name', 'email', 'city', 'age', 'gender']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        if 'customer_id' in df.columns:
            # Unique customer IDs
            if df['customer_id'].duplicated().any():
                issues.append("Duplicate customer IDs found")
            
            # Customer ID range
            if df['customer_id'].min() < 1:
                issues.append("Customer IDs should start from 1")
        
        if 'age' in df.columns:
            # Age validation
            if df['age'].min() < 18 or df['age'].max() > 100:
                issues.append("Invalid age ranges found")
        
        if 'gender' in df.columns:
            # Gender validation
            valid_genders = {'M', 'F', 'O'}
            invalid_genders = set(df['gender'].unique()) - valid_genders
            if invalid_genders:
                issues.append(f"Invalid gender values: {invalid_genders}")
        
        if 'email' in df.columns:
            # Email format validation (basic)
            invalid_emails = df[~df['email'].str.contains('@', na=False)]
            if not invalid_emails.empty:
                issues.append(f"Invalid email formats found: {len(invalid_emails)} records")
        
        return issues
    
    def _validate_products(self, df: pd.DataFrame) -> List[str]:
        """Validate products table"""
        issues = []
        
        required_cols = ['sku', 'product_id', 'category', 'subcategory', 'base_price']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        if 'sku' in df.columns:
            # Unique SKUs
            if df['sku'].duplicated().any():
                issues.append("Duplicate SKUs found")
        
        if 'category' in df.columns:
            # Valid categories
            invalid_categories = set(df['category'].unique()) - set(CATEGORIES.keys())
            if invalid_categories:
                issues.append(f"Invalid categories: {invalid_categories}")
        
        if 'base_price' in df.columns:
            # Price validation
            if df['base_price'].min() <= 0:
                issues.append("Zero or negative prices found")
            
            if df['base_price'].max() > 1_000_000:
                issues.append("Extremely high prices found (>1M)")
        
        # Category-subcategory consistency
        if 'category' in df.columns and 'subcategory' in df.columns:
            for _, row in df.iterrows():
                if row['subcategory'] not in CATEGORIES.get(row['category'], []):
                    issues.append(f"Invalid subcategory '{row['subcategory']}' for category '{row['category']}'")
                    break  # Don't spam with all invalid combinations
        
        return issues
    
    def _validate_offers(self, df: pd.DataFrame) -> List[str]:
        """Validate offers table"""
        issues = []
        
        required_cols = ['merchant_id', 'sku', 'inventory', 'offer_price']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        if 'inventory' in df.columns:
            # Inventory validation
            if df['inventory'].min() < 0:
                issues.append("Negative inventory found")
        
        if 'offer_price' in df.columns:
            # Price validation
            if df['offer_price'].min() <= 0:
                issues.append("Zero or negative offer prices found")
        
        # Duplicate merchant-SKU combinations
        if 'merchant_id' in df.columns and 'sku' in df.columns:
            duplicates = df.duplicated(subset=['merchant_id', 'sku'])
            if duplicates.any():
                issues.append("Duplicate merchant-SKU combinations found")
        
        return issues
    
    def _validate_views(self, df: pd.DataFrame) -> List[str]:
        """Validate views table"""
        issues = []
        
        required_cols = ['http_request_id', 'timestamp', 'sku', 'instock']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        if 'http_request_id' in df.columns:
            # Unique request IDs
            if df['http_request_id'].duplicated().any():
                issues.append("Duplicate HTTP request IDs found")
        
        if 'timestamp' in df.columns:
            # Timestamp validation
            try:
                pd.to_datetime(df['timestamp'])
            except:
                issues.append("Invalid timestamp formats found")
        
        if 'instock' in df.columns:
            # Stock status validation
            if not df['instock'].dtype == bool and not set(df['instock'].unique()).issubset({True, False, 0, 1}):
                issues.append("Invalid instock values (should be boolean)")
        
        if 'price_viewed' in df.columns:
            # Price consistency with stock status
            instock_with_no_price = df[(df['instock'] == True) & (df['price_viewed'].isna())]
            if not instock_with_no_price.empty:
                issues.append(f"In-stock items without prices: {len(instock_with_no_price)} records")
        
        return issues
    
    def _validate_sales(self, df: pd.DataFrame) -> List[str]:
        """Validate sales table"""
        issues = []
        
        required_cols = ['order_id', 'customer_id', 'merchant_id', 'sku', 'quantity', 'unit_price']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        if 'order_id' in df.columns:
            # Unique order IDs
            if df['order_id'].duplicated().any():
                issues.append("Duplicate order IDs found")
        
        if 'quantity' in df.columns:
            # Quantity validation
            if df['quantity'].min() <= 0:
                issues.append("Zero or negative quantities found")
            
            if df['quantity'].max() > 100:
                issues.append("Extremely high quantities found (>100)")
        
        if 'unit_price' in df.columns:
            # Price validation
            if df['unit_price'].min() <= 0:
                issues.append("Zero or negative unit prices found")
        
        if 'fulfilment_status' in df.columns:
            # Status validation
            invalid_statuses = set(df['fulfilment_status'].unique()) - set(FULFILLMENT_STATUSES)
            if invalid_statuses:
                issues.append(f"Invalid fulfillment statuses: {invalid_statuses}")
        
        return issues
    
    def _validate_inventory_events(self, df: pd.DataFrame) -> List[str]:
        """Validate inventory events table"""
        issues = []
        
        if df.empty:
            return issues
        
        required_cols = ['timestamp', 'merchant_id', 'sku', 'event_type', 'quantity']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        if 'event_type' in df.columns:
            # Valid event types
            valid_events = {'initial_stock', 'restock', 'sale', 'damage', 'adjustment'}
            invalid_events = set(df['event_type'].unique()) - valid_events
            if invalid_events:
                issues.append(f"Invalid inventory event types: {invalid_events}")
        
        if 'quantity' in df.columns and 'event_type' in df.columns:
            # Quantity signs should match event types
            positive_events = ['initial_stock', 'restock', 'adjustment']
            negative_events = ['sale', 'damage']
            
            for event_type in positive_events:
                negative_qty = df[(df['event_type'] == event_type) & (df['quantity'] < 0)]
                if not negative_qty.empty:
                    issues.append(f"Negative quantities found for {event_type} events")
            
            for event_type in negative_events:
                positive_qty = df[(df['event_type'] == event_type) & (df['quantity'] > 0)]
                if not positive_qty.empty:
                    issues.append(f"Positive quantities found for {event_type} events")
        
        return issues
    
    def _validate_cart_events(self, df: pd.DataFrame) -> List[str]:
        """Validate cart events table"""
        issues = []
        
        if df.empty:
            return issues
        
        required_cols = ['cart_id', 'customer_id', 'sku', 'event_type', 'timestamp']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        if 'event_type' in df.columns:
            # Valid event types
            valid_events = {'add_to_cart', 'cart_abandoned', 'ready_to_purchase', 'removed_from_cart'}
            invalid_events = set(df['event_type'].unique()) - valid_events
            if invalid_events:
                issues.append(f"Invalid cart event types: {invalid_events}")
        
        return issues
    
    def validate_referential_integrity(self, datasets: Dict[str, pd.DataFrame]) -> List[str]:
        """Validate referential integrity across tables"""
        issues = []
        
        # Customer references
        if 'views.csv' in datasets and 'customers.csv' in datasets:
            view_customers = datasets['views.csv']['customer_id'].dropna().unique()
            valid_customers = datasets['customers.csv']['customer_id'].unique()
            invalid_refs = set(view_customers) - set(valid_customers)
            if invalid_refs:
                issues.append(f"Invalid customer references in views: {len(invalid_refs)} customers")
        
        # Product/SKU references
        if 'views.csv' in datasets and 'products.csv' in datasets:
            view_skus = datasets['views.csv']['sku'].unique()
            valid_skus = datasets['products.csv']['sku'].unique()
            invalid_refs = set(view_skus) - set(valid_skus)
            if invalid_refs:
                issues.append(f"Invalid SKU references in views: {len(invalid_refs)} SKUs")
        
        # Offer references
        if 'sales.csv' in datasets and 'offers.csv' in datasets:
            sales_merchant_sku = set(zip(datasets['sales.csv']['merchant_id'], 
                                       datasets['sales.csv']['sku']))
            valid_merchant_sku = set(zip(datasets['offers.csv']['merchant_id'], 
                                       datasets['offers.csv']['sku']))
            invalid_refs = sales_merchant_sku - valid_merchant_sku
            if invalid_refs:
                issues.append(f"Invalid merchant-SKU references in sales: {len(invalid_refs)} combinations")
        
        return issues
    
    def validate_business_logic(self, datasets: Dict[str, pd.DataFrame]) -> List[str]:
        """Validate business logic rules"""
        issues = []
        
        # Sales should only occur for in-stock items (with some tolerance for timing)
        if 'sales.csv' in datasets and 'views.csv' in datasets:
            sales_df = datasets['sales.csv']
            views_df = datasets['views.csv']
            
            # Check for sales of out-of-stock items
            for _, sale in sales_df.iterrows():
                # Find corresponding view (approximate timestamp matching)
                matching_views = views_df[
                    (views_df['sku'] == sale['sku']) & 
                    (views_df['customer_id'] == sale['customer_id']) &
                    (abs((views_df['timestamp'] - sale['order_timestamp']).dt.total_seconds()) < 3600)  # Within 1 hour
                ]
                
                if not matching_views.empty:
                    out_of_stock_sales = matching_views[matching_views['instock'] == False]
                    if not out_of_stock_sales.empty:
                        issues.append(f"Sale of out-of-stock item: Order {sale['order_id']}")
                        break  # Don't spam with all instances
        
        # Price consistency between views and sales
        if 'sales.csv' in datasets and 'views.csv' in datasets:
            sales_df = datasets['sales.csv']
            views_df = datasets['views.csv']
            
            # Sample check for price consistency (not exhaustive due to performance)
            sample_sales = sales_df.head(100)  # Check first 100 sales
            price_mismatches = 0
            
            for _, sale in sample_sales.iterrows():
                matching_views = views_df[
                    (views_df['sku'] == sale['sku']) & 
                    (views_df['customer_id'] == sale['customer_id'])
                ]
                
                if not matching_views.empty:
                    view_prices = matching_views['price_viewed'].dropna()
                    if not view_prices.empty and sale['unit_price'] not in view_prices.values:
                        price_mismatches += 1
            
            if price_mismatches > 5:  # Allow some tolerance
                issues.append(f"Price mismatches between views and sales: {price_mismatches} in sample")
        
        return issues
    
    def validate_inventory_consistency(self, datasets: Dict[str, pd.DataFrame]) -> List[str]:
        """Validate inventory consistency"""
        issues = []
        
        if 'inventory_events.csv' not in datasets:
            return issues
        
        inventory_events = datasets['inventory_events.csv']
        
        # Check for negative inventory balances
        if 'new_balance' in inventory_events.columns:
            negative_balances = inventory_events[inventory_events['new_balance'] < 0]
            if not negative_balances.empty:
                issues.append(f"Negative inventory balances found: {len(negative_balances)} events")
        
        # Validate inventory depletion matches sales
        if 'sales.csv' in datasets:
            sales_df = datasets['sales.csv']
            
            # Sum sales quantities by merchant-SKU
            sales_summary = sales_df.groupby(['merchant_id', 'sku'])['quantity'].sum().reset_index()
            
            # Sum inventory depletion events
            sale_events = inventory_events[inventory_events['event_type'] == 'sale']
            if not sale_events.empty:
                inventory_depletion = sale_events.groupby(['merchant_id', 'sku'])['quantity'].sum().abs().reset_index()
                
                # Compare (allowing some tolerance for timing differences)
                merged = sales_summary.merge(inventory_depletion, on=['merchant_id', 'sku'], how='outer', suffixes=('_sales', '_inventory'))
                merged = merged.fillna(0)
                
                mismatches = merged[abs(merged['quantity_sales'] - merged['quantity_inventory']) > 1]
                if not mismatches.empty:
                    issues.append(f"Sales-inventory depletion mismatches: {len(mismatches)} merchant-SKU combinations")
        
        return issues
    
    def validate_customer_behavior(self, datasets: Dict[str, pd.DataFrame]) -> List[str]:
        """Validate customer behavior patterns"""
        issues = []
        
        # Conversion rate validation
        if 'views.csv' in datasets and 'sales.csv' in datasets:
            views_df = datasets['views.csv']
            sales_df = datasets['sales.csv']
            
            customer_views = views_df[views_df['customer_id'].notna()]
            total_customer_views = len(customer_views)
            total_sales = len(sales_df)
            
            if total_customer_views > 0:
                conversion_rate = total_sales / total_customer_views
                
                min_rate = VALIDATION_THRESHOLDS['min_conversion_rate']
                max_rate = VALIDATION_THRESHOLDS['max_conversion_rate']
                
                if conversion_rate < min_rate:
                    issues.append(f"Conversion rate too low: {conversion_rate:.4f} < {min_rate}")
                elif conversion_rate > max_rate:
                    issues.append(f"Conversion rate too high: {conversion_rate:.4f} > {max_rate}")
        
        # Cart abandonment validation
        if 'cart_events.csv' in datasets:
            cart_events = datasets['cart_events.csv']
            
            if not cart_events.empty:
                total_carts = cart_events[cart_events['event_type'] == 'add_to_cart']['cart_id'].nunique()
                abandoned_carts = cart_events[cart_events['event_type'] == 'cart_abandoned']['cart_id'].nunique()
                
                if total_carts > 0:
                    abandonment_rate = abandoned_carts / total_carts
                    
                    min_rate = VALIDATION_THRESHOLDS['min_cart_abandonment']
                    max_rate = VALIDATION_THRESHOLDS['max_cart_abandonment']
                    
                    if abandonment_rate < min_rate:
                        issues.append(f"Cart abandonment rate too low: {abandonment_rate:.4f} < {min_rate}")
                    elif abandonment_rate > max_rate:
                        issues.append(f"Cart abandonment rate too high: {abandonment_rate:.4f} > {max_rate}")
        
        return issues
    
    def validate_statistical_distributions(self, datasets: Dict[str, pd.DataFrame]) -> List[str]:
        """Validate statistical distributions look reasonable"""
        issues = []
        
        # Age distribution
        if 'customers.csv' in datasets:
            customers_df = datasets['customers.csv']
            if 'age' in customers_df.columns:
                age_mean = customers_df['age'].mean()
                if age_mean < 25 or age_mean > 50:
                    issues.append(f"Unusual age distribution mean: {age_mean:.1f}")
        
        # Price distribution
        if 'products.csv' in datasets:
            products_df = datasets['products.csv']
            if 'base_price' in products_df.columns:
                price_median = products_df['base_price'].median()
                # Adjusted thresholds for ₹100-₹100,000 range
                if price_median < 1000 or price_median > 80000:
                    issues.append(f"Unusual price distribution median: ₹{price_median:.0f}")
        
        # Temporal distribution
        if 'views.csv' in datasets:
            views_df = datasets['views.csv']
            if 'timestamp' in views_df.columns:
                views_per_day = views_df.groupby(views_df['timestamp'].dt.date).size()
                cv = views_per_day.std() / views_per_day.mean()  # Coefficient of variation
                
                if cv > 2.0:  # High variability might indicate issues
                    issues.append(f"High variability in daily views: CV = {cv:.2f}")
        
        return issues
    
    def _generate_dataset_summary(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate summary statistics for the dataset"""
        summary = {}
        
        for name, df in datasets.items():
            summary[name] = {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'null_values': df.isnull().sum().sum(),
                'duplicate_rows': df.duplicated().sum()
            }
        
        # Cross-table metrics
        if 'customers.csv' in datasets and 'sales.csv' in datasets:
            total_customers = len(datasets['customers.csv'])
            customers_with_purchases = datasets['sales.csv']['customer_id'].nunique()
            summary['customer_purchase_rate'] = customers_with_purchases / total_customers if total_customers > 0 else 0
        
        if 'views.csv' in datasets and 'sales.csv' in datasets:
            total_views = len(datasets['views.csv'])
            total_sales = len(datasets['sales.csv'])
            summary['overall_conversion_rate'] = total_sales / total_views if total_views > 0 else 0
        
        return summary
    
    def _calculate_validation_metrics(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate key validation metrics"""
        metrics = {}
        
        # Data completeness
        total_cells = sum(df.shape[0] * df.shape[1] for df in datasets.values())
        null_cells = sum(df.isnull().sum().sum() for df in datasets.values())
        metrics['data_completeness'] = 1 - (null_cells / total_cells) if total_cells > 0 else 0
        
        # Referential integrity score
        ref_issues = len([issue for issue in self.issues_found if 'reference' in issue.lower()])
        total_refs = 100  # Approximate number of potential reference checks
        metrics['referential_integrity'] = max(0, 1 - (ref_issues / total_refs))
        
        # Business logic compliance
        business_issues = len([issue for issue in self.issues_found if any(word in issue.lower() 
                                                                          for word in ['price', 'stock', 'inventory'])])
        metrics['business_logic_compliance'] = max(0, 1 - (business_issues / 10))
        
        # Overall quality score
        total_issues = len(self.issues_found)
        metrics['overall_quality_score'] = max(0, 1 - (total_issues / 50))  # Normalize to 0-1
        
        return metrics
    
    def generate_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate a human-readable validation report"""
        report = []
        report.append("=" * 60)
        report.append("DATA VALIDATION REPORT")
        report.append("=" * 60)
        
        # Overall status
        status = "✓ PASSED" if validation_results['is_valid'] else "✗ FAILED"
        report.append(f"Overall Status: {status}")
        report.append(f"Total Issues Found: {validation_results['total_issues']}")
        report.append("")
        
        # Validation metrics
        report.append("Validation Metrics:")
        for metric, value in validation_results['validation_metrics'].items():
            percentage = value * 100
            report.append(f"  {metric}: {percentage:.1f}%")
        report.append("")
        
        # Dataset summary
        report.append("Dataset Summary:")
        for table, stats in validation_results['dataset_summary'].items():
            if isinstance(stats, dict):
                report.append(f"  {table}:")
                report.append(f"    Rows: {stats['rows']:,}")
                report.append(f"    Columns: {stats['columns']}")
                report.append(f"    Memory: {stats['memory_mb']:.1f} MB")
                if stats['null_values'] > 0:
                    report.append(f"    Null Values: {stats['null_values']:,}")
        report.append("")
        
        # Issues found
        if validation_results['issues']:
            report.append("Issues Found:")
            for i, issue in enumerate(validation_results['issues'][:20], 1):  # Limit to first 20
                report.append(f"  {i}. {issue}")
            
            if len(validation_results['issues']) > 20:
                remaining = len(validation_results['issues']) - 20
                report.append(f"  ... and {remaining} more issues")
        else:
            report.append("✓ No issues found!")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)