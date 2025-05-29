# ecommerce_mock_data_generator.py
"""
Generate a connected mock dataset for an Indian e‑commerce marketplace, **now with realistic Out‑of‑Stock (OOS) events** that vary by product category (never exceeding 20 percent) and granular OOS reasons.

Tables generated (each as a CSV by default):
    1. customers.csv              – Customer master
    2. products.csv               – SKU master with category hierarchy
    3. product_descriptions.csv   – Attribute/value pairs for every SKU
    4. offers.csv                 – Merchant‑SKU level offer details
    5. views.csv                  – Product view events (HTTP request granularity) incl. in‑stock flag
    6. instock_events.csv         – Instock / detailed OOS reason for every view
    7. sales.csv                  – Customer transaction facts
    8. ratings.csv                – Product ratings / reviews

Key updates
-----------
* **Category‑specific OOS rates** – configurable per category (all < 20%).
* **Granular OOS reasons** mapped to four business areas plus a catch‑all.

Usage (CLI)
-----------
```bash
python ecommerce_mock_data_generator.py \
    --outdir ./mock_data \
    --num-customers 10000 \
    --num-products 5000 \
    --num-merchants 30 \
    --start 2024-05-19 \
    --end   2025-05-18
```

Dependencies
------------
```
pip install pandas numpy faker python-dateutil tqdm
```
"""

import argparse
import random
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from dateutil.rrule import rrule, DAILY
from faker import Faker
from tqdm import tqdm

fake = Faker("en_IN")

# ------------------------------ CONFIG DEFAULTS ----------------------------- #
DEFAULT_NUM_CUSTOMERS = 5_000
DEFAULT_NUM_PRODUCTS = 2_000
DEFAULT_NUM_MERCHANTS = 20
DEFAULT_START = date(2024, 5, 19)
DEFAULT_END = date(2025, 5, 18)

# Promotional / seasonal multipliers (lower‑case keys are month numbers)
PROMO_WINDOWS: Dict[tuple, float] = {
    # Republic Day Sale (week centred on 26 Jan)
    (date(1900, 1, 23).timetuple().tm_yday, date(1900, 1, 29).timetuple().tm_yday): 3.0,
    # Holi (approx 18 Mar, ±2 days)
    (date(1900, 3, 16).timetuple().tm_yday, date(1900, 3, 20).timetuple().tm_yday): 2.0,
    # Independence Day Sale
    (date(1900, 8, 13).timetuple().tm_yday, date(1900, 8, 17).timetuple().tm_yday): 2.5,
    # Diwali (around early Nov)
    (date(1900, 10, 28).timetuple().tm_yday, date(1900, 11, 3).timetuple().tm_yday): 4.0,
    # Christmas week
    (date(1900, 12, 23).timetuple().tm_yday, date(1900, 12, 29).timetuple().tm_yday): 1.8,
}

# Maximum 20 % OOS per category – tune as required
OOS_RATE_BY_CATEGORY = {
    "Electronics": 0.10,
    "Home":        0.15,
    "Fashion":     0.12,
    "Beauty":      0.18,
    "Sports":      0.14,
}

# OOS reason hierarchy – flattened into a single list for random selection
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

FULFILMENT_STATUSES = ["Shipped", "Delivered", "Returned", "Cancelled"]
CATEGORIES = {
    "Electronics": ["Mobiles", "Laptops", "Headphones", "Wearables"],
    "Home": ["Kitchen", "Furniture", "Decor", "Tools"],
    "Fashion": ["Men Topwear", "Women Topwear", "Footwear", "Accessories"],
    "Beauty": ["Skincare", "Haircare", "Fragrances"],
    "Sports": ["Fitness", "Outdoor", "Team Sports"],
}

# For category‑specific dynamic attributes
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

# ------------------------------ HELPER FUNCTIONS ---------------------------- #

def daterange(start: date, end: date) -> List[date]:
    return [dt.date() for dt in rrule(DAILY, dtstart=start, until=end)]


def promo_multiplier(d: date) -> float:
    """Return traffic multiplier for a given calendar date."""
    doy = date(1900, d.month, d.day).timetuple().tm_yday
    for (start_doy, end_doy), mult in PROMO_WINDOWS.items():
        if start_doy <= doy <= end_doy:
            return mult
    # Slight weekly seasonality – weekends busier
    return 1.4 if d.weekday() >= 5 else 1.0


def weighted_date_choice(start: date, end: date, size: int) -> np.ndarray:
    dates = daterange(start, end)
    weights = np.array([promo_multiplier(d) for d in dates], dtype=float)
    weights /= weights.sum()
    return np.random.choice(dates, p=weights, size=size)


def make_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

# ------------------------------ GENERATORS ---------------------------------- #

def gen_customers(n: int) -> pd.DataFrame:
    records = []
    for cid in range(1, n + 1):
        name = fake.name()
        email = fake.email()
        city = fake.city()
        age = random.randint(18, 65)
        gender = random.choice(["M", "F", "O"])
        records.append((cid, name, email, city, age, gender))
    return pd.DataFrame(records, columns=["customer_id", "name", "email", "city", "age", "gender"])


def gen_products(n: int) -> pd.DataFrame:
    records = []
    for pid in range(1, n + 1):
        cat = random.choice(list(CATEGORIES.keys()))
        subcat = random.choice(CATEGORIES[cat])
        sku = f"SKU{pid:06d}"
        base_price = round(random.uniform(100, 100_000), 2)
        records.append((sku, pid, cat, subcat, base_price))
    return pd.DataFrame(records, columns=["sku", "product_id", "category", "subcategory", "base_price"])


def gen_product_descriptions(products: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in products.iterrows():
        attrs = CATEGORY_ATTRIBUTES.get(row.subcategory, ["Brand"])
        for attr in attrs:
            value = fake.word().title() if "Brand" in attr else fake.word().title()
            rows.append((row.sku, attr, value))
    return pd.DataFrame(rows, columns=["sku", "description", "value"])


def gen_offers(products: pd.DataFrame, num_merchants: int) -> pd.DataFrame:
    merchants = [f"MER{m:04d}" for m in range(1, num_merchants + 1)]
    rows = []
    for _, prod in products.iterrows():
        # Each SKU available with 1‑4 merchants
        for m in random.sample(merchants, k=random.randint(1, 4)):
            inventory = random.randint(0, 1_000)
            price = round(prod.base_price * random.uniform(0.7, 1.2), 2)
            listing_date = fake.date_between(start_date="-2y", end_date="today")
            rows.append((m, prod.sku, inventory, price, listing_date))
    return pd.DataFrame(rows, columns=["merchant_id", "sku", "inventory", "offer_price", "listing_date"])


def gen_views(products: pd.DataFrame, customers: pd.DataFrame, offers: pd.DataFrame,
              start: date, end: date, avg_views_per_day: int = 5_000) -> pd.DataFrame:
    # Pre‑compute helpers
    date_list = daterange(start, end)
    total_views = sum(int(avg_views_per_day * promo_multiplier(d)) for d in date_list)

    sku_to_offers = offers.groupby("sku")["offer_price"].apply(list).to_dict()
    sku_to_cat = products.set_index("sku")["category"].to_dict()

    rows = []
    for view_id in tqdm(range(1, total_views + 1), desc="Generating views"):
        view_date = weighted_date_choice(start, end, 1)[0]
        timestamp = datetime.combine(view_date, datetime.min.time()) + timedelta(seconds=random.randint(0, 86_399))

        sku = random.choice(products.sku.values)
        category = sku_to_cat[sku]
        oos_rate = OOS_RATE_BY_CATEGORY.get(category, 0.1)
        instock = random.random() >= oos_rate  # True if random value >= rate

        offer_prices = sku_to_offers.get(sku, [])
        price_viewed = random.choice(offer_prices) if offer_prices else np.nan
        customer_id = random.choice(customers.customer_id.values) if random.random() < 0.8 else np.nan

        rows.append((view_id, timestamp, sku, len(offer_prices), instock, price_viewed, customer_id))

    cols = [
        "http_request_id", "timestamp", "sku", "num_offers",
        "instock", "price_viewed", "customer_id"
    ]
    return pd.DataFrame(rows, columns=cols)


def gen_instock_events(views: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, v in views.iterrows():
        if v.instock:
            reason = "Instock"
        else:
            reason = random.choice(FLAT_OOS_REASONS)
        rows.append((v.http_request_id, v.sku, reason))
    return pd.DataFrame(rows, columns=["http_request_id", "sku", "reason"])


def gen_sales(views: pd.DataFrame, offers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    order_id = 1
    for _, v in views.iterrows():
        if np.isnan(v.customer_id):
            continue
        # Base conversion
        base_prob = 0.02
        # Promotional uplift
        promo_mult = 1.8 if promo_multiplier(v.timestamp.date()) > 1.0 else 1.0
        prob = base_prob * promo_mult
        if random.random() < prob:
            merchant_ids = offers.loc[offers.sku == v.sku, "merchant_id"].tolist()
            if not merchant_ids:
                continue
            merchant_id = random.choice(merchant_ids)
            qty = random.randint(1, 3)
            status = random.choices(FULFILMENT_STATUSES, weights=[0.1, 0.7, 0.1, 0.1])[0]
            rows.append((order_id, v.customer_id, merchant_id, v.sku, qty, v.price_viewed, v.timestamp, status))
            order_id += 1
    cols = ["order_id", "customer_id", "merchant_id", "sku", "quantity", "unit_price", "order_timestamp", "fulfilment_status"]
    return pd.DataFrame(rows, columns=cols)


def gen_ratings(sales: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, s in sales.iterrows():
        if random.random() < 0.4:  # 40 % of orders leave a rating
            rating = random.randint(1, 5)
            review = fake.sentence(nb_words=12)
            review_date = s.order_timestamp + timedelta(days=random.randint(1, 30))
            rows.append((s.order_id, s.customer_id, s.sku, rating, review, review_date))
    return pd.DataFrame(rows, columns=["rating_id", "customer_id", "sku", "rating", "review", "rating_timestamp"])

# ------------------------------ MAIN PIPELINE ------------------------------- #

def main(
    outdir: Path,
    num_customers: int,
    num_products: int,
    num_merchants: int,
    start: date,
    end: date,
):
    make_dir(outdir)

    customers = gen_customers(num_customers)
    products = gen_products(num_products)
    product_desc = gen_product_descriptions(products)
    offers = gen_offers(products, num_merchants)
    views = gen_views(products, customers, offers, start, end)
    instock_ev = gen_instock_events(views)
    sales = gen_sales(views, offers)
    ratings = gen_ratings(sales)

    dfs = {
        "customers.csv": customers,
        "products.csv": products,
        "product_descriptions.csv": product_desc,
        "offers.csv": offers,
        "views.csv": views,
        "instock_events.csv": instock_ev,
        "sales.csv": sales,
        "ratings.csv": ratings,
    }

    for fname, df in dfs.items():
        df.to_csv(outdir / fname, index=False)

    print("\nGenerated row counts:")
    for fname, df in dfs.items():
        print(f"{fname:25s} {len(df):>10,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mock ecommerce database CSVs")
    parser.add_argument("--outdir", type=Path, default=Path("./mock_data"))
    parser.add_argument("--num-customers", type=int, default=DEFAULT_NUM_CUSTOMERS)
    parser.add_argument("--num-products", type=int, default=DEFAULT_NUM_PRODUCTS)
    parser.add_argument("--num-merchants", type=int, default=DEFAULT_NUM_MERCHANTS)
    parser.add_argument("--start", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(), default=DEFAULT_START)
    parser.add_argument("--end", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(), default=DEFAULT_END)
    args = parser.parse_args()

    main(
        outdir=args.outdir,
        num_customers=args.num_customers,
        num_products=args.num_products,
        num_merchants=args.num_merchants,
        start=args.start,
        end=args.end,
    )
