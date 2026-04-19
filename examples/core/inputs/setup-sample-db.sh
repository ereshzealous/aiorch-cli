#!/usr/bin/env bash
#
# setup-sample-db.sh — create a sample SQLite database for the
# DB-read example pipelines (33-35).
#
# Run once before running the DB pipelines:
#
#   cd ~/Documents/Dev/aiorch/cli
#   ./inputs/setup-sample-db.sh
#
# Creates ~/Documents/Dev/aiorch/cli/inputs/sample.db with three
# tables: customers, products, orders. Idempotent — safe to re-run.

set -euo pipefail

DB_PATH="$(dirname "$0")/sample.db"

# Wipe and recreate so re-runs always start fresh
rm -f "$DB_PATH"

sqlite3 "$DB_PATH" <<'SQL'
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    region TEXT NOT NULL,
    signup_date TEXT NOT NULL
);

CREATE TABLE products (
    sku TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    sku TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    order_date TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(id),
    FOREIGN KEY (sku) REFERENCES products(sku)
);

-- Customers
INSERT INTO customers (id, name, region, signup_date) VALUES
    (1, 'Alice Chen',   'us-west',    '2025-01-15'),
    (2, 'Bob Smith',    'us-east',    '2025-02-03'),
    (3, 'Carol Davis',  'eu-central', '2025-02-19'),
    (4, 'David Kim',    'us-west',    '2025-03-01'),
    (5, 'Eve Patel',    'eu-central', '2025-03-12'),
    (6, 'Frank Liu',    'apac-south', '2025-03-20'),
    (7, 'Grace Wang',   'us-east',    '2025-04-01'),
    (8, 'Henry Müller', 'eu-central', '2025-04-08');

-- Products
INSERT INTO products (sku, name, category, price) VALUES
    ('WGT-001', 'Widget',     'Tools',       9.99),
    ('GDG-002', 'Gadget',     'Electronics', 49.99),
    ('GZM-003', 'Gizmo',      'Toys',        14.99),
    ('DHK-004', 'Doohickey',  'Electronics', 29.99),
    ('THM-005', 'Thingamajig','Tools',       19.99),
    ('WHJ-006', 'Whatsit',    'Toys',         7.99);

-- Orders (mix of customers, products, dates)
INSERT INTO orders (customer_id, sku, quantity, order_date) VALUES
    (1, 'WGT-001', 3, '2026-04-01'),
    (1, 'GDG-002', 1, '2026-04-01'),
    (2, 'GZM-003', 2, '2026-04-02'),
    (3, 'DHK-004', 1, '2026-04-02'),
    (3, 'WGT-001', 5, '2026-04-03'),
    (4, 'THM-005', 2, '2026-04-03'),
    (4, 'WHJ-006', 4, '2026-04-04'),
    (5, 'GDG-002', 1, '2026-04-04'),
    (5, 'WGT-001', 2, '2026-04-05'),
    (6, 'DHK-004', 3, '2026-04-05'),
    (6, 'GZM-003', 1, '2026-04-06'),
    (7, 'WGT-001', 1, '2026-04-06'),
    (7, 'THM-005', 2, '2026-04-07'),
    (8, 'WHJ-006', 6, '2026-04-07'),
    (8, 'GDG-002', 2, '2026-04-08');
SQL

echo "✓ Created $DB_PATH"
echo
echo "Quick sanity check:"
sqlite3 "$DB_PATH" "SELECT COUNT(*) || ' customers' FROM customers;
                    SELECT COUNT(*) || ' products'  FROM products;
                    SELECT COUNT(*) || ' orders'    FROM orders;"
