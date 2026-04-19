"""Compute per-customer order totals from a list of orders."""
from collections import defaultdict


def totals_by_customer(orders):
    """Given a list of order dicts, return {customer_id: total_amount}.

    Each order is expected to have keys: customer_id, quantity, unit_price.
    Amount is quantity * unit_price. Missing or non-numeric values count
    as zero; we don't raise.
    """
    totals = defaultdict(float)
    for order in orders:
        cid = order.get("customer_id")
        if cid is None:
            continue
        try:
            qty = int(order.get("quantity", 0) or 0)
            price = float(order.get("unit_price", 0) or 0)
        except (TypeError, ValueError):
            continue
        totals[cid] += qty * price
    return dict(totals)


def top_n_customers(totals, n=5):
    """Return the n customers with the highest totals, as (cid, amount) pairs."""
    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:n]


if __name__ == "__main__":
    import json
    import sys
    orders = json.load(sys.stdin)
    totals = totals_by_customer(orders)
    for cid, amt in top_n_customers(totals):
        print(f"{cid}: ${amt:.2f}")
