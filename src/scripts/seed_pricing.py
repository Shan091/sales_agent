"""
Seed representative TEST-MODE product prices into products_pricing.

Why this exists
---------------
For the Razorpay AI Buildathon the agent owns the whole sale, so it needs a trusted,
code-owned source of truth for every amount. That source is the products_pricing table
read by src/logic/pricing.py::PricingEngine — never the LLM. This script populates it.

These are REPRESENTATIVE prices for a test-mode demo, not official Otohom retail pricing.
No real money moves (Razorpay test keys), and the numbers are round, plausible figures
chosen to make the bounded-discount story legible on camera. The bounded-pricing guarantee
(discounts.py + guardrails.validate_payment_request) does not depend on the exact values —
only on them coming from here rather than the model.

Names match the hand-curated catalogue in docs/catalog/** so a product the agent recommends
from RAG resolves to a price. PricingEngine also does a fail-closed normalized match, so a
small phrasing difference ('6sw' vs '6 SW') still lands on the right row.

Region: prices are per region_code (PricingEngine defaults to 'IN-KL', Kerala, India).

Usage
-----
    python -m src.scripts.seed_pricing                 # idempotent upsert into IN-KL
    python -m src.scripts.seed_pricing --region IN-KL
    python -m src.scripts.seed_pricing --purge         # delete existing rows for the region, then seed
"""
import argparse
import asyncio
import logging
from typing import List, Tuple

from sqlmodel import select, delete

from src.core.database import async_session_maker, engine
from src.storage.models import ProductPricing

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# (product_name, base_price, installation_fee). product_name is the canonical string stored
# in the DB and echoed into the itemized quote; keep it clean and close to the catalogue.
# Installation fee is real labour cost and is NEVER discounted (see discounts.apply_offer).
SEED_PRICES: List[Tuple[str, float, float]] = [
    # ── Grande smart switches (premium glass touch panels) ──
    ("1 SW", 1800.0, 300.0),
    ("2 SW", 2200.0, 300.0),
    ("2 Way 2 SW", 2600.0, 300.0),
    ("3 SW", 2800.0, 350.0),
    ("4 SW", 3200.0, 350.0),
    ("6 SW", 4200.0, 400.0),
    ("6 SW - DIMMER", 4800.0, 400.0),
    ("6 SW - SOCKET", 4600.0, 400.0),
    ("6 SW FAN", 4800.0, 400.0),
    ("8 SW", 5200.0, 450.0),
    ("Grande Socket", 2400.0, 250.0),
    # ── Eco / Hider ──
    ("Hider Retrofit Module", 1200.0, 0.0),
    # ── Security ──
    ("Smart Door Lock Premium", 28000.0, 1500.0),
    ("Smart Door Lock Base", 18000.0, 1500.0),
    ("Video Door Phone", 12000.0, 1200.0),
    ("Biometric Access Control", 22000.0, 1500.0),
    ("Smart Flood Light Camera", 6500.0, 800.0),
    ("Indoor Smart Camera", 3500.0, 500.0),
    ("Smoke Detector", 2800.0, 300.0),
    ("Gas Leak Detector", 3200.0, 300.0),
    # ── Sensors & smart controls ──
    ("PIR Motion Sensor", 1800.0, 200.0),
    ("Microwave Sensor", 2200.0, 200.0),
    ("Energy Meter Single Phase", 3500.0, 400.0),
    ("Energy Meter 3 Phase", 5500.0, 600.0),
    ("Door Window Sensor", 1500.0, 150.0),
    ("IR Blaster", 1800.0, 0.0),
    ("Smart Water Valve Controller", 4500.0, 600.0),
    ("Smart MCB Controller", 3800.0, 500.0),
    ("Water Tank Level Sensor", 3200.0, 500.0),
    # ── Curtain / gate automation ──
    ("Curtain Motor", 6500.0, 800.0),
    ("Gate Automation", 8500.0, 1500.0),
    # ── Hubs & control panels ──
    ("Zigbee Hub", 4500.0, 400.0),
    ("Touch Screen Control Panel 7 inch", 15000.0, 1000.0),
    ("Touch Screen Control Panel 10 inch", 22000.0, 1200.0),
]


async def seed_pricing(region_code: str = "IN-KL", purge: bool = False, currency: str = "INR") -> None:
    inserted = 0
    updated = 0

    async with async_session_maker() as session:
        if purge:
            result = await session.execute(
                delete(ProductPricing).where(ProductPricing.region_code == region_code)
            )
            await session.commit()
            logger.info(f"Purged {result.rowcount or 0} existing pricing row(s) for region {region_code}.")

        for name, base, install in SEED_PRICES:
            existing = (await session.execute(
                select(ProductPricing).where(
                    ProductPricing.product_name == name,
                    ProductPricing.region_code == region_code,
                )
            )).scalars().first()

            if existing is None:
                session.add(ProductPricing(
                    product_name=name,
                    region_code=region_code,
                    base_price=base,
                    installation_fee=install,
                    currency=currency,
                    is_active=True,
                ))
                inserted += 1
            else:
                # Idempotent: re-running refreshes the seed values without duplicating rows.
                existing.base_price = base
                existing.installation_fee = install
                existing.currency = currency
                existing.is_active = True
                updated += 1

        await session.commit()

    await engine.dispose()
    logger.info(
        f"Seeded TEST-MODE pricing for region {region_code}: "
        f"{inserted} inserted, {updated} updated, {len(SEED_PRICES)} total. "
        "(Representative demo prices — not official Otohom retail pricing.)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed representative test-mode product prices.")
    parser.add_argument("--region", default="IN-KL", help="Region code to seed (default: IN-KL).")
    parser.add_argument("--currency", default="INR", help="Currency code (default: INR).")
    parser.add_argument(
        "--purge", action="store_true",
        help="Delete all existing pricing rows for the region before seeding.",
    )
    args = parser.parse_args()
    asyncio.run(seed_pricing(region_code=args.region, purge=args.purge, currency=args.currency))


if __name__ == "__main__":
    main()
