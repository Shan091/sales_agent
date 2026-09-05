# file: src/logic/pricing.py
import logging
import re
from typing import Dict, Optional, List
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from pydantic import BaseModel, Field

from src.storage.models import ProductPricing # Assuming this is your SQLModel class
logger = logging.getLogger(__name__)

class AmbiguousProductError(Exception):
    """Raised when a product query matches multiple variants (e.g. 1-gang, 2-gang)."""
    pass

class ProductNotFoundError(Exception):
    """Raised when a product doesn't exist in the catalog."""
    pass


def _normalize_name(name: str) -> str:
    """
    Loosely normalize a product name to alphanumerics-only, casefolded, so tolerant
    matching can bridge the LLM's phrasing and the catalogue's exact string: '6 SW',
    '6sw', '6-SW' and 'Smart Door Lock (Premium)' vs 'Smart Door Lock Premium' all
    collapse to the same key. Used ONLY as a fallback after an exact match misses, and
    only when it resolves to exactly one row (ambiguity fails closed) — so it can never
    silently pick the wrong price.
    """
    return re.sub(r"[^a-z0-9]", "", (name or "").casefold())


# Strict schema to prevent LLM hallucination and ensure predictable tool outputs
class PriceResult(BaseModel):
    product_name: str = Field(..., description="The unique name or SKU of the Otohom product.")
    base_price: float = Field(..., description="The verified retail base price.")
    installation_fee: float = Field(default=0.0, description="Standard installation fee for this product.")
    total_estimated_price: float = Field(..., description="Calculated sum of base_price and installation_fee.")
    currency: str = Field(default="INR", description="Currency code, defaults to INR.")
    is_available: bool = Field(default=True, description="Inventory availability flag.")

class PricingEngine:
    """
    Asynchronous pricing service utilizing asyncpg to query PostgreSQL.
    Enforces strict Pydantic return schemas to prevent downstream LLM hallucinations.
    """
    def __init__(self, session: AsyncSession):
        self.session = session

    @staticmethod
    def _to_result(row: "ProductPricing") -> PriceResult:
        return PriceResult(
            product_name=row.product_name,
            base_price=row.base_price,
            installation_fee=row.installation_fee,
            total_estimated_price=row.base_price + row.installation_fee,
            currency=row.currency,
            is_available=True,
        )

    async def get_product_price(self, product_name: str, region_code: str = "IN-KL") -> Optional[PriceResult]:
        """Fetches single product pricing."""
        statement = select(ProductPricing).where(
            ProductPricing.product_name.ilike(f"%{product_name}%"),
            ProductPricing.region_code == region_code,
            ProductPricing.is_active == True
        )
        results = (await self.session.execute(statement)).scalars().all()

        if not results:
            logger.warning(f"Pricing lookup failed for '{product_name}' in region '{region_code}'")
            raise ProductNotFoundError(f"Product '{product_name}' not found.")

        if len(results) > 1:
            logger.warning(f"Pricing lookup ambiguous for '{product_name}': {len(results)} matches.")
            raise AmbiguousProductError(f"Multiple matches found for '{product_name}'.")

        return PricingEngine._to_result(results[0])

    async def get_product_prices_batch(self, product_names: List[str], region_code: str = "IN-KL") -> Dict[str, Optional[PriceResult]]:
        """
        Resolve many product names to prices in one pass. Returns a map from EACH REQUESTED
        name to its PriceResult (whose `product_name` is the canonical catalogue string) or
        None — fail-closed, so a name that can't be resolved never becomes a wrong charge.

        Two-stage matching:
          1. Exact `.in_()` — the fast, unambiguous path (unchanged behaviour for hits).
          2. Normalized fallback (_normalize_name) for anything the exact pass missed, so the
             LLM's phrasing ('6sw', 'Smart Door Lock (Premium)') still lands on the catalogue
             row. A normalized name that matches more than one active row fails closed to None
             rather than guessing.

        Keying by the requested name (not the canonical one) keeps the caller's downstream
        lookups — discounts.price_line_items(raw_items, trusted) — correct even when the two
        strings differ.
        """
        if not product_names:
            return {}

        # De-dupe while preserving the caller's exact strings as keys.
        requested = list(dict.fromkeys(product_names))

        # 1. Exact pass.
        exact_rows = (await self.session.execute(
            select(ProductPricing).where(
                ProductPricing.product_name.in_(requested),
                ProductPricing.region_code == region_code,
                ProductPricing.is_active == True,
            )
        )).scalars().all()
        exact = {r.product_name: r for r in exact_rows}

        pricing_map: Dict[str, Optional[PriceResult]] = {}
        unresolved: List[str] = []
        for name in requested:
            row = exact.get(name)
            if row is not None:
                pricing_map[name] = PricingEngine._to_result(row)
            else:
                unresolved.append(name)

        # 2. Normalized fallback — only touch the DB again if the exact pass left gaps.
        if unresolved:
            all_rows = (await self.session.execute(
                select(ProductPricing).where(
                    ProductPricing.region_code == region_code,
                    ProductPricing.is_active == True,
                )
            )).scalars().all()
            norm_index: Dict[str, List["ProductPricing"]] = {}
            for r in all_rows:
                norm_index.setdefault(_normalize_name(r.product_name), []).append(r)

            for name in unresolved:
                req_norm = _normalize_name(name)
                matches = norm_index.get(req_norm, [])

                # 3. Substring fallback — if the LLM added descriptors (e.g., 'Grande 6 SW - Golden frame'),
                # check if a catalogue SKU is a substring of the requested string.
                if not matches:
                    substring_matches = []
                    for db_norm, db_rows in norm_index.items():
                        if db_norm in req_norm:
                            substring_matches.append((db_norm, db_rows))
                    
                    if substring_matches:
                        # Sort by length descending so '6swdimmer' wins over '6sw'
                        substring_matches.sort(key=lambda x: len(x[0]), reverse=True)
                        longest_db_norm = substring_matches[0][0]
                        matches = next(rows for n, rows in substring_matches if n == longest_db_norm)

                if len(matches) == 1:
                    logger.info(f"Batch pricing: '{name}' resolved to '{matches[0].product_name}' via normalized match.")
                    pricing_map[name] = PricingEngine._to_result(matches[0])
                else:
                    if len(matches) > 1:
                        logger.warning(
                            f"Batch pricing: '{name}' normalized-matched {len(matches)} rows "
                            f"({[m.product_name for m in matches]}); failing closed to None."
                        )
                    else:
                        logger.warning(f"Batch pricing lookup missed SKU: '{name}'")
                    pricing_map[name] = None

        return pricing_map

    async def list_catalogue_names(self, region_code: str = "IN-KL") -> List[str]:
        """
        Every active, priceable product name for this region, alphabetically.

        Injected into the sales prompt so the model chooses `checkout_items[].sku` from a
        closed set it can actually see, the same principle the offer registry already uses.
        Without it the model is asked for "the exact catalogue name" while never having been
        shown one, and invents plausible-but-unresolvable strings ('GRANDE_6GANG_PANEL') that
        fail-close to no quote at all. A product missing from this list cannot be priced, so
        it must not be offered for sale either.
        """
        rows = (await self.session.execute(
            select(ProductPricing.product_name).where(
                ProductPricing.region_code == region_code,
                ProductPricing.is_active == True,
            )
        )).scalars().all()
        return sorted({name for name in rows if name})
