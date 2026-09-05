# file: src/logic/catalog.py
# dependencies: pydantic>=2.5.0

import asyncio
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from src.logic.pricing import PricingEngine

# 1. Canonical Product Nomenclature
class OtohomProduct(str, Enum):
    SMART_DOOR_LOCK_PREMIUM = "Smart Door Lock (Premium Model)"
    SMART_DOOR_LOCK_BASE = "Smart Door Lock (Base Model)"
    GRANDE_SWITCH_4 = "Grande Series 4 SW"
    GRANDE_SWITCH_6_DIMMER = "Grande Series 6 SW DIMMER"
    TOUCH_PANEL_10 = "10 Inch Smart Home Control Panel"
    VIDEO_DOORBELL = "HD WiFi Video Doorbell"
    CURTAIN_MOTOR = "Curtain Automation Motor"
    ZIGBEE_HUB = "Zigbee Hub"
    PIR_SENSOR = "Zigbee PIR Motion Sensor"

# 2. Base Templates
class PricingTierTemplate(BaseModel):
    name: str = Field(..., description="Package display name.")
    products: List[OtohomProduct] = Field(..., description="Products in this tier.")
    psychological_role: str = Field(..., description="BASE, DECOY, or TARGET.")
    pitch_text: str = Field(..., description="WhatsApp copy for this option.")
    # markup_multiplier completely removed to guarantee strict 1:1 database pricing

class UpsellPlaybookTemplate(BaseModel):
    entry_product: OtohomProduct
    acknowledgment_text: str = Field(..., description="Immediate value delivery.")
    media_url: Optional[str] = Field(None, description="Brochure/Video URL.")
    cross_sell_hook: str = Field(..., description="No-Oriented pivot question.")
    tiers: List[PricingTierTemplate]

# 3. Hydrated Models (Passed to the LLM)
class PricingTier(PricingTierTemplate):
    price_inr: int = Field(..., description="Calculated price in INR. Exact DB sum.")

class UpsellPlaybook(UpsellPlaybookTemplate):
    tiers: List[PricingTier]

# 4. Master Catalog Template Map
CATALOG_MAP: Dict[str, UpsellPlaybookTemplate] = {
    "smart_lock": UpsellPlaybookTemplate(
        entry_product=OtohomProduct.SMART_DOOR_LOCK_PREMIUM,
        acknowledgment_text="The Premium Smart Door Lock includes 5-in-1 access and anti-pry alarm.",
        media_url="https://otohom.com/assets/premium-lock-demo.mp4",
        cross_sell_hook="Would you be opposed to seeing bundled pricing for our HD Video Doorbell?",
        tiers=[
            PricingTierTemplate(
                name="Standalone Security",
                products=[OtohomProduct.SMART_DOOR_LOCK_PREMIUM],
                psychological_role="BASE",
                pitch_text="Premium Door Lock with installation."
            ),
            PricingTierTemplate(
                name="Basic Camera Add-on",
                products=[OtohomProduct.SMART_DOOR_LOCK_PREMIUM, OtohomProduct.PIR_SENSOR],
                psychological_role="DECOY",
                pitch_text="Lock + 1 basic motion sensor."
            ),
            PricingTierTemplate(
                name="Complete Front Door Command",
                products=[
                    OtohomProduct.SMART_DOOR_LOCK_PREMIUM, 
                    OtohomProduct.VIDEO_DOORBELL, 
                    OtohomProduct.ZIGBEE_HUB
                ],
                psychological_role="TARGET",
                pitch_text="Lock + Video Doorbell + Zigbee Hub."
            )
        ]
    )
}

# 5. Async Hydration Logic
async def get_hydrated_playbook(ad_tag: str, pricing_engine: PricingEngine, region_code: str = "IN-KL") -> Optional[UpsellPlaybook]:
    """
    Fetches the template and hydrates prices using batch database lookups.
    Enforces strict guardrails to prevent zero-pricing hallucination risks.
    """
    template = CATALOG_MAP.get(ad_tag)
    if not template:
        return None

    # Batch extract products to avoid sequential DB roundtrips
    required_products = list({product.value for tier in template.tiers for product in tier.products})
    
    # Execute single batch database lookup
    pricing_map = await pricing_engine.get_product_prices_batch(required_products, region_code)

    hydrated_tiers = []
    for tier in template.tiers:
        tier_total_price = 0
        for product in tier.products:
            price_info = pricing_map.get(product.value, {})
            
            # LAYER 9 GUARDRAIL: Hard fail on missing or zero prices
            if price_info.get("error", True) or price_info.get("total_estimated_price", 0) <= 0:
                raise ValueError(f"CRITICAL: Pricing unavailable or zero for {product.value}. Aborting playbook hydration to prevent underpricing.")
                
            tier_total_price += price_info.get("total_estimated_price")
        
        hydrated_tiers.append(
            PricingTier(
                **tier.model_dump(),
                price_inr=tier_total_price
            )
        )

    return UpsellPlaybook(
        **template.model_dump(exclude={"tiers"}),
        tiers=hydrated_tiers
    )