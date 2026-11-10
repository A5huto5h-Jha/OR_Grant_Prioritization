"""
data_generator.py
Synthetic Grant Application Dataset Generator for Socio-Economic Impact Engine (INR ₹).
Generates realistic, offline grant proposal feeds without external APIs.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
import numpy as np
import pandas as pd


SECTORS = [
    "Healthcare",
    "Education",
    "Agriculture & Water",
    "Clean Energy",
    "Rural Infrastructure",
    "Women Empowerment"
]

REGIONS = [
    "Rural/Distressed",
    "Semi-Urban",
    "Urban"
]

SAMPLE_TITLES = {
    "Healthcare": [
        "Maternal & Child Primary Care Center in Tribal Belt",
        "Mobile Diagnostic Vans for Remote Villages",
        "Telemedicine Kiosk Network for Chronic Diseases",
        "Community Mental Health & Counseling Clinics",
        "Emergency Ambulance & Trauma Care Network"
    ],
    "Education": [
        "Digital Smart Classrooms for Government Schools",
        "Vocational IT & STEM Skilling for Underprivileged Youth",
        "Girls Secondary School Retention & Scholarship Program",
        "Community Libraries & Early Childhood Literacy Centers",
        "Teacher Training & Inclusive Learning for Disabled Children"
    ],
    "Agriculture & Water": [
        "Drip Irrigation & Soil Moisture Sensor Kits for Smallholders",
        "Solar-Powered Decentralized Cold Storage for Farmers",
        "Community Rainwater Harvesting & Groundwater Recharge",
        "Organic Millets Processing & Direct Market Linkages",
        "Community Seed Banks & Climate Resilient Farming"
    ],
    "Clean Energy": [
        "Rooftop Solar Electrification for Remote Primary Schools",
        "Decentralized Solar Micro-Grids for Off-Grid Hamlets",
        "Clean Biomass Cookstoves for Rural Households",
        "Solar-Powered Water Pumping & Purification Plants",
        "Waste-to-Bioenergy Digesters for Dairy Co-operatives"
    ],
    "Rural Infrastructure": [
        "All-Weather Footbridges & Feeder Road Connectivity",
        "Village Solid & Liquid Waste Management Facilities",
        "Community Flood Shelters & Early Warning Disaster Systems",
        "Public Sanitation Blocks with Greywater Recycling",
        "Solar Street Lighting for Village Safety & Mobility"
    ],
    "Women Empowerment": [
        "Micro-Enterprise Incubator for Women Self-Help Groups (SHGs)",
        "Financial Literacy, Banking Access & Micro-Credit Network",
        "Artisan Handloom & Handicraft Fair-Trade Collective",
        "Digital Literacy & E-Commerce Training for Rural Women",
        "Community Daycare Centers for Women Agricultural Laborers"
    ]
}


def generate_grant_applications(
    output_path: str = "data/grant_applications.csv",
    num_applications: int = 60,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate realistic synthetic grant proposals in Indian Rupees (₹).

    Columns:
      - grant_id: Unique proposal identifier (GNT-001, GNT-002, etc.)
      - title: Descriptive project title
      - sector: Socio-economic development sector
      - region_tier: Geographic vulnerability classification
      - requested_amount: Funding requested in INR (₹3,00,000 to ₹35,00,000)
      - direct_beneficiaries: Estimated direct individual beneficiaries
      - sroi_ratio: Social Return on Investment ratio ($ / ₹ economic multiplier, 1.3x to 6.2x)
      - vulnerability_index: Socio-economic vulnerability of target group (0.35 to 0.98)
      - execution_feasibility: Historical track record / implementation feasibility (0.60 to 0.95)
    """
    np.random.seed(random_seed)

    records = []
    for i in range(1, num_applications + 1):
        grant_id = f"GNT-{i:03d}"
        
        # Sector distribution: Balanced across developmental areas
        sector = SECTORS[(i - 1) % len(SECTORS)]
        
        # Region distribution: 50% Rural/Distressed, 30% Semi-Urban, 20% Urban
        region_prob = np.random.uniform(0, 1)
        if region_prob < 0.50:
            region = "Rural/Distressed"
        elif region_prob < 0.80:
            region = "Semi-Urban"
        else:
            region = "Urban"

        # Descriptive Title
        titles_pool = SAMPLE_TITLES[sector]
        title_base = titles_pool[(i // len(SECTORS)) % len(titles_pool)]
        title = f"{title_base} - Phase {((i - 1) % 3) + 1}"

        # Requested Budget Amount in INR (Rounded to nearest ₹25,000)
        # Typically between ₹3,50,000 and ₹32,00,000
        base_cost = np.random.uniform(350000.0, 3200000.0)
        requested_amount = round(base_cost / 25000.0) * 25000.0

        # Direct Beneficiaries: Positively correlated with funding amount + noise
        beneficiary_density = np.random.uniform(0.005, 0.025)
        raw_beneficiaries = int(requested_amount * beneficiary_density + np.random.randint(200, 2500))
        direct_beneficiaries = max(350, min(65000, raw_beneficiaries))

        # Social Return on Investment (SROI): 1.4x to 6.0x
        # Higher in rural/distressed areas due to baseline under-development
        sroi_boost = 0.5 if region == "Rural/Distressed" else 0.0
        sroi_ratio = round(float(np.random.uniform(1.4, 5.5) + sroi_boost), 2)

        # Vulnerability Index (0.0 to 1.0): Distressed rural communities get higher scores
        if region == "Rural/Distressed":
            vulnerability = round(float(np.random.uniform(0.70, 0.98)), 3)
        elif region == "Semi-Urban":
            vulnerability = round(float(np.random.uniform(0.48, 0.78)), 3)
        else:
            vulnerability = round(float(np.random.uniform(0.35, 0.65)), 3)

        # Execution Feasibility / NGO Track Record (0.60 to 0.95)
        feasibility = round(float(np.random.uniform(0.65, 0.95)), 3)

        records.append({
            "grant_id": grant_id,
            "title": title,
            "sector": sector,
            "region_tier": region,
            "requested_amount": float(requested_amount),
            "direct_beneficiaries": int(direct_beneficiaries),
            "sroi_ratio": float(sroi_ratio),
            "vulnerability_index": float(vulnerability),
            "execution_feasibility": float(feasibility)
        })

    df = pd.DataFrame(records)

    # Ensure output directory exists and save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[Data Generator] Successfully created {len(df)} grant proposals at '{output_path}'.")
    return df


if __name__ == "__main__":
    generate_grant_applications()
