from __future__ import annotations

from itertools import product

import pandas as pd


TEMPERATURE_SHIFTS_C = [
    0.0,
    1.0,
    2.0,
]

RAINFALL_SHIFTS_PCT = [
    0.0,
    -10.0,
    -20.0,
    -30.0,
]


def scenario_type(
    temperature_shift_c: float,
    rainfall_shift_pct: float,
) -> str:

    heat = temperature_shift_c > 0
    drought = rainfall_shift_pct < 0

    if not heat and not drought:
        return "Baseline"

    if heat and drought:
        return "Compound"

    if heat:
        return "Heat"

    return "Drought"


def scenario_name(
    temperature_shift_c: float,
    rainfall_shift_pct: float,
) -> str:

    temp = (
        f"T+{temperature_shift_c:g}"
    )

    rain = (
        "R0"
        if rainfall_shift_pct == 0
        else f"R{rainfall_shift_pct:g}"
    )

    return f"{temp}_{rain}"


def build_scenarios() -> pd.DataFrame:

    rows = []

    for temp, rain in product(
        TEMPERATURE_SHIFTS_C,
        RAINFALL_SHIFTS_PCT,
    ):

        rows.append(
            {
                "Scenario":
                    scenario_name(
                        temp,
                        rain,
                    ),
                "Scenario_Type":
                    scenario_type(
                        temp,
                        rain,
                    ),
                "Temperature_Shift_C":
                    temp,
                "Rainfall_Shift_pct":
                    rain,
            }
        )

    return pd.DataFrame(rows)