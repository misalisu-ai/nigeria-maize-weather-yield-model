from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


WEATHER_COLS = [
    "PRECTOTCORR",
    "T2M_MAX",
    "T2M_MIN",
    "RH2M",
    "ALLSKY_SFC_SW_DWN",
]


SEASON_MONTHS = {
    "North": list(range(5, 11)),
    "Middle": list(range(4, 11)),
    "South": list(range(3, 12)),
}


NORTH = {
    "Adamawa",
    "Bauchi",
    "Borno",
    "Gombe",
    "Jigawa",
    "Kaduna",
    "Kano",
    "Katsina",
    "Kebbi",
    "Sokoto",
    "Taraba",
    "Yobe",
    "Zamfara",
}


SOUTH = {
    "Abia",
    "Akwa Ibom",
    "Anambra",
    "Bayelsa",
    "Cross River",
    "Delta",
    "Ebonyi",
    "Edo",
    "Ekiti",
    "Enugu",
    "Imo",
    "Lagos",
    "Ogun",
    "Ondo",
    "Osun",
    "Oyo",
    "Rivers",
}


def season_group(state: str) -> str:
    """
    Exact state grouping used to construct Dataset v1.0.
    """

    if state in NORTH:
        return "North"

    if state in SOUTH:
        return "South"

    return "Middle"


def max_consecutive_dry_days(
    rain: pd.Series,
    threshold_mm: float = 1.0,
) -> int:
    """
    Exact CDD definition used in Dataset v1.0.
    """

    dry = (
        rain
        .fillna(0)
        .lt(threshold_mm)
    )

    groups = (
        dry
        .ne(dry.shift())
        .cumsum()
    )

    lengths = dry.groupby(
        groups
    ).sum()

    return (
        int(lengths.max())
        if len(lengths)
        else 0
    )


def build_raw_file_map(
    raw_dir: str | Path,
) -> dict[str, Path]:
    """
    Map each state to exactly one raw NASA POWER file.

    Uses metadata stored inside each CSV rather than relying
    entirely on filenames.
    """

    raw_dir = Path(raw_dir)

    files = sorted(
        raw_dir.glob("*.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No NASA POWER CSV files found in {raw_dir}"
        )

    mapping = {}

    for file in files:

        header = pd.read_csv(
            file,
            nrows=1,
        )

        if "State" not in header.columns:
            raise ValueError(
                f"{file.name} does not contain a State column."
            )

        state = str(
            header.iloc[0]["State"]
        ).strip()

        if state in mapping:
            raise RuntimeError(
                f"More than one raw file found for state: {state}"
            )

        mapping[state] = file

    return mapping


def load_state_daily(
    state: str,
    file_map: dict[str, Path],
) -> pd.DataFrame:

    if state not in file_map:
        raise FileNotFoundError(
            f"No raw weather file mapped to {state}"
        )

    df = pd.read_csv(
        file_map[state],
        parse_dates=["Date"],
    )

    for col in WEATHER_COLS:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    return df


def perturb_daily_weather(
    daily: pd.DataFrame,
    *,
    temperature_shift_c: float,
    rainfall_shift_pct: float,
) -> pd.DataFrame:
    """
    Apply controlled counterfactual changes to daily weather.

    RH and solar radiation remain unchanged.
    """

    stressed = daily.copy()

    rainfall_factor = (
        1.0
        + rainfall_shift_pct / 100.0
    )

    if rainfall_factor < 0:
        raise ValueError(
            "Rainfall perturbation would produce "
            "a negative multiplier."
        )

    stressed["PRECTOTCORR"] = (
        stressed["PRECTOTCORR"]
        * rainfall_factor
    )

    stressed["T2M_MAX"] = (
        stressed["T2M_MAX"]
        + temperature_shift_c
    )

    stressed["T2M_MIN"] = (
        stressed["T2M_MIN"]
        + temperature_shift_c
    )

    return stressed


def recompute_state_year_features(
    *,
    state: str,
    year: int,
    daily: pd.DataFrame,
    rainfall_climatology: pd.DataFrame,
    t_base: float = 10.0,
    dry_threshold_mm: float = 1.0,
) -> dict:
    """
    Recompute the exact seasonal weather features used in
    Dataset v1.0 from a daily weather series.
    """

    season = season_group(
        state
    )

    months = SEASON_MONTHS[
        season
    ]

    year_daily = daily[
        daily["Date"].dt.year
        == year
    ].copy()

    g = year_daily[
        year_daily[
            "Date"
        ].dt.month.isin(
            months
        )
    ].copy()

    if g.empty:
        raise ValueError(
            f"No seasonal weather data for {state}-{year}"
        )

    # ---------------------------------------------------------
    # GDD — exact Dataset v1.0 formulation
    # ---------------------------------------------------------

    daily_gdd = (
        (
            (
                g["T2M_MAX"]
                + g["T2M_MIN"]
            )
            / 2.0
        )
        - t_base
    ).clip(
        lower=0
    )

    seasonal_gdd = (
        daily_gdd.sum(
            min_count=1
        )
    )

    # ---------------------------------------------------------
    # Rainfall
    # ---------------------------------------------------------

    rainfall = (
        g[
            "PRECTOTCORR"
        ].sum(
            min_count=1
        )
    )

    # ---------------------------------------------------------
    # Fixed repaired 2000–2019 climatology
    # ---------------------------------------------------------

    clim = rainfall_climatology[
        rainfall_climatology[
            "State"
        ] == state
    ]

    if len(clim) != 1:
        raise RuntimeError(
            f"Expected one rainfall climatology row for "
            f"{state}, found {len(clim)}"
        )

    baseline_mean = float(
        clim.iloc[0][
            "Baseline_Mean_mm"
        ]
    )

    baseline_std = float(
        clim.iloc[0][
            "Baseline_SD_mm"
        ]
    )

    if baseline_std <= 0:
        raise ValueError(
            f"Non-positive rainfall baseline SD for {state}"
        )

    anomaly = (
        rainfall
        - baseline_mean
    ) / baseline_std

    # ---------------------------------------------------------
    # Return frozen-schema weather features
    # ---------------------------------------------------------

    return {
        "State":
            state,

        "Year":
            year,

        "Season_Group":
            season,

        "Seasonal_Rainfall_mm":
            float(rainfall),

        "Seasonal_GDD_C":
            float(seasonal_gdd),

        "Max_CDD_days":
            max_consecutive_dry_days(
                g["PRECTOTCORR"],
                threshold_mm=
                    dry_threshold_mm,
            ),

        "Mean_Tmax_C":
            float(
                g["T2M_MAX"].mean()
            ),

        "Mean_Tmin_C":
            float(
                g["T2M_MIN"].mean()
            ),

        "Mean_RH_pct":
            float(
                g["RH2M"].mean()
            ),

        "Mean_Solar_Radiation_MJ_m2_day":
            float(
                g[
                    "ALLSKY_SFC_SW_DWN"
                ].mean()
            ),

        "Rainfall_Anomaly_Z_2000_2019":
            float(anomaly),

        "Weather_Valid_Days":
            int(
                g[
                    WEATHER_COLS
                ]
                .notna()
                .all(
                    axis=1
                )
                .sum()
            ),

        "Weather_Total_Days":
            int(
                len(g)
            ),
    }