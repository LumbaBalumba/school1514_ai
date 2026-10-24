"""Recreate the small synthetic datasets used in lesson 02."""

from pathlib import Path

import numpy as np
import pandas as pd


def main(data_dir: Path | None = None) -> None:
    data = (
        data_dir if data_dir is not None else Path(__file__).resolve().parent / "data"
    )
    data.mkdir(parents=True, exist_ok=True)

    weather = pd.DataFrame(
        {
            "day": np.arange(1, 15),
            "moscow": [15, 17, 16, 20, 19, 18, 21, 22, 20, 18, 17, 19, 21, 23],
            "kazan": [13, 14, 17, 18, 16, 19, 20, 19, 18, 17, 19, 20, 22, 21],
            "wind_ms": [1, 3, 2, 5, 4, 2, 6, 3, 1, 4, 7, 5, 2, 3],
        }
    )
    weather.to_csv(data / "weather.csv", index=False)

    rng = np.random.default_rng(1514)
    count = 60
    bike = np.resize(["Городской", "Горный", "Шоссейный"], count)
    distance = rng.uniform(2, 25, count).round(1)
    speed = np.select([bike == "Городской", bike == "Горный"], [14, 12], default=20)
    minutes = np.maximum(5, distance / speed * 60 + rng.normal(0, 3, count)).round(1)
    rides = pd.DataFrame(
        {
            "trip_id": np.arange(1, count + 1),
            "bike": bike,
            "distance_km": distance,
            "minutes": minutes,
        }
    )
    rides.loc[[4, 23], "distance_km"] = np.nan
    rides.loc[[9, 30, 51], "minutes"] = np.nan
    rides = pd.concat([rides, rides.iloc[[0]]], ignore_index=True)
    rides.to_csv(data / "rides.csv", index=False)


if __name__ == "__main__":
    main()
