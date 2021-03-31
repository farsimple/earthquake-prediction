#  ********************************************************************************
#                            ___
#                           /\_ \           CS-C3240: Machine Learning
#     __     __     ___ ___ \//\ \          Student Project
#   /"__`\ /"__`\ /" __` __`\ \ \ \         Predicting Earthquake Magnitudes
#  /\  __//\ \L\ \/\ \/\ \/\ \ \_\ \_
#  \ \____\ \___, \ \_\ \_\ \_\/\____\      Copyright (c) 2021 FarSimple Oy
#   \/____/\/___/\ \/_/\/_/\/_/\/____/      All rights reserved.
#               \ \_\
#                \/_/
#
#  ********************************************************************************

from __future__ import annotations
from ast import literal_eval

from enum import Enum
import logging
import math
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

from autologging import logged, traced
from codetiming import Timer
from haversine import haversine
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from scipy.optimize import curve_fit
from shapely.geometry.linestring import LineString
from sklearn.metrics import r2_score, mean_squared_error


@traced(logging.getLogger(__name__))
@logged(logging.getLogger(__name__))
class DepthCategory(Enum):
    """
    Enumeration for earthquake depth category.
    """
    SHALLOW = 0
    INTERMEDIATE = 1
    DEEP = 2

    @staticmethod
    def from_depth(depth: float) -> DepthCategory:
        """Convert from depth.
        """
        if depth < 70:
            return DepthCategory.SHALLOW
        elif depth < 300:
            return DepthCategory.INTERMEDIATE
        else:
            return DepthCategory.DEEP


@traced(logging.getLogger(__name__))
@logged(logging.getLogger(__name__))
class FaultCategory(Enum):
    """
    Enumeration for fault type.
    """
    UNKNOWN = 0
    DIP_NORMAL = 1
    DIP_REVERSE = 2
    STRIKE = 3
    OBLIQUE = 4
    INACTIVE = 5

    @staticmethod
    def from_gem(name: str) -> FaultCategory:
        """Convert from GEM classification.
        """
        normal_keywords = ["Normal"]
        reverse_keywords = ["Reverse", "Subduction_Thrust", "Blind Thrust"]
        strike_keywords = ["Dextral", "Sinistral", "Strike-Slip", "Sinistral_Transform", "Dextral_Transform"]
        oblique_keywords = ["Dextral-Normal", "Normal-Dextral", "Sinistral-Reverse", "Reverse-Sinistral",
                            "Reverse-Dextral", "Dextral-Reverse", "Sinistral-Normal", "Reverse-Strike-Slip",
                            "Normal-Sinistral", "Normal-Strike-Slip", "Dextral-Oblique"]
        inactive_keywords = ["Anticline", "Syncline", "Spreading_Ridge"]
        if name is None:
            return FaultCategory.UNKNOWN
        elif name.lower() in (value.lower() for value in normal_keywords):
            return FaultCategory.DIP_NORMAL
        elif name.lower() in (value.lower() for value in reverse_keywords):
            return FaultCategory.DIP_REVERSE
        elif name.lower() in (value.lower() for value in strike_keywords):
            return FaultCategory.STRIKE
        elif name.lower() in (value.lower() for value in oblique_keywords):
            return FaultCategory.OBLIQUE
        elif name.lower() in (value.lower() for value in inactive_keywords):
            return FaultCategory.INACTIVE
        else:
            return FaultCategory.UNKNOWN


@traced(logging.getLogger(__name__))
@logged(logging.getLogger(__name__))
class MagnitudeScale(Enum):
    """
    Enumeration for earthquake magnitude scale.
    """
    BODY = 0
    CODA = 1
    DURATION = 2
    ENERGY = 3
    LOCAL = 4
    MOMENT = 5
    SURFACE = 6
    OTHER = 7

    def to_prefix(self) -> Optional[str]:
        """Convert to typical string prefix used for denoting magnitude scales in earthquake catalogs.
        """
        if self is MagnitudeScale.BODY:
            return "Mb"
        elif self is MagnitudeScale.CODA:
            return "Mc"
        elif self is MagnitudeScale.DURATION:
            return "Md"
        elif self is MagnitudeScale.ENERGY:
            return "Me"
        elif self is MagnitudeScale.LOCAL:
            return "Ml"
        elif self is MagnitudeScale.MOMENT:
            return "Mw"
        elif self is MagnitudeScale.SURFACE:
            return "Ms"
        else:
            return None

    @staticmethod
    def from_prefix(value: str) -> MagnitudeScale:
        """Convert from typical string prefix used for denoting magnitude scales in earthquake catalogs.
        """
        prefix = str(value).lower()[:2]
        if prefix == "mb":
            return MagnitudeScale.BODY
        elif prefix == "mc":
            return MagnitudeScale.CODA
        elif prefix == "md":
            return MagnitudeScale.DURATION
        elif prefix == "me":
            return MagnitudeScale.ENERGY
        elif prefix == "ml":
            return MagnitudeScale.LOCAL
        elif prefix == "mw":
            return MagnitudeScale.MOMENT
        elif prefix == "ms":
            return MagnitudeScale.SURFACE
        else:
            return MagnitudeScale.OTHER


@traced(logging.getLogger(__name__))
@logged(logging.getLogger(__name__))
class MagnitudeCategory(Enum):
    """
    Enumeration for earthquake magnitude scale.
    """
    MINOR = 0
    LIGHT = 1
    MODERATE = 2
    STRONG = 3
    MAJOR = 4
    GREAT = 5

    @staticmethod
    def from_magnitude(value: float) -> MagnitudeCategory:
        """Convert from magnitude.
        """
        if value >= 8.0:
            return MagnitudeCategory.GREAT
        elif value >= 7.0:
            return MagnitudeCategory.MAJOR
        elif value >= 6.0:
            return MagnitudeCategory.STRONG
        elif value >= 5.0:
            return MagnitudeCategory.MODERATE
        elif value >= 4.0:
            return MagnitudeCategory.LIGHT
        else:
            return MagnitudeCategory.MINOR


@traced(logging.getLogger(__name__))
@logged(logging.getLogger(__name__))
class Catalog:
    """
    Class for earthquake catalog containing data and visualization logic.
    """

    DATA_FOLDER = Path(__file__).parent.parent.parent / "data"

    def __init__(self, use_raw: bool = False):
        pio.templates.default = "plotly_white"
        if use_raw:
            self.faults = self._load_raw_faults()
            self.earthquakes = self._load_raw_earthquakes()
            self._add_fault_data()
        else:
            self.faults = self._load_faults()
            self.earthquakes = self._load_earthquakes()
        self.__log.info("First five lines of faults dataframe:\n{}".format(self.faults.head()))
        self.__log.info("First five lines of earthquakes dataframe:\n{}".format(self.earthquakes.head()))

    def save(self) -> None:
        self._save_faults()
        self._save_earthquakes()

    def plot_earthquake_map(self, earthquakes: pd.DataFrame = None, faults: pd.DataFrame = None) -> go.Figure:
        figure = go.Figure()
        if earthquakes is None:
            earthquakes = self.earthquakes
        if faults is None:
            faults = self.faults
        for index, row in faults.iterrows():
            fault_line = go.Scattergeo(
                lat=row["latitudes"], lon=row["longitudes"], mode="lines",
                showlegend=False, line=dict(color="black", width=2), hoverinfo="skip"
            )
            figure.add_trace(fault_line)
        earthquake_points = go.Scattergeo(
            lat=earthquakes["latitude"], lon=earthquakes["longitude"], mode="markers",
            hoverinfo="text", hovertext=earthquakes["magnitude_hover_text"],
            showlegend=False,
            marker=dict(
                color=earthquakes["magnitude"],
                colorbar=dict(
                    title="Magnitude"
                )
            )
        )
        figure.add_trace(earthquake_points)
        figure.update_geos(
            projection_type="equirectangular",
            showland=True, landcolor="rgb(230, 145, 56)",
            showocean=True, oceancolor="rgb(0, 255, 255)", lakecolor="rgb(0, 255, 255)"
        )
        figure.update_layout(margin=dict(l=10, r=0, b=0, t=0, pad=0))
        config = dict(displayModeBar=False, displaylogo=False, scrollZoom=False, showTips=False, staticPlot=True)
        figure.show(config=config)
        return figure

    def plot_magnitude_vs_distance(self, data: pd.DataFrame = None) -> go.Figure:
        if data is None:
            data = self.earthquakes
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=data["fault_distance"], y=data["magnitude"], mode="markers"))
        figure.show()
        return figure

    def plot(
        self, data: pd.DataFrame, fits: Union[pd.DataFrame, List[pd.DataFrame]] = None,
        title: str = "", legend1: str = "data", legend2: Union[str, List[str]] = "fit"
    ) -> go.Figure:
        if fits is None:
            figure = self._plot_data(data, legend1)
        else:
            if isinstance(fits, list):
                figure = self._plot_data_and_fits(data, fits, legend1, legend2)
            else:
                figure = self._plot_data_and_fit(data, fits, legend1, legend2)
        figure.update_xaxes(title_text="Distance to Closest Fault (km)")
        figure.update_yaxes(title_text="Probability of Strong Earthquake")
        figure.update_layout(
            title=dict(text=title, xanchor="left", yanchor="top"),
            legend=dict(xanchor="right", yanchor="top")
        )
        figure.show()
        return figure

    def _plot_data(self, data: pd.DataFrame, legend1: str) -> go.Figure:
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=data["distance"], y=data["probability"], mode="markers", name=legend1))
        return figure

    def _plot_data_and_fit(
        self, data: pd.DataFrame, fit: pd.DataFrame, legend1: str, legend2: str
    ) -> go.Figure:
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=data["distance"], y=data["probability"], mode="markers", name=legend1))
        figure.add_trace(go.Scatter(x=fit["distance"], y=fit["fit"], mode="lines", name=legend2))
        return figure

    def _plot_data_and_fits(
        self, data: pd.DataFrame, fits: List[pd.DataFrame], legend1: str, legend2: str
    ) -> go.Figure:
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=data["distance"], y=data["probability"], mode="markers", name=legend1))
        figure.add_trace(go.Scatter(x=fits[0]["distance"], y=fits[0]["fit"], mode="lines", name=legend2[0]))
        figure.add_trace(go.Scatter(x=fits[1]["distance"], y=fits[1]["fit"], mode="lines", name=legend2[1]))
        figure.add_trace(go.Scatter(x=fits[2]["distance"], y=fits[2]["fit"], mode="lines", name=legend2[2]))
        figure.add_trace(go.Scatter(x=fits[3]["distance"], y=fits[3]["fit"], mode="lines", name=legend2[3]))
        return figure

    def compute_error(self, data: pd.DataFrame, fit: pd.DataFrame) -> Tuple[float, float]:
        r2 = r2_score(data["probability"], fit["fit"])
        mse = mean_squared_error(data["probability"], fit["fit"])
        return r2, mse

    def compute_probabilities(self, data: pd.DataFrame = None) -> pd.DataFrame:
        if data is None:
            data = self.earthquakes
        max_distance = math.ceil(data["fault_distance"].max())
        distances = []
        probabilities = []
        for distance in range(0, max_distance):
            if distance == 0:
                distance = 0.001
            distances.append(distance)
            df = self.earthquakes[self.earthquakes["fault_distance"] >= distance]
            probabilities.append(len(df)/len(self.earthquakes))
        df = pd.DataFrame()
        df["distance"] = pd.Series(distances)
        df["probability"] = pd.Series(probabilities)
        return df

    def fit_log_decay(self, data: pd.DataFrame) -> Tuple[Any, pd.DataFrame]:
        params, df = self._fit(data, lambda x, a0, a1, a2: a0 - a1 * np.log(a2*x))
        self.__log.info("Logarithmic decay fit = {} - {} * log({} * d)".format(*params))
        return params, df

    def fit_exp_decay(self, data: pd.DataFrame) -> Tuple[Any, pd.DataFrame]:
        params, df = self._fit(data, lambda d, a0, a1, a2: a0 * np.exp(-a1 * d) + a2)
        self.__log.info("Exponential decay fit = {} * exp(-{} * d) + {})".format(*params))
        return params, df

    def fit_poly2(self, data: pd.DataFrame) -> Tuple[Any, pd.DataFrame]:
        params, df = self._fit(data, lambda d, a0, a1, a2: a0 * d**2 + a1 * d + a2)
        self.__log.info("2nd degree polynomial fit = {} * d^2 + {} * d + {})".format(*params))
        return params, df

    def fit_poly3(self, data: pd.DataFrame) -> Tuple[Any, pd.DataFrame]:
        params, df = self._fit(data, lambda d, a0, a1, a2, a3: a0 * d**3 + a1 * d**2 + a2 * d + a3)
        self.__log.info("3rd degree polynomial fit = {} * d^3 + {} * d^2 + {} * d + {})".format(*params))
        return params, df

    def _fit(
        self, data: pd.DataFrame, func: Callable[[float, ...], float]
    ) -> Tuple[Any, pd.DataFrame]:
        params, _ = curve_fit(func, data["distance"].values, data["probability"].values, maxfev=1000)
        fit = []
        for d in data["distance"].values:
            fit.append(func(d, *params))
        df = pd.DataFrame()
        df["distance"] = pd.Series(data["distance"].values)
        df["fit"] = pd.Series(fit)
        return params, df

    def predict_log_decay(self, distances: np.ndarray, *params) -> pd.DataFrame:
        return self._predict(distances, lambda x, a0, a1, a2: a0 - a1 * np.log(a2*x), *params)

    def predict_exp_decay(self, distances: np.ndarray, *params) -> pd.DataFrame:
        return self._predict(distances, lambda d, a0, a1, a2: a0 * np.exp(-a1 * d) + a2, *params)

    def predict_poly2(self, distances: np.ndarray, *params) -> pd.DataFrame:
        return self._predict(distances, lambda d, a0, a1, a2: a0 * d**2 + a1 * d + a2, *params)

    def predict_poly3(self, distances: np.ndarray, *params) -> pd.DataFrame:
        return self._predict(distances, lambda d, a0, a1, a2, a3: a0 * d**3 + a1 * d**2 + a2 * d + a3, *params)

    def _predict(self, distances: np.ndarray, func: Callable[[float, ...], float], *params) -> pd.DataFrame:
        prediction = []
        for d in distances:
            prediction.append(func(d, *params))
        df = pd.DataFrame()
        df["distance"] = pd.Series(distances)
        df["fit"] = pd.Series(prediction)
        return df

    def filter_by_datetime(self, min_datetime: str = None, max_datetime: str = None) -> pd.DataFrame:
        if min_datetime is None:
            if max_datetime is None:
                data = self.earthquakes
            else:
                data = self.earthquakes[self.earthquakes["datetime"] < max_datetime]
        else:
            if max_datetime is None:
                data = self.earthquakes[self.earthquakes["datetime"] >= min_datetime]
            else:
                data = self.earthquakes[
                    (self.earthquakes["datetime"] >= min_datetime) & (self.earthquakes["datetime"] < max_datetime)
                    ]
        self.__log.info("Found {} earthquakes between dates {} to {}.".format(len(data), min_datetime, max_datetime))
        return data

    def filter_by_location(
            self, latitudes: Tuple[float, float] = None, longitudes: Tuple[float, float] = None
    ) -> pd.DataFrame:
        min_lat = self.earthquakes["Latitude"].min()
        max_lat = self.earthquakes["Latitude"].max()
        min_lon = self.earthquakes["Longitude"].min()
        max_lon = self.earthquakes["Longitude"].max()
        lat1 = min_lat if latitudes[0] is None else max(latitudes[0], min_lat)
        lat2 = max_lat if latitudes[1] is None else min(latitudes[1], max_lat)
        lon1 = min_lon if longitudes[0] is None else max(longitudes[0], min_lon)
        lon2 = max_lon if longitudes[1] is None else min(longitudes[1], max_lon)
        filtered_by_latitude = self.earthquakes[
            (self.earthquakes["Latitude"] >= lat1) & (self.earthquakes["Latitude"] < lat2)
            ]
        data = filtered_by_latitude[
            (filtered_by_latitude["Longitude"] >= lon1) & (filtered_by_latitude["Longitude"] < lon2)
            ]
        self.__log.info(
            "Found {} earthquakes between coordinates ({}, {}) to ({}, {}).".format(
                len(data), latitudes[0], latitudes[1], longitudes[0], longitudes[1]
            )
        )
        return data

    def filter_by_depth(self, min_depth: float = None, max_depth: float = None) -> pd.DataFrame:
        if min_depth is None:
            if max_depth is None:
                data = self.earthquakes
            else:
                data = self.earthquakes[self.earthquakes["depth"] < max_depth]
        else:
            if max_depth is None:
                data = self.earthquakes[self.earthquakes["depth"] >= min_depth]
            else:
                data = self.earthquakes[
                    (self.earthquakes["depth"] >= min_depth) & (self.earthquakes["depth"] < max_depth)
                    ]
        self.__log.info("Found {} earthquakes between depths {} to {}.".format(len(data), min_depth, max_depth))
        return data

    def filter_by_magnitude_scale(self, scale: MagnitudeScale) -> pd.DataFrame:
        data = self.earthquakes[self.earthquakes["magnitude_scale_value"] == scale.value]
        self.__log.info("Found {} earthquakes for {} magnitude scale.".format(len(data), scale.name))
        return data

    def filter_by_magnitude(self, min_magnitude: float = None, max_magnitude: float = None) -> pd.DataFrame:
        if min_magnitude is None:
            if max_magnitude is None:
                data = self.earthquakes
            else:
                data = self.earthquakes[self.earthquakes["magnitude"] < max_magnitude]
        else:
            if max_magnitude is None:
                data = self.earthquakes[self.earthquakes["magnitude"] >= min_magnitude]
            else:
                data = self.earthquakes[
                    (self.earthquakes["magnitude"] >= min_magnitude) & (self.earthquakes["magnitude"] < max_magnitude)
                    ]
        self.__log.info(
            "Found {} earthquakes between magnitudes {} to {}.".format(len(data), min_magnitude, max_magnitude)
        )
        return data

    @Timer(text="Loaded fault data in {:.3f} seconds.", logger=logging.getLogger(__name__).info)
    def _load_faults(self) -> pd.DataFrame:
        path = (Catalog.DATA_FOLDER / "faults.csv").absolute()
        data = pd.read_csv(path)
        data["latitudes"] = data["latitudes"].apply(lambda value: np.asarray(literal_eval(value.replace(" ", ","))))
        data["longitudes"] = data["longitudes"].apply(lambda value: np.asarray(literal_eval(value.replace(" ", ","))))
        return data

    @Timer(text="Loaded raw fault data in {:.3f} seconds.", logger=logging.getLogger(__name__).info)
    def _load_raw_faults(self) -> pd.DataFrame:
        """
        Load active fault data provided by the GEM Global Active Faults Database (GEM GAF-DB).
        https://github.com/GEMScienceTools/gem-global-active-faults
        """
        rows = []
        geo_data = gpd.read_file((Catalog.DATA_FOLDER / "gem_active_faults.geojson").absolute())
        for index, row in geo_data.iterrows():
            fault_type = FaultCategory.from_gem(row.slip_type)
            if row.net_slip_rate is None or not row.net_slip_rate.startswith("("):
                slip_rate = np.nan
            else:
                slip_rate = list(
                    map(float, filter(None, row.net_slip_rate.replace("(", "").replace(")", "").split(",")))
                )[0]
            if isinstance(row.geometry, LineString):
                lines = [row.geometry]
            else:
                lines = row.geometry.geoms
            latitudes = []
            longitudes = []
            for line in lines:
                x, y = line.xy
                latitudes = np.append(latitudes, y)
                latitudes = np.append(latitudes, None)
                longitudes = np.append(longitudes, x)
                longitudes = np.append(longitudes, None)
            coordinates = list(zip(latitudes, longitudes))
            size = len(coordinates)
            lines = []
            for i, coordinate in enumerate(coordinates):
                if i < size - 1:
                    p1 = (coordinate[0], coordinate[1])
                    p2 = (coordinates[i+1][0], coordinates[i+1][1])
                    lines.append((p1, p2))
            cx, cy = row.geometry.centroid.xy
            rows.append({
                "name": row["name"], "centroid": (cy[0], cx[0]), "lines": lines,
                "latitudes": latitudes, "longitudes": longitudes,
                "fault_category_key": fault_type.name, "fault_category_value": fault_type.value, "slip_rate": slip_rate
            })
        data = pd.DataFrame(
            data=rows,
            columns=[
                "name", "centroid", "lines", "latitudes", "longitudes",
                "fault_category_key", "fault_category_value", "slip_rate"
            ]
        )
        self.__log.info("Completed loading {} active faults from GEM database.".format(len(data)))
        return data

    def _save_faults(self) -> None:
        path = (Catalog.DATA_FOLDER / "faults.csv").absolute()
        self.faults.to_csv(path, index=False)
        self.__log.info("Saved {} active faults to {}.".format(len(self.faults), path))

    @Timer(text="Loaded earthquake data in {:.3f} seconds.", logger=logging.getLogger(__name__).info)
    def _load_earthquakes(self) -> pd.DataFrame:
        path = (Catalog.DATA_FOLDER / "earthquakes.csv").absolute()
        data = pd.read_csv(path)
        return data

    @Timer(text="Loaded raw earthquake data in {:.3f} seconds.", logger=logging.getLogger(__name__).info)
    def _load_raw_earthquakes(self) -> pd.DataFrame:
        # data = self._load_earthquake_catalog("before_1900")
        # data = pd.concat([data, self._load_earthquake_catalog("1900_10")])
        # data = pd.concat([data, self._load_earthquake_catalog("1910_20")])
        # data = pd.concat([data, self._load_earthquake_catalog("1920_30")])
        # data = pd.concat([data, self._load_earthquake_catalog("1930_40")])
        # data = pd.concat([data, self._load_earthquake_catalog("1940_50")])
        # data = pd.concat([data, self._load_earthquake_catalog("1950_60")])
        # data = pd.concat([data, self._load_earthquake_catalog("1960_70")])
        data = self._load_earthquake_catalog("2011")
        for year in range(2012, 2022):
            data = pd.concat([data, self._load_earthquake_catalog(str(year))])
        self.__log.info("Completed loading {} earthquakes into catalog.".format(len(data)))
        return data

    def _load_earthquake_catalog(self, postfix: str) -> pd.DataFrame:
        """
        Load earthquake catalog provided by the Advanced National Seismic System (ANSS)
        Comprehensive Earthquake Catalog (ComCat).
        https://earthquake.usgs.gov/earthquakes/search/
        """
        data = pd.read_csv(
            "../data/usgs/earthquakes_{}.csv".format(postfix),
            header=None,
            skiprows=1,
            delimiter=",",
            decimal=".",
            usecols=[0, 1, 2, 3, 4, 5],
            parse_dates=[1]
        )
        data.columns = ["datetime", "latitude", "longitude", "depth", "magnitude", "magnitude_scale"]
        data["depth"].fillna(10.0, inplace=True)
        data["depth"] = np.where(data["depth"] >= 0.0, data["depth"], 10.0)
        data["depth_category_key"] = data["depth"].map(lambda value: DepthCategory.from_depth(value).name)
        data["depth_category_value"] = data["depth"].map(lambda value: DepthCategory.from_depth(value).value)
        data["magnitude_scale_key"] = data["magnitude_scale"].map(
            lambda value: MagnitudeScale.from_prefix(str(value)).to_prefix()
        )
        data["magnitude_scale_value"] = data["magnitude_scale"].map(
            lambda value: MagnitudeScale.from_prefix(str(value)).value
        )
        data["magnitude_category_key"] = data["magnitude"].map(
            lambda value: MagnitudeCategory.from_magnitude(value).name
        )
        data["magnitude_category_value"] = data["magnitude"].map(
            lambda value: MagnitudeCategory.from_magnitude(value).value
        )
        data["magnitude_bubble_size"] = data["magnitude"].map(lambda value: 10 * (2 ** (math.floor(value) - 4)))
        data["magnitude_hover_text"] = data["magnitude"].astype(str) + " (" + data["magnitude_scale_key"] + ")"
        data = data.drop("magnitude_scale", axis=1)
        data = data[data["magnitude"] >= 6]
        self.__log.debug("Loaded {} earthquakes from the ANSS catalog for period [{}].".format(len(data), postfix))
        return data

    def _save_earthquakes(self) -> None:
        path = (Catalog.DATA_FOLDER / "earthquakes.csv").absolute()
        self.earthquakes.to_csv(path, index=False)
        self.__log.info("Saved {} earthquakes to {}.".format(len(self.earthquakes), path))

    @Timer(text="Added fault data in {:.3f} seconds.", logger=logging.getLogger(__name__).info)
    def _add_fault_data(self) -> None:
        self.earthquakes["fault"] = self.earthquakes.apply(
            lambda row: self._find_closest_fault(float(row["latitude"]), float(row["longitude"])), axis=1
        )
        print(self.earthquakes["fault"])
        self.earthquakes["fault_name"] = self.earthquakes["fault"].apply(
            lambda x: x[0] if isinstance(x, list) else np.nan
        )
        self.earthquakes["fault_distance"] = self.earthquakes["fault"].apply(
            lambda x: x[1] if isinstance(x, list) else np.nan
        )
        self.earthquakes["fault_category_key"] = self.earthquakes["fault"].apply(
            lambda x: x[2] if isinstance(x, list) else np.nan
        )
        self.earthquakes["fault_category_value"] = self.earthquakes["fault"].apply(
            lambda x: x[3] if isinstance(x, list) else np.nan
        )
        self.earthquakes["fault_slip_rate"] = self.earthquakes["fault"].apply(
            lambda x: x[4] if isinstance(x, list) else np.nan
        )
        self.earthquakes = self.earthquakes.drop("fault", axis=1)

    # @Timer(text="Found closest fault in {:.3f} seconds.", logger=logging.getLogger(__name__).info)
    def _find_closest_fault(self, latitude: float, longitude: float) -> List[Any]:
        point = (latitude, longitude)
        self.faults["distance"] = self.faults["lines"].apply(
            lambda lines: self._find_shortest_distance_to_fault(point, lines)
        )
        min_distance = self.faults["distance"].min()
        fault = self.faults[self.faults["distance"] == min_distance]
        if fault["name"].values[0] is None:
            name = "Unknown"
        else:
            name = fault["name"].values[0]
        return [
            name, min_distance, fault["fault_category_key"].values[0], fault["fault_category_value"].values[0],
            fault["slip_rate"].values[0]
        ]

    def _find_shortest_distance_to_fault(
            self, point: Tuple[float, float], lines: List[Tuple[Tuple[float, float], Tuple[float, float]]]
    ) -> float:
        # df = pd.DataFrame(lines, columns=["p1", "p2"])
        # df["distance"] = df.apply(
        #     lambda row: self._calculate_distance_to_line(point, row["p1"], row["p2"]), axis=1
        # )
        # return df["distance"].min()
        distances = []
        for line in lines:
            distance = self._calculate_distance_to_line(point, line[0], line[1])
            distances.append(distance)
        return min(distances)

    def _calculate_distance_to_line(
            self, p: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> float:
        if p2[0] is None or p2[1] is None:
            return np.nan
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        d2 = dx*dx + dy*dy
        if d2 > 0:
            u = ((p[0] - p1[0]) * dx + (p[1] - p1[1]) * dy) / float(d2)
            u = 0 if u < 0 else 1 if u > 1 else u
        else:
            u = 0
        x = p1[0] + u * dx
        y = p1[1] + u * dy
        return haversine(p, (x, y))
