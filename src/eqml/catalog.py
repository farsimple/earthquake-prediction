#  ********************************************************************************
#                            ___
#                           /\_ \           CS-C3240: Machine Learning
#     __     __     ___ ___ \//\ \          Student Project
#   /'__`\ /'__`\ /' __` __`\ \ \ \         Predicting Earthquake Magnitudes
#  /\  __//\ \L\ \/\ \/\ \/\ \ \_\ \_
#  \ \____\ \___, \ \_\ \_\ \_\/\____\      Copyright (c) 2021 FarSimple Oy
#   \/____/\/___/\ \/_/\/_/\/_/\/____/      All rights reserved.
#               \ \_\
#                \/_/
#
#  ********************************************************************************

from __future__ import annotations

from enum import Enum
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from autologging import logged, traced
from codetiming import Timer
import geopandas as gpd
from haversine import haversine
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shapely.geometry.linestring import LineString


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
class Catalog():
    """
    Class for earthquake catalog containing data and visualization logic.
    """

    DATA_FOLDER = Path(__file__).parent.parent.parent / "data"

    def __init__(self, load_raw: bool = False):
        if load_raw:
            self.faults = self._load_raw_faults()
            self.earthquakes = self._load_raw_earthquakes()
            self._add_fault_information()
        else:
            self.faults = self._load_faults()
            self.earthquakes = self._load_earthquakes()

    def save(self) -> None:
        self._save_faults()
        self._save_earthquakes()

    def plot_earthquake_map(self, earthquakes: pd.DataFrame = None, faults: pd.DataFrame = None) -> go.Figure:
        figure = go.Figure()
        figure.add_trace(self._generate_plot_layer_for_faults(faults))
        figure.add_trace(self._generate_plot_layer_for_earthquakes(earthquakes))
        figure.update_geos(
            projection_type="equirectangular",
            showcoastlines=True, coastlinecolor="White",
            showland=True, landcolor="Gainsboro"
        )
        figure.update_layout(title_text="Global ANSS Earthquake Catalog")
        config = dict({"displayModeBar": False, "displaylogo": False, "scrollZoom": False, "showTips": False})
        figure.show(config=config)
        return figure

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
        data = pd.read_csv(
            path,
            header=None,
            skiprows=1,
            delimiter=",",
            decimal=".",
            usecols=[0, 1, 2, 3, 4, 5],
            parse_dates=[1]
        )
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
            cx, cy = row.geometry.centroid.xy
            if row.net_slip_rate is None or not row.net_slip_rate.startswith("("):
                slip_rate = (np.nan, np.nan, np.nan)
            else:
                value = map(float, filter(None, row.net_slip_rate.replace('(', '').replace(')', '').split(',')))
                slip_rate = (list(value) + [np.nan]*3)[:3]
            rows.append({
                "name": row["name"], "centroid": (cy[0], cx[0]), "coordinates": zip(latitudes, longitudes),
                "slip_type": row.slip_type, "slip_rate": slip_rate
            })
        data = pd.DataFrame(
            data=rows,
            columns=["name", "centroid", "coordinates", "slip_type", "slip_rate"]
        )
        self.__log.info("Completed loading {} active faults from GEM database.".format(len(data)))
        self.__log.debug("First five lines of faults:\n{}".format(data.head()))
        return data

    def _save_faults(self) -> None:
        path = (Catalog.DATA_FOLDER / "faults.csv").absolute()
        self.faults.to_csv(path, index=False)
        self.__log.info("Saved {} active faults to {}.".format(len(self.faults), path))

    @Timer(text="Loaded raw earthquake data in {:.3f} seconds.", logger=logging.getLogger(__name__).info)
    def _load_raw_earthquakes(self) -> pd.DataFrame:
        data = self._load_earthquake_catalog("before_1900")
        data = pd.concat([data, self._load_earthquake_catalog("1900_10")])
        data = pd.concat([data, self._load_earthquake_catalog("1910_20")])
        data = pd.concat([data, self._load_earthquake_catalog("1920_30")])
        data = pd.concat([data, self._load_earthquake_catalog("1930_40")])
        data = pd.concat([data, self._load_earthquake_catalog("1940_50")])
        data = pd.concat([data, self._load_earthquake_catalog("1950_60")])
        data = pd.concat([data, self._load_earthquake_catalog("1960_70")])
        for year in range(1970, 2022):
            data = pd.concat([data, self._load_earthquake_catalog(str(year))])
        self.__log.info("Completed loading {} earthquakes into catalog.".format(len(data)))
        self.__log.debug("First five lines of earthquake catalog:\n{}".format(data.head()))
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
            lambda value: MagnitudeScale.from_prefix(str(value)).name
        )
        data["magnitude_scale_value"] = data["magnitude_scale"].map(
            lambda value: MagnitudeScale.from_prefix(str(value)).value
        )
        data["magnitude_bubble_size"] = data["magnitude"].map(lambda value: 10*(2**(math.floor(value) - 4)))
        data["magnitude_hover_text"] = data.apply(
            lambda row: MagnitudeScale.to_prefix(row["magnitude_scale"]) + " = " + row["magnitude"]
        )
        self.__log.debug("Loaded {} earthquakes from the ANSS catalog for period [{}].".format(len(data), postfix))
        return data

    def _save_earthquakes(self) -> None:
        path = (Catalog.DATA_FOLDER / "earthquakes.csv").absolute()
        self.earthquakes.to_csv(path, index=False)
        self.__log.info("Saved {} earthquakes to {}.".format(len(self.earthquakes), path))

    @Timer(text="Added fault information in {:.3f} seconds.", logger=logging.getLogger(__name__).info)
    def _add_fault_information(self) -> None:
        self.earthquakes["fault_name"] = np.nan
        self.earthquakes["fault_slip_type"] = np.nan
        self.earthquakes["fault_slip_rate"] = np.nan
        for index, row in self.earthquakes.iterrows():
            fault = self._find_closest_fault(float(row["latitude"]), float(row["longitude"]))
            self.earthquakes.at[index, "fault_name"] = fault["name"]
            self.earthquakes.at[index, "fault_slip_type"] = fault["slip_type"]
            self.earthquakes.at[index, "fault_slip_rate"] = fault["slip_rate"]
            self.__log.info("Added fault information for earthquake #{}.".format(index))

    def _find_closest_fault(self, latitude: float, longitude: float) -> Dict[str, Any]:
        self.faults["distance"] = self.faults["depth"].centroid(lambda value: haversine(value, (latitude, longitude)))
        fault = self.faults.iloc[self.faults["distance"].idxmin()]
        return fault.to_dict("records")[0]

    def _generate_plot_layer_for_faults(self, data: pd.DataFrame = None) -> go.Scattergeo:
        if data is None:
            data = self.faults
        lat, lon = list(*data["coordinates"].values)
        return go.Scattergeo(
            lat=lat, lon=lon, mode="lines", showlegend=False, line=dict(color="DimGray", width=1), hoverinfo="skip"
        )

    def _generate_plot_layer_for_earthquakes(self, data: pd.DataFrame = None) -> go.Scattergeo:
        if data is None:
            data = self.earthquakes
        return go.Scattergeo(
            lat=data["latitude"], lon=data["longitude"], mode="markers",
            hoverinfo="text", hovertext=data["magnitude_hover_text"],
            showlegend=False,
            marker=dict(
                size=data["magnitude_bubble_size"], sizemode="area",
                color=-data["depth"], reversescale=False, colorscale="Inferno",
                colorbar=dict(
                    title="Earthquake Depth (km)", tickmode="array", ticks="outside",
                    tickvals=[0, -70, -300],
                    ticktext=["Shallow (0-70)", "Intermediate (70-300)", "Deep (300+)"]
                ),
                line_color="rgb(40,40,40)", line_width=0.5
            )
        )
