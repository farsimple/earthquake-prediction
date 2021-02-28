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

from ast import literal_eval as make_tuple
from enum import Enum
import logging
import math
from pathlib import Path
from typing import Optional, Tuple

from autologging import logged, traced
from codetiming import Timer
import geopandas as gpd
import numpy as np
import pandas as pd
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

    def __init__(self):
        self.faults = self.load_faults()
        self.save_faults()
        self.earthquakes = self.load_earthquakes()
        self.add_fault_information()
        self.save_earthquakes()

    @Timer(text="Loaded earthquakes in {:.3f} seconds.", logger=logging.getLogger(__name__).info)
    def load_earthquakes(self) -> pd.DataFrame:
        data = self._load("before_1900")
        data = pd.concat([data, self._load("1900_10")])
        data = pd.concat([data, self._load("1910_20")])
        data = pd.concat([data, self._load("1920_30")])
        data = pd.concat([data, self._load("1930_40")])
        data = pd.concat([data, self._load("1940_50")])
        data = pd.concat([data, self._load("1950_60")])
        data = pd.concat([data, self._load("1960_70")])
        for year in range(1970, 2022):
            data = pd.concat([data, self._load(str(year))])
        self.__log.info("Completed loading {} earthquakes into catalog.".format(len(data)))
        self.__log.debug("First five lines of earthquake catalog:\n{}".format(data.head()))
        return data

    def save_earthquakes(self, name: str = "earthquakes.csv") -> None:
        path = (Catalog.DATA_FOLDER / name).absolute()
        self.earthquakes.to_csv(path, index=False)
        self.__log.info("Saved {} earthquakes to {}.".format(len(self.earthquakes), path))

    @Timer(text="Loaded faults in {:.3f} seconds.", logger=logging.getLogger(__name__).info)
    def load_faults(self) -> pd.DataFrame:
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
                sr, min_sr, max_sr = (np.nan, np.nan, np.nan)
            else:
                value = map(float, filter(None, row.net_slip_rate.replace('(', '').replace(')', '').split(',')))
                sr, min_sr, max_sr = (list(value) + [np.nan]*3)[:3]
            rows.append({
                "Name": row["name"],
                "CentroidLatitude": cy[0], "CentroidLongitude": cx[0],
                "Latitudes": latitudes, "Longitudes": longitudes,
                "SlipType": row.slip_type, "SlipRate": sr, "MinimumSlipRate": min_sr, "MaximumSlipRate": max_sr
            })
        data = pd.DataFrame(
            data=rows,
            columns=[
                "Name", "CentroidLatitude", "CentroidLongitude", "Latitudes", "Longitudes",
                "SlipType", "SlipRate", "MinimumSlipRate", "MaximumSlipRate"
            ]
        )
        self.__log.info("Completed loading {} active faults from GEM database.".format(len(data)))
        self.__log.debug("First five lines of faults:\n{}".format(data.head()))
        return data

    def save_faults(self, name: str = "faults.csv") -> None:
        path = (Catalog.DATA_FOLDER / name).absolute()
        self.faults.to_csv(path, index=False)
        self.__log.info("Saved {} active faults to {}.".format(len(self.faults), path))

    @Timer(text="Added fault information in {:.3f} seconds.", logger=logging.getLogger(__name__).info)
    def add_fault_information(self) -> None:
        self.earthquakes["FaultName"] = np.nan
        for index, row in self.earthquakes.iterrows():
            fault_index = self.find_closest_fault(float(row["Latitude"]), float(row["Longitude"]))
            self.earthquakes.at[index, "FaultName"] = fault_index

    def find_closest_fault(self, latitude: float, longitude: float) -> int:
        df = pd.DataFrame()
        temp1 = self.faults["CentroidLatitude"] - latitude
        temp2 = self.faults["CentroidLongitude"] - longitude
        df["distance"] = np.square(temp1) + np.square(temp2)
        return df["distance"].idxmin()

    def filter_by_datetime(self, min_datetime: str = None, max_datetime: str = None) -> pd.DataFrame:
        if min_datetime is None:
            if max_datetime is None:
                data = self.earthquakes
            else:
                data = self.earthquakes[self.earthquakes["DateTime"] < max_datetime]
        else:
            if max_datetime is None:
                data = self.earthquakes[self.earthquakes["DateTime"] >= min_datetime]
            else:
                data = self.earthquakes[
                    (self.earthquakes["DateTime"] >= min_datetime) & (self.earthquakes["DateTime"] < max_datetime)
                ]
        self.__log.info("Found {} earthquakes between dates {} to {}.".format(len(data), min_datetime, max_datetime))
        return data

    def filter_by_location(
        self, min_latitude: float = None, max_latitude: float = None,
        min_longitude: float = None, max_longitude: float = None
    ) -> pd.DataFrame:
        min_lat = self.earthquakes["Latitude"].min()
        max_lat = self.earthquakes["Latitude"].max()
        min_lon = self.earthquakes["Longitude"].min()
        max_lon = self.earthquakes["Longitude"].max()
        lat1 = min_lat if min_latitude is None else max(min_latitude, min_lat)
        lat2 = max_lat if max_latitude is None else min(max_latitude, max_lat)
        lon1 = min_lon if min_longitude is None else max(min_longitude, min_lon)
        lon2 = max_lon if max_longitude is None else min(max_longitude, max_lon)
        filtered_by_latitude = self.earthquakes[
            (self.earthquakes["Latitude"] >= lat1) & (self.earthquakes["Latitude"] < lat2)
        ]
        data = filtered_by_latitude[
            (filtered_by_latitude["Longitude"] >= lon1) & (filtered_by_latitude["Longitude"] < lon2)
        ]
        self.__log.info(
            "Found {} earthquakes between coordinates ({}, {}) to ({}, {}).".format(
                len(data), min_latitude, max_latitude, min_longitude, max_longitude
            )
        )
        return data

    def filter_by_depth(self, min_depth: float = None, max_depth: float = None) -> pd.DataFrame:
        if min_depth is None:
            if max_depth is None:
                data = self.earthquakes
            else:
                data = self.earthquakes[self.earthquakes["Depth"] < max_depth]
        else:
            if max_depth is None:
                data = self.earthquakes[self.earthquakes["Depth"] >= min_depth]
            else:
                data = self.earthquakes[
                    (self.earthquakes["Depth"] >= min_depth) & (self.earthquakes["Depth"] < max_depth)
                ]
        self.__log.info("Found {} earthquakes between depths {} to {}.".format(len(data), min_depth, max_depth))
        return data

    def filter_by_magnitude_scale(self, scale: MagnitudeScale) -> pd.DataFrame:
        data = self.data.drop(
            self.data[self.data["MagnitudeScale"].str.contains(
                scale.to_prefix(), case=False, na=False, regex=False
            )].index
        )
        self.__log.info("Found {} earthquakes for {} magnitude scale.".format(len(data), scale.name))
        return data

    def filter_by_magnitude(self, min_magnitude: float = None, max_magnitude: float = None) -> pd.DataFrame:
        if min_magnitude is None:
            if max_magnitude is None:
                data = self.earthquakes
            else:
                data = self.earthquakes[self.earthquakes["Magnitude"] < max_magnitude]
        else:
            if max_magnitude is None:
                data = self.earthquakes[self.earthquakes["Magnitude"] >= min_magnitude]
            else:
                data = self.earthquakes[
                    (self.earthquakes["Magnitude"] >= min_magnitude) & (self.earthquakes["Magnitude"] < max_magnitude)
                ]
        self.__log.info(
            "Found {} earthquakes between magnitudes {} to {}.".format(len(data), min_magnitude, max_magnitude)
        )
        return data

    def _load_greece_data(self) -> pd.DataFrame:
        """
        Load earthquake data provided by the Institute of Geodynamics, National Observatory of Athens.
        http://www.gein.noa.gr/en/seismicity/earthquake-catalogs
        """
        data = pd.read_csv(
            (Catalog.DATA_FOLDER / "greece/earthquakes.csv").absolute(),
            skiprows=1,
            delimiter=";",
            thousands=".",
            decimal=",",
            converters={5: lambda arg: str(arg).replace(",", ".")}
        )
        data["DateTime"] = pd.to_datetime(
            data.iloc[:, 0].map(str)
            + "-" + data.iloc[:, 1].map(str)
            + "-" + data.iloc[:, 2].map(str)
            + " " + data.iloc[:, 3].map(str)
            + ":" + data.iloc[:, 4].map(str)
            + ":" + data.iloc[:, 5].map(str)
        )
        data = data.iloc[:, 6:]
        data.insert(0, "DateTime", data.pop("DateTime"))
        data.columns = ["DateTime", "Latitude", "Longitude", "Depth", "Ms", "Mw"]
        data.insert(4, "Mb", np.nan)
        data.insert(5, "Mc", np.nan)
        data.insert(6, "Me", np.nan)
        data.insert(7, "Md", np.nan)
        data.insert(8, "Ml", np.nan)
        data["Mweq"] = np.nan
        self.__log.info("Loaded {} earthquakes from the Greece catalog.".format(len(data)))
        return data

    def _load_italy_data(self) -> pd.DataFrame:
        """
        Load earthquake data provided by the Istituto Nazionale di Geofisica e Vulcanologia.
        http://cnt.rm.ingv.it/en
        """
        data = pd.read_csv(
            (Catalog.DATA_FOLDER / "italy/earthquakes.csv").absolute(),
            skiprows=1,
            delimiter="|",
            usecols=[1, 2, 3, 4, 9, 10],
            parse_dates=[1]
        )
        data.columns = ["DateTime", "Latitude", "Longitude", "Depth", "type", "M"]
        data["Mb"] = np.where(data["type"].str.lower().str.startswith("mb"), data["M"], np.nan)
        data["Mc"] = np.where(data["type"].str.lower().str.startswith("mc"), data["M"], np.nan)
        data["Md"] = np.where(data["type"].str.lower().str.startswith("md"), data["M"], np.nan)
        data["Me"] = np.where(data["type"].str.lower().str.startswith("me"), data["M"], np.nan)
        data["Ml"] = np.where(data["type"].str.lower().str.startswith("ml"), data["M"], np.nan)
        data["Ms"] = np.where(data["type"].str.lower().str.startswith("ms"), data["M"], np.nan)
        data["Mw"] = np.where(data["type"].str.lower().str.startswith("mw"), data["M"], np.nan)
        data["Mweq"] = np.nan
        data.drop("type", axis=1, inplace=True)
        data.drop("M", axis=1, inplace=True)
        self.__log.info("Loaded {} earthquakes from the Italy catalog.".format(len(data)))
        return data

    def _load_mexico_data(self) -> pd.DataFrame:
        """
        Load earthquake data provided by
        Sawires, R., Santoyo, M.A., Peláez, J.A. et al.
        An updated and unified earthquake catalog from 1787 to 2018 for seismic hazard assessment studies in Mexico.
        Sci Data 6, 241 (2019).
        https://www.nature.com/articles/s41597-019-0234-z
        https://doi.org/10.6084/m9.figshare.c.4492763
        """
        data = pd.read_csv(
            (Catalog.DATA_FOLDER / "mexico/earthquakes.csv").absolute(),
            skiprows=1,
            delimiter=";",
            thousands=".",
            decimal=",",
            usecols=[0,1,2,3,4,5,6,7,9,11,13,15,17,19,21,23],
            converters={5: lambda arg: str(arg).replace(",", ".")}
        )
        data["DateTime"] = pd.to_datetime(
            data.iloc[:, 0].map(str)
            + "-" + data.iloc[:, 1].map(str)
            + "-" + data.iloc[:, 2].map(str)
            + " " + data.iloc[:, 3].map(str)
            + ":" + data.iloc[:, 4].map(str)
            + ":" + data.iloc[:, 5].map(str)
        )
        data = data.iloc[:, 6:]
        data.insert(0, "DateTime", data.pop("DateTime"))
        data.columns = ["DateTime", "Latitude", "Longitude", "Depth", "Mb", "Ms1", "Ms2", "Mw", "Md", "Ml", "Mweq"]
        data["Ms"] = np.where(
            (data["Ms1"] > 0.0) & (data["Ms2"] > 0.0),
            0.5 * (data["Ms1"] + data["Ms2"]),
            np.where(data["Ms1"] > 0.0, data["Ms1"], data["Ms2"])
        )
        data.drop("Ms1", axis=1, inplace=True)
        data.drop("Ms2", axis=1, inplace=True)
        self.__log.info("Loaded {} earthquakes from the Mexico catalog.".format(len(data)))
        return data

    def _load_russia_data(self) -> pd.DataFrame:
        """
        Load earthquake data provided by the Geophysical Survey of the Russian Academy of Sciences.
        http://eqru.gsras.ru
        """
        data = pd.read_csv(
            "../data/russia/earthquakes.csv",
            encoding="cp1251",
            skiprows=1,
            delimiter=";",
            thousands=".",
            decimal=",",
            usecols=[0,1,2,3,5,7,9,11]
        )
        data["DateTime"] = pd.to_datetime(
            data.iloc[:, 0].map(str)
            + " " + data.iloc[:, 1].map(str)
            + ":" + data.iloc[:, 2].map(str)
            + ":" + data.iloc[:, 3].map(str)
        )
        data = data.iloc[:, 4:]
        data.insert(0, "DateTime", data.pop("DateTime"))
        data.columns = ["DateTime", "Latitude", "Longitude", "Depth", "Ms"]
        data["Mb"] = np.nan
        data["Mc"] = np.nan
        data["Md"] = np.nan
        data["Me"] = np.nan
        data["Ml"] = np.nan
        data["Mw"] = np.nan
        data["Mweq"] = np.nan
        self.__log.info("Loaded {} earthquakes from the Russia catalog.".format(len(data)))
        return data

    def _load_turkey_data(self) -> pd.DataFrame:
        """
        Load earthquake data provided by the Disaster & Emergency Management Authority (AFAD).
        https://deprem.afad.gov.tr/depremkatalogu?lang=en
        """
        data = pd.read_csv(
            "../data/turkey/earthquakes.csv",
            skiprows=1,
            delimiter=",",
            decimal=".",
            usecols=[1,4,5,6,10,11],
            parse_dates=[1]
        )
        data.columns = ["DateTime", "Latitude", "Longitude", "Depth", "type", "M"]
        data["Mb"] = np.where(data["type"].str.lower().str.startswith("mb"), data["M"], np.nan)
        data["Mc"] = np.where(data["type"].str.lower().str.startswith("mc"), data["M"], np.nan)
        data["Md"] = np.where(data["type"].str.lower().str.startswith("md"), data["M"], np.nan)
        data["Me"] = np.where(data["type"].str.lower().str.startswith("me"), data["M"], np.nan)
        data["Ml"] = np.where(data["type"].str.lower().str.startswith("ml"), data["M"], np.nan)
        data["Ms"] = np.where(data["type"].str.lower().str.startswith("ms"), data["M"], np.nan)
        data["Mw"] = np.where(data["type"].str.lower().str.startswith("mw"), data["M"], np.nan)
        data["Mweq"] = np.nan
        data.drop("type", axis=1, inplace=True)
        data.drop("M", axis=1, inplace=True)
        self.__log.info("Loaded {} earthquakes from the Turkey catalog.".format(len(data)))
        return data

    def _load(self, postfix: str) -> pd.DataFrame:
        data = pd.read_csv(
            "../data/usgs/earthquakes_{}.csv".format(postfix),
            header=None,
            skiprows=1,
            delimiter=",",
            decimal=".",
            usecols=[0, 1, 2, 3, 4, 5],
            parse_dates=[1]
        )
        data.columns = ["DateTime", "Latitude", "Longitude", "Depth", "Magnitude", "MagnitudeScale"]
        data["Depth"].fillna(10.0, inplace=True)
        data["Depth"] = np.where(data["Depth"] >= 0.0, data["Depth"], 10.0)
        data["DepthCategory"] = data["Depth"].map(lambda d: DepthCategory.from_depth(d))
        data["MagnitudeScale"] = data["MagnitudeScale"].map(
            lambda value: MagnitudeScale.from_prefix(str(value)).to_prefix()
        )
        data["MagnitudeBubble"] = data["Magnitude"].map(lambda m: 10*(2**(math.floor(m) - 4)))
        data["MagnitudeHover"] = data["MagnitudeScale"] + " = " + data["Magnitude"].astype(str)
        self.__log.debug("Loaded {} earthquakes from the ANSS catalog for period [{}].".format(len(data), postfix))
        return data
