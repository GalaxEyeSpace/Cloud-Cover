"""Download DEM from OpenTopography."""

import requests
import numpy as np
import rasterio
from rasterio.io import MemoryFile
import geopandas as gpd
import os
from dotenv import load_dotenv

load_dotenv()  

"""
Download DEM using OpenTopography OpenAPI

Supported products:
    - SRTMGL1  (30m)
    - SRTMGL3  (90m)
    - NASADEM
    - COP30
"""


def download_dem(
    bbox: gpd.GeoDataFrame, dem_type="SRTMGL1", output_path=None
) -> np.ndarray:
    """
    Download DEM from OpenTopography for a given bounding box.

    Args:
        bbox (tuple/list): (min_lon, min_lat, max_lon, max_lat)
        dem_type (str): DEM product ("SRTMGL1", "NASADEM", "COP30")
        output_path (str): File path to save output GeoTiff

    Returns:
        (np.ndarray, dict, affine)
        dem_array, metadata, transform
    """
    if bbox.crs.to_epsg() != 4326:
        bbox_ = bbox.to_crs(4326)
    else:
        bbox_ = bbox

    bounds = bbox_.bounds.iloc[0].to_list()  # (minx, miny, maxx, maxy)

    base_url = "https://portal.opentopography.org/API/globaldem"

    params = {
        "demtype": dem_type,
        "south": bounds[1],
        "north": bounds[3],
        "west": bounds[0],
        "east": bounds[2],
        "outputFormat": "GTiff",
        "API_Key": os.getenv("OPEN_TOPO_API"),
    }

    print("Requesting DEM from OpenTopography...")
    r = requests.get(base_url, params=params)

    if r.status_code != 200:
        raise RuntimeError(f"Error: HTTP {r.status_code}\n{r.text}")

    print("DEM data received. Writing to file...")

    # Write GeoTIFF received as bytes
    with MemoryFile(r.content) as memfile:
        with memfile.open() as dataset:
            dem_data = dataset.read(1)  # band 1
            meta = dataset.meta.copy()
            
            if output_path is not None:
                # Save to output path
                with rasterio.open(output_path, "w", **meta) as dst:
                    dst.write(dem_data, 1)

            return dem_data, meta["crs"], meta["transform"]

