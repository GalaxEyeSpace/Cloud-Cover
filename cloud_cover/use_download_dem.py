"""Code to download DEM for a given image bounding box and save as TIF file. Works with Sentinel-2 data."""

import rasterio
from src.physical_cloud_detection.download_dem import download_dem
import geopandas as gpd
from shapely.geometry import box
import xml.etree.ElementTree as ET
import glob
import os
from pyproj import CRS


# ------------------------
# Input paths
# ------------------------
image_path = r"/home/echofusion/Downloads/S2C_MSIL1C_20251201T102411_N0511_R065_T32UPB_20251201T121939.SAFE/GRANULE/L1C_T32UPB_A006470_20251201T102413/IMG_DATA/T32UPB_20251201T102411_B01.jp2"

tile_dir = r"/home/echofusion/Downloads/S2C_MSIL1C_20251201T102411_N0511_R065_T32UPB_20251201T121939.SAFE/GRANULE/L1C_T32UPB_A006470_20251201T102413"

dem_save_path = r"/home/echofusion/Hemanth/Cloud_Cover/data/S2_dem/T32UPB_dem.tif"


def get_sentinel_crs_from_xml(tile_dir):
    """
    Reads EPSG code from Sentinel-2 MTD_TL.xml
    Works for all tiles (e.g. EPSG:32643, EPSG:32755, etc.)
    
    Args:
        tile_dir (str): Path to Sentinel-2 Granule directory containing MTD_TL.xml. SAFE Format data.
    """
    xml_file = glob.glob(os.path.join(tile_dir, "MTD_TL.xml"))[0]

    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Tile_Geocoding has namespace, EPSG field does not
    tg = root.find(".//{*}Tile_Geocoding")
    epsg = tg.find("HORIZONTAL_CS_CODE").text

    return CRS.from_string(epsg)


# ------------------------
# Read Sentinel tile CRS from XML
# ------------------------
sentinel_crs = get_sentinel_crs_from_xml(tile_dir)
print("Sentinel CRS:", sentinel_crs)

# ------------------------
# Read image bounds (numbers only)
# ------------------------
with rasterio.open(image_path) as src:
    bounds = src.bounds

# ------------------------
# Build bounding box in correct CRS
# ------------------------
bbox_poly = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

gdf = gpd.GeoDataFrame({"geometry": [bbox_poly]}, crs=sentinel_crs)

print("Original bounds (UTM):", gdf.total_bounds)

# ------------------------
# Convert to WGS84 for DEM download
# ------------------------
gdf_wgs84 = gdf.to_crs(epsg=4326)

print("WGS84 bounds:", gdf_wgs84.total_bounds)

# ------------------------
# Download DEM
# ------------------------
dem_data, dem_crs, dem_transform = download_dem(
    bbox=gdf_wgs84,
    dem_type="SRTMGL1",
    output_path=dem_save_path
)

print("DEM shape:", dem_data.shape)
print("DEM max:", dem_data.max())
print("DEM min:", dem_data.min())
print("DEM CRS:", dem_crs)
print("DEM transform:", dem_transform)
