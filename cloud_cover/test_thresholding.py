import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling
import glob
import os
from pyproj import CRS
import xml.etree.ElementTree as ET
from pyproj import Transformer
from src.physical_cloud_detection.threshold_mask import ThresholdMask
import matplotlib.pyplot as plt

def get_sentinel_crs(safe_root):
    """Extract CRS from Sentinel-2 metadata XML file."""
    xml_file = glob.glob(os.path.join(safe_root, "**", "MTD_TL.xml"), recursive=True)[0]
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Find Tile_Geocoding block
    tg = root.find(".//{*}Tile_Geocoding")

    # Find CRS code inside it (no namespace on child)
    epsg = tg.find("HORIZONTAL_CS_CODE").text

    return CRS.from_string(epsg)



def find_band(safe_root, band):
    """
    Works for compact Sentinel-2 SAFE where files are:
    T32UPB_20251201T102411_B01.jp2
    """
    pattern = os.path.join(safe_root, "**", f"*_{band}.jp2")
    files = glob.glob(pattern, recursive=True)

    if len(files) == 0:
        raise FileNotFoundError(f"Band {band} not found in {safe_root}")

    return files[0]


xml_dir_path = "/home/echofusion/Downloads/S2B_MSIL1C_20251226T052139_N0511_R062_T43PER_20251226T072137 (2).SAFE/GRANULE/L1C_T43PER_A045991_20251226T053028"
safe_img_dir = "/home/echofusion/Downloads/S2B_MSIL1C_20251226T052139_N0511_R062_T43PER_20251226T072137 (2).SAFE/GRANULE/L1C_T43PER_A045991_20251226T053028/IMG_DATA"
dem_path = "/home/echofusion/Hemanth/Cloud_Cover/data/S2_dem/23N_dem_2.tif"

# Coastal Blue as reference band - 60m
b1_path = find_band(safe_img_dir, "B01")   # Coastal blue 60m

sentinel_crs = get_sentinel_crs(xml_dir_path)

with rasterio.open(b1_path) as ref:
    ref_profile = ref.profile
    ref_profile["crs"] = sentinel_crs
    ref_profile["transform"] = ref.transform

    ref_transform = ref.transform
    ref_crs = sentinel_crs
    H, W = ref.height, ref.width
    B1 = ref.read(1).astype(np.float32)


def resample_to_ref(path, ref_profile, resampling):
    """Resample a raster to match reference profile (resolution, shape, CRS, transform)"""
    with rasterio.open(path) as src:
        dst = np.zeros((ref_profile["height"], ref_profile["width"]), dtype=np.float32)

        reproject(
            source=src.read(1),
            destination=dst,
            src_transform=src.transform,
            src_crs=ref_profile["crs"],    # force tile CRS
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            resampling=resampling
        )
    return dst



bands = {}

bands["Blue"]  = resample_to_ref(find_band(safe_img_dir, "B02"), ref_profile, Resampling.average)
bands["Green"] = resample_to_ref(find_band(safe_img_dir, "B03"), ref_profile, Resampling.average)
bands["Red"]   = resample_to_ref(find_band(safe_img_dir, "B04"), ref_profile, Resampling.average)
bands["NIR"]   = resample_to_ref(find_band(safe_img_dir, "B08"), ref_profile, Resampling.average)
bands["RE"]    = resample_to_ref(find_band(safe_img_dir, "B06"), ref_profile, Resampling.average)

bands["CB"]    = B1   # Coastal blue already 60m

with rasterio.open(dem_path) as dem_src:
    dem = np.zeros((H, W), dtype=np.float32)

    reproject(
        source=dem_src.read(1),
        destination=dem,
        src_transform=dem_src.transform,
        src_crs=ref_crs, # From XML
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.bilinear
    )

image = np.stack([
    bands["Blue"],
    bands["Green"],
    bands["Red"],
    bands["NIR"],
    bands["CB"],
    bands["RE"]
], axis=-1)

# Convert image to Reflectance values. (This is for Sentinel-2 L1C data)
if image.max() > 5:
    image /= 10000.0

bounds = rasterio.transform.array_bounds(H, W, ref_transform)
# bounds = (ymin, xmin, ymax, xmax) → convert to lon/lat
# but Sentinel is in UTM → use pyproj

transformer = Transformer.from_crs(ref_crs, "EPSG:4326", always_xy=True)
lon1, lat1 = transformer.transform(bounds[1], bounds[0])
lon2, lat2 = transformer.transform(bounds[3], bounds[2])

bounds_latlon = [lon1, lat1, lon2, lat2]

tm = ThresholdMask(image, bounds_latlon, dem)
cloud_mask, snow_mask = tm.compute()

# Display cloud mask
plt.imshow(cloud_mask, cmap='gray')
plt.title("Cloud Mask")
plt.show()

# Save cloud mask
save_mask = r"/home/echofusion/Hemanth/Cloud_Cover/data/cloud_mask/cloud_mask_data2_band6_0.4.tif"
with rasterio.open(
    save_mask,
    "w",
    driver="GTiff",
    height=H,
    width=W,
    count=1,
    dtype=rasterio.uint16,
    crs=ref_crs,
    transform=ref_transform,
) as dst:
    dst.write(cloud_mask.astype(rasterio.uint16), 1)