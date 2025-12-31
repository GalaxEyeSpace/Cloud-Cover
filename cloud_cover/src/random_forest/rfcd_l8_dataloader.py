import os
import rasterio
import numpy as np

class L8RFCDDataLoader:
    """
    Landsat-8 Biome Cloud dataset loader for RFCD
    Uses RGB + NIR + manual fixedmask.img
    Gives RGB, NIR bands + Cloud mask and valid mask
    """

    def __init__(self, scene_dir: str):
        self.scene_dir = scene_dir
        self.files = os.listdir(scene_dir)

    # --------------------------------------------
    # Find Landsat band
    # --------------------------------------------
    def _find_band(self, band_id):
        for f in self.files:
            if f.upper().endswith(f"_B{band_id}.TIF"):
                return os.path.join(self.scene_dir, f)
        raise FileNotFoundError(f"Band B{band_id} not found in {self.scene_dir}")

    # --------------------------------------------
    # Find manual cloud mask
    # --------------------------------------------
    def _find_mask(self):
        for f in self.files:
            if f.lower().endswith("fixedmask.img"):
                return os.path.join(self.scene_dir, f)
        return None

    # --------------------------------------------
    # Read raster
    # --------------------------------------------
    def _read(self, path):
        with rasterio.open(path) as src:
            return src.read(1)

    # --------------------------------------------
    # Main loader
    # --------------------------------------------
    def load(self):
        # Load bands
        blue  = self._read(self._find_band(2)).astype(np.float32)
        green = self._read(self._find_band(3)).astype(np.float32)
        red   = self._read(self._find_band(4)).astype(np.float32)
        nir   = self._read(self._find_band(5)).astype(np.float32)

        H, W = red.shape

        # Load manual mask
        mask_path = self._find_mask()

        if mask_path is not None:
            mask = self._read(mask_path)

            # Valid data pixels (exclude Fill)
            valid = (mask != 0)

            # RFCD labels
            cloud = np.zeros((H, W), dtype=np.uint8)
            cloud[(mask == 192) | (mask == 255)] = 1

        else:
            # 0% cloud scene
            cloud = np.zeros((H, W), dtype=np.uint8)
            valid = np.ones((H, W), dtype=bool)

        return blue, green, red, nir, cloud, valid



# import rasterio

# scene = "/home/echofusion/Downloads/cloud_cover_data/LC81310182013108LGN01/BC/LC81310182013108LGN01"
# loader = L8RFCDDataLoader(scene)

# blue, green, red, nir, cloud_mask, valid = loader.load()

# print("Blue band shape:", blue.shape) 
# print("Green band shape:", green.shape) 
# print("Red band shape:", red.shape) 
# print("NIR band shape:", nir.shape) 
# print("Cloud Mask band shape:", cloud_mask.shape)
# print("Valid Mask band shape:", valid.shape)

# # Copy georeference from Red band
# red_path = loader._find_band(4)

# with rasterio.open(red_path) as src:
#     profile = src.profile.copy()

# profile.update(
#     dtype=rasterio.uint8,
#     count=1,
#     compress="lzw",
#     nodata=255
# )

# # Apply NoData
# cloud_out = cloud_mask.copy()
# cloud_out[~valid] = 255

# out_path = "/home/echofusion/Hemanth/Cloud_Cover/data/LC81310182013108LGN01_cloudmask_RFCD_3.tif"

# with rasterio.open(out_path, "w", **profile) as dst:
#     dst.write(cloud_out, 1)

# print("Saved:", out_path)