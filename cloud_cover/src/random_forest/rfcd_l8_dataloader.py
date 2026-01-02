import os
import rasterio
import numpy as np
import math
    
class L8RFCDDataLoader:
    """
    Landsat-8 Biome Cloud dataset loader for RFCD
    Uses RGB + NIR + manual fixedmask.img
    Gives RGB, NIR bands + Cloud mask and valid mask
    """

    def __init__(self, scene_dir: str):
        self.scene_dir = scene_dir
        self.files = os.listdir(scene_dir)

    # Find Landsat band
    def _find_band(self, band_id):
        for f in self.files:
            if f.upper().endswith(f"_B{band_id}.TIF"):
                return os.path.join(self.scene_dir, f)
        raise FileNotFoundError(f"Band B{band_id} not found in {self.scene_dir}")

    # Find manual cloud mask
    def _find_mask(self):
        for f in self.files:
            if f.lower().endswith("fixedmask.img"):
                return os.path.join(self.scene_dir, f)
        return None

    # Read Metadata XML file 
    def read_mtl(self, mtl_path):
        params = {}
        with open(mtl_path) as f:
            for line in f:
                if "=" in line:
                    k,v = line.split("=")
                    params[k.strip()] = v.strip().replace('"','')
        return params

    # Convert raw DN Landsat-8 values to TOA reflectance
    def landsat_toa_reflectance(self, dn, band_id, mtl):
        M = float(mtl[f"REFLECTANCE_MULT_BAND_{band_id}"])
        A = float(mtl[f"REFLECTANCE_ADD_BAND_{band_id}"])
        sun = float(mtl["SUN_ELEVATION"])
        
        rho = M * dn + A
        rho = rho / math.sin(math.radians(sun))

        return rho

    # Read raster
    def _read(self, path):
        with rasterio.open(path) as src:
            return src.read(1)

    # Main loader
    def load(self):
        """Load bands - Raw DBN values"""
        # blue  = self._read(self._find_band(2)).astype(np.float32)
        # green = self._read(self._find_band(3)).astype(np.float32)
        # red   = self._read(self._find_band(4)).astype(np.float32)
        # nir   = self._read(self._find_band(5)).astype(np.float32)
        
        """Load bands - TOA Reflectance converted values"""
        mtl_path = [f for f in self.files if f.endswith("_MTL.txt")][0]
        mtl = self.read_mtl(os.path.join(self.scene_dir, mtl_path))
        
        dn_blue = self._read(self._find_band(2)).astype(np.float32)
        dn_green = self._read(self._find_band(3)).astype(np.float32)
        dn_red = self._read(self._find_band(4)).astype(np.float32)
        dn_nir = self._read(self._find_band(5)).astype(np.float32)

        mask = self._read(self._find_mask())
        dn_red = self._read(self._find_band(4))
        valid = (mask != 0) & (dn_red > 0) & (dn_green > 0) & (dn_blue > 0) & (dn_nir > 0)

        blue = np.zeros_like(dn_blue, dtype=np.float32)
        green = np.zeros_like(dn_green, dtype=np.float32)
        red = np.zeros_like(dn_red, dtype=np.float32)
        nir = np.zeros_like(dn_nir, dtype=np.float32)

        blue[valid] = self.landsat_toa_reflectance(dn_blue[valid], 2, mtl)
        green[valid] = self.landsat_toa_reflectance(dn_green[valid], 3, mtl)
        red[valid] = self.landsat_toa_reflectance(dn_red[valid], 4, mtl)
        nir[valid] = self.landsat_toa_reflectance(dn_nir[valid], 5, mtl)

        H, W = red.shape

        # Load manual mask
        mask_path = self._find_mask()

        if mask_path is not None:
            mask = self._read(mask_path)

            # Valid data pixels (exclude Fill)
            # For TOA Reflectance
            valid = (mask != 0) & (dn_red > 0) & (dn_green > 0) & (dn_blue > 0) & (dn_nir > 0)
            # For Raw DN values
            # valid = (mask != 0) & (red > 0) & (green > 0) & (blue > 0) & (nir > 0)

            # RFCD labels
            cloud = np.zeros((H, W), dtype=np.uint8)
            cloud[(mask == 192) | (mask == 255)] = 1

        else:
            # 0% cloud scene
            cloud = np.zeros((H, W), dtype=np.uint8)
            valid = np.ones((H, W), dtype=bool)

        return blue, green, red, nir, cloud, valid

if __name__ == "__main__":
    pass
    # import rasterio

    # scene = "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Training_L8_Data/Barren/LC80420082013220LGN00/BC/LC80420082013220LGN00"
    # loader = L8RFCDDataLoader(scene)

    # blue, green, red, nir, cloud_mask, valid = loader.load()

    # print("Blue band shape:", blue.shape) 
    # print("Green band shape:", green.shape) 
    # print("Red band shape:", red.shape) 
    # print("NIR band shape:", nir.shape) 
    # print("Cloud Mask band shape:", cloud_mask.shape)
    # print("Valid Mask band shape:", valid.shape)

    # print("Red band TOA:", red[0:2, 0:9])
    # print("Green band TOA:", green[0:2, 0:9])
    # print("Blue band TOA:", blue[0:2, 0:9])
    # print("NIR band TOA:", nir[0:2, 0:9])
    # print("Red band Unique:", np.unique(red))
    # print("Green band Unique:", np.unique(green))
    # print("Blue band Unique:", np.unique(blue))
    # print("NIR band Unique:", np.unique(nir))

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
