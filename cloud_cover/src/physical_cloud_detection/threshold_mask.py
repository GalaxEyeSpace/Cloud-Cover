"""Code to generate a basic cloud mask based thresholding on RGB, NIR, Coastal Blue and Red Edge bands. (Sentinel-2)"""

import numpy as np
from scipy.ndimage import uniform_filter

"""
Ground Rules for snow detection based on latitude and elevation:
If |latitude| ≤ 23.5° and elevation < 4500 m → NO SNOW
If |latitude| ≤ 23.5° and elevation ≥ 4500 m → SNOW POSSIBLE

23.5° - 35°	≥ 2,500 m
35° - 50°	≥ 1,000 m
> 50°	    ≥ 0 - 500 m

For tropical regions  (23.5 N to 23.5 S):
We can set a threshold of 4500m above which snow is possible.
"""

class ThresholdMask:
    def __init__(self, image_data:np.ndarray, bounds:np.array, dem_data:np.ndarray):
        self.image = image_data.astype(np.float32) # (H, W, C)
        self.dem_data = dem_data.astype(np.float32) #(H, W)
        # Bounds : Lon1, Lat1, Lon2, Lat2
        self.lat1 = bounds[1]
        self.lat2 = bounds[3]
        self.lat = 0.5 * (self.lat1 + self.lat2)
        
        self.snow_dem_limit = self.snow_elevation_threshold(self.lat)
        
        # bands
        self.B   = self.image[:,:,0]
        self.G   = self.image[:,:,1]
        self.R   = self.image[:,:,2]
        self.NIR = self.image[:,:,3]
        self.CB  = self.image[:,:,4]
        self.RE  = self.image[:,:,5]

        self.eps = 1e-6
        
        
    def snow_elevation_threshold(self, lat:float) -> int:
        """
        Determine Snow Altitude Limit based on latitude
        """
        lat = abs(lat)
        if lat <= 23.5:
            return 4500
        elif lat <= 35:
            return 2500
        elif lat <= 50:
            return 1000
        else:
            return 500
        
    def compute_indices(self):
        """ 
        Compute spectral indices needed for cloud detection
        1) VIS - reflectances avg of RGB. Clouds have high reflectance in VIS
        2) NDVI - Clouds have low NDVI values
        3) Red Edge ratio - Clouds have high reflectance in RE compared to red band
        """

        self.VIS = (self.B + self.G + self.R) / 3.0
        self.NDVI = (self.NIR - self.R) / (self.NIR + self.R + self.eps)
        self.re_ratio = self.RE / (self.R + self.eps)
        
    def cloud_candidates(self):
        """
        Use thresholds to get an estimate of cloud candidates (includes snow, bright surfaces too)
        VIS > 0.25
        NIR > 0.4
        np.abs(self.NDVI) < 0.20
        np.abs(self.re_ratio - 1.0) < 0.15
        """
        return (
            (self.VIS > 0.20) &
            (self.NIR > 0.4) &
            (np.abs(self.NDVI) < 0.20) &
            (np.abs(self.re_ratio - 1.0) < 0.15)            
        )
       
    def water_mask(self):
        return (self.NDVI < 1) & (self.NIR < 0.10)

    def snow_spectral(self):
        return (
            (self.VIS > 0.35) &
            (self.B > self.R) &
            (self.NIR < self.VIS) &
            (self.RE < self.R)
        )

    def snow_allowed(self):
        return self.dem_data > self.snow_dem_limit

    def desert_mask(self):
        return (
            (self.VIS > 0.30) &
            (self.CB < 0.15) &
            (self.NIR > self.VIS)
        )

    def dem_gradient(self):
        mean = uniform_filter(self.dem_data, size=15)
        return np.abs(self.dem_data - mean)

    def compute(self):
        self.compute_indices()

        cloud = self.cloud_candidates()

        # Remove water
        cloud &= ~self.water_mask()

        # Spectral snow
        snow_spec = self.snow_spectral()

        # DEM-allowed snow
        snow_allowed = self.snow_allowed()

        # High-confidence snow
        snow = snow_spec & snow_allowed

        # Spectral snow in impossible terrain → cloud
        false_snow = snow_spec & (~snow_allowed)
        cloud |= false_snow

        # Remove deserts
        cloud &= ~self.desert_mask()

        # DEM consistency
        grad = self.dem_gradient()
        terrain_like = grad > 200   # strong terrain slope
        cloud &= ~terrain_like

        return cloud.astype(np.uint8), snow.astype(np.uint8)

