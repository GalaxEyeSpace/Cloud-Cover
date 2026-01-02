import joblib
import numpy as np
import cv2
from skimage.filters import gabor

class RandomForestCloudDetection:
    def __init__(self, input_img:np.ndarray, model_path:str=None):
        
        self.model = joblib.load(model_path) if model_path is not None else None
        
        self.input_img = input_img.astype(np.float32) # (C, H, W) -> R, G, B, NIR are required bands
        
        self.blue = self.input_img[0, :, :]
        self.green = self.input_img[1, :, :]
        self.red = self.input_img[2, :, :]
        self.nir = self.input_img[3, :, :]
        
        self.output_img = None
        # Valid reflectance mask (excludes fill pixels)
        self.valid = (self.red + self.green + self.blue) > 0
        
    def band_feature(self):
        """
        1) Band Feature: Uses reflectance values of RGB and NIR bands as features
        2) Each pixel is represented by its 4-band reflectance vector.
        """
        band_features = np.stack([self.blue, self.green, self.red, self.nir], axis=-1) # (H, W, 4)
        
        return band_features
    
    def color_feature(self):
        """
        1) Color Feature: Converts the image from RGB color space to IHS color space (Intensity, Hue, Saturation)
        2) RGB space: Dense clouds are bright but thin clouds are not.
        3) IHS space: Both clouds have high intensity and low saturtion.
        """
        eps = 1e-6
        tou = 0.05
                
        Intensity = (self.red + self.green + self.blue) / 3.0
        Saturation = 1 - ((3 * np.minimum(np.minimum(self.red, self.green), self.blue)) / (self.red + self.green + self.blue + eps))
        
        v = self.valid

        Inorm = np.zeros_like(Intensity)
        Snorm = np.zeros_like(Saturation)

        Inorm[v] = (Intensity[v] - Intensity[v].min()) / (Intensity[v].max() - Intensity[v].min() + eps)
        Snorm[v] = (Saturation[v] - Saturation[v].min()) / (Saturation[v].max() - Saturation[v].min() + eps)
        
        # Basal Figure (J)
        J = (Inorm + tou) / (Snorm +tou)
        
        color_features = np.stack([Intensity, Saturation, J], axis=-1) # (H, W, 3)
        
        return  color_features
    
    def dark_channel_feature(self, window_size:int=15):
        """
        1) Dark Channel feature: There are always some pixels in the image that contain
        a very low value in the 3 color channel components of RGB.
        2) These pixels form the dark channel of the image.
        3) Implemented min(R,G,B) + local minimum filter
        """
        dark_channel = np.minimum(np.minimum(self.red, self.green), self.blue)
        
        # Local minimum filter (structuring element)
        kernel = np.ones((window_size, window_size), np.uint8)
        dark_channel = cv2.erode(dark_channel, kernel).astype(np.float32)

        return dark_channel[..., None]   # (H, W, 1)
    
    
    def whitness_index_feature(self):
        """ 
        Clouds have relatively flat reflectivity in the RGB, NIR bands and usually display as white. 
        """
        eps = 1e-6
        mean = (self.red + self.green + self.blue) / 3.0

        whiteness = np.zeros_like(mean, dtype=np.float32)

        v = self.valid
        whiteness[v] = (
            np.abs(self.red[v] - mean[v]) +
            np.abs(self.green[v] - mean[v]) +
            np.abs(self.blue[v] - mean[v])
        ) / (mean[v] + eps)

        return whiteness[..., None] # (H, W, 1)

        
    def cloud_index_feature(self):
        """
        1) Cloud index is used to measure the similarity of reflectivit between RGB nad NIR bands.
        2) Mostly reflectivity in RGb and NIR of clouds is same, so CI is around 1.
        """
        eps = 1e-6
        den = self.red + self.green + self.blue

        CI = np.zeros_like(den, dtype=np.float32)

        v = self.valid
        CI[v] = (3.0 * self.nir[v]) / (den[v] + eps)

        CIF = np.abs(CI - 1.0)

        return CIF[..., None]

    
    def gabor_transform_feature(self):
        """
        Gabor Transform: RFCD Gabor texture feature
        Uses scale=7, angle=45° on Intensity map
        """

        # Intensity
        Intensity = (self.red + self.green + self.blue) / 3.0

        # RFCD-selected parameters
        theta = np.deg2rad(45)
        frequency = 1.0 / 7.0   # scale=7 → wavelength

        gabor_real, _ = gabor(Intensity, frequency=frequency, theta=theta)

        # Normalize
        gabor_real = (gabor_real - gabor_real.min()) / (gabor_real.max() - gabor_real.min() + 1e-6)

        return gabor_real[..., None]
    
    def get_all_features(self):
        f1 = self.band_feature()
        f2 = self.color_feature()
        f3 = self.dark_channel_feature()
        f4 = self.whitness_index_feature()
        f5 = self.cloud_index_feature()
        f6 = self.gabor_transform_feature()

        return np.concatenate([f1, f2, f3, f4, f5, f6], axis=-1)
    
    def predict(self):
        if self.model is None:
            raise ValueError("Model not loaded")

        features = self.get_all_features()          # (H,W,11)
        H, W, C = features.shape

        X = features.reshape(-1, C)
        mask = self.valid.reshape(-1)

        y = np.zeros(H*W, dtype=np.uint8)

        # Predict only valid pixels
        y[mask] = self.model.predict(X[mask])

        self.output_img = y.reshape(H, W)
        return self.output_img

    def predict_proba_map(self):
        feats = self.get_all_features()        # (H,W,11)
        H, W, C = feats.shape

        X = feats.reshape(-1, C)

        probs = self.model.predict_proba(X)[:, 1]   # cloud probability
        prob_map = probs.reshape(H, W)

        prob_map[~self.valid] = 0
        return prob_map

    def raw_dark_channel(self):
        return np.minimum(np.minimum(self.red, self.green), self.blue)
    
    # ---------------------------------------------------------
    # GUIDED FILTER (He et al. 2013)
    # ---------------------------------------------------------
    def guided_filter(self, I, p, r=30, eps=0.09):
        I = I.astype(np.float32)
        p = p.astype(np.float32)

        mean_I  = cv2.boxFilter(I, -1, (r,r))
        mean_p  = cv2.boxFilter(p, -1, (r,r))
        mean_Ip = cv2.boxFilter(I*p, -1, (r,r))

        cov_Ip = mean_Ip - mean_I*mean_p

        mean_II = cv2.boxFilter(I*I, -1, (r,r))
        var_I = mean_II - mean_I*mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a*mean_I

        mean_a = cv2.boxFilter(a, -1, (r,r))
        mean_b = cv2.boxFilter(b, -1, (r,r))

        q = mean_a*I + mean_b
        return q

    # ---------------------------------------------------------
    # RFCD FINAL OUTPUT
    # ---------------------------------------------------------
    def rfcd_refined(self):
        rf_prob = self.predict_proba_map()

        dark = self.raw_dark_channel()
        dark = (dark-dark.min())/(dark.max()-dark.min()+1e-6)

        refined = self.guided_filter(dark, rf_prob, 30, 0.09)

        refined_255 = np.clip(refined*255,0,255).astype(np.uint8)

        binary = (refined_255 >= 80).astype(np.uint8)
        binary[~self.valid]=0

        return refined_255, binary