import numpy as np
import os
from tqdm import tqdm

from rfcd_l8_dataloader import L8RFCDDataLoader
from random_forest_cloud_detection import RandomForestCloudDetection  


class RFCDDatasetBuilder:
    """
    Builds RFCD per-pixel training dataset from Landsat-8 Biome scenes
    """

    def __init__(self, scene_dirs, samples_per_scene=15000):
        self.scene_dirs = scene_dirs
        self.samples_per_scene = samples_per_scene

        self.X = []
        self.y = []

    # --------------------------------------------------
    # Sample balanced pixels from one scene
    # --------------------------------------------------
    def _sample_scene(self, features, cloud, valid):
        H, W, C = features.shape

        cloud_idx = np.where((cloud == 1) & valid)
        clear_idx = np.where((cloud == 0) & valid)

        if len(cloud_idx[0]) == 0 or len(clear_idx[0]) == 0:
            return None, None

        n = self.samples_per_scene // 2
        n_cloud = min(n, len(cloud_idx[0]))
        n_clear = min(n, len(clear_idx[0]))

        cloud_sel = np.random.choice(len(cloud_idx[0]), n_cloud, replace=False)
        clear_sel = np.random.choice(len(clear_idx[0]), n_clear, replace=False)

        X_cloud = features[cloud_idx[0][cloud_sel], cloud_idx[1][cloud_sel]]
        X_clear = features[clear_idx[0][clear_sel], clear_idx[1][clear_sel]]

        y_cloud = np.ones(n_cloud)
        y_clear = np.zeros(n_clear)

        X = np.vstack([X_cloud, X_clear])
        y = np.hstack([y_cloud, y_clear])

        return X, y

    # --------------------------------------------------
    # Process all scenes
    # --------------------------------------------------
    def build(self):
        for scene in tqdm(self.scene_dirs):
            try:
                loader = L8RFCDDataLoader(scene)
                blue, green, red, nir, cloud, valid = loader.load()
                
                img = np.stack([blue, green, red, nir], axis=0)  # (C, H, W)

                # Extract RFCD features
                extractor = RandomForestCloudDetection(img)
                features = extractor.get_all_features()   # (H,W,F)

                Xs, ys = self._sample_scene(features, cloud, valid)

                if Xs is None:
                    continue

                self.X.append(Xs)
                self.y.append(ys)

            except Exception as e:
                print("Skipping", scene, ":", e)

        self.X = np.vstack(self.X)
        self.y = np.hstack(self.y)

        print("Final dataset shape:", self.X.shape, self.y.shape)
        return self.X, self.y

    # --------------------------------------------------
    # Save dataset
    # --------------------------------------------------
    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "X_TOA_25k.npy"), self.X)
        np.save(os.path.join(out_dir, "y_TOA_25k.npy"), self.y)


if __name__ == "__main__":
    
    scene_dirs = [
        "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Training_L8_Data/Barren/LC80420082013220LGN00/BC/LC80420082013220LGN00",
        "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Training_L8_Data/Barren/LC80500092014231LGN00/BC/LC80500092014231LGN00",
        "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Training_L8_Data/Barren/LC81330312013202LGN00/BC/LC81330312013202LGN00",
        "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Training_L8_Data/Barren/LC81570452014213LGN00/BC/LC81570452014213LGN00",
        "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Training_L8_Data/Forest/LC80200462014005LGN00/BC/LC80200462014005LGN00",
        "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Training_L8_Data/Forest/LC81310182013108LGN01/BC/LC81310182013108LGN01",
        "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Training_L8_Data/Forest/LC82290572014141LGN00/BC/LC82290572014141LGN00",
        "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Training_L8_Data/Forest/LC82310592014139LGN00/BC/LC82310592014139LGN00",
        "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Training_L8_Data/Grass_Crops/LC81440462014250LGN00/BC/LC81440462014250LGN00",
        "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Training_L8_Data/Grass_Crops/LC81490432014141LGN00/BC/LC81490432014141LGN00",
        "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Training_L8_Data/Grass_Crops/LC81750512013208LGN00/BC/LC81750512013208LGN00",
        "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Training_L8_Data/Grass_Crops/LC82020522013141LGN01/BC/LC82020522013141LGN01",
        "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Training_L8_Data/Shrubland/LC80010732013109LGN00/BC/LC80010732013109LGN00",
        "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Training_L8_Data/Shrubland/LC80980762014216LGN00/BC/LC80980762014216LGN00",
        "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Training_L8_Data/Shrubland/LC81590362014051LGN00/BC/LC81590362014051LGN00",
        "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Training_L8_Data/Shrubland/LC81600462013215LGN00/BC/LC81600462013215LGN00",
        "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Training_L8_Data/Snow_Ice/LC80060102014147LGN00/BC/LC80060102014147LGN00",
        "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Training_L8_Data/Snow_Ice/LC80841202014309LGN00/BC/LC80841202014309LGN00",
        "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Training_L8_Data/Snow_Ice/LC82171112014297LGN00/BC/LC82171112014297LGN00",
        "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Training_L8_Data/Snow_Ice/LC82320072014226LGN00/BC/LC82320072014226LGN00",
        "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Training_L8_Data/Urban/LC80150312014226LGN00/BC/LC80150312014226LGN00",
        "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Training_L8_Data/Urban/LC81620432014072LGN00/BC/LC81620432014072LGN00",
        "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Training_L8_Data/Urban/LC81660432014020LGN00/BC/LC81660432014020LGN00",
        "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Training_L8_Data/Urban/LC81920192013103LGN01/BC/LC81920192013103LGN01",
        "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Training_L8_Data/Water/LC80120552013202LGN00/BC/LC80120552013202LGN00",
        "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Training_L8_Data/Water/LC80180082014215LGN00/BC/LC80180082014215LGN00",
        "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Training_L8_Data/Water/LC80210072014236LGN00/BC/LC80210072014236LGN00",
        "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Training_L8_Data/Water/LC80430122014214LGN00/BC/LC80430122014214LGN00",
        "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Training_L8_Data/Wetlands/LC80310202013223LGN00/BC/LC80310202013223LGN00",
        "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Training_L8_Data/Wetlands/LC81030162014107LGN00/BC/LC81030162014107LGN00",
        "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Training_L8_Data/Wetlands/LC81070152013260LGN00/BC/LC81070152013260LGN00",
        "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Training_L8_Data/Wetlands/LC81080182014238LGN00/BC/LC81080182014238LGN00",
    ]

    builder = RFCDDatasetBuilder(scene_dirs, samples_per_scene=25000)
    X, y = builder.build()

    builder.save("/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_pixel_training_data/25k_samples")
