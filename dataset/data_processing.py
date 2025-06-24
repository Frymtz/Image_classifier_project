from math import e
import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from PIL import Image
import h5py
from tqdm import tqdm
from utils import ImageFeatureExtractor
from utils.augmentation import Augmentation


class ImageDatasetGenerator:
    def __init__(
        self,
        train_data_path, train_label_path, train_percent,
        validation_data_path, validation_label_path, validation_percent,
        test_data_path, test_label_path, test_percent,
        height_width, extraction_technique, extension
    ):
        self.train_data_path = train_data_path
        self.train_label_path = train_label_path
        self.train_percent = train_percent
        self.validation_data_path = validation_data_path
        self.validation_label_path = validation_label_path
        self.validation_percent = validation_percent
        self.test_data_path = test_data_path
        self.test_label_path = test_label_path
        self.test_percent = test_percent
        self.height_width = height_width
        self.extraction_technique = extraction_technique
        self.extension = extension

        # Load and process the datasets
        self.train_df = self._load_and_process_csv(self.train_label_path, self.train_percent,self.extension)
        self.validation_df = self._load_and_process_csv(self.validation_label_path, self.validation_percent, self.extension)
        self.test_df = self._load_and_process_csv(self.test_label_path, self.test_percent, self.extension)

    def _load_and_process_csv(self, csv_path, percent=100.0, extension='.png') -> pd.DataFrame:
        """
        Load the CSV, keep only the 'image' and 'MEL' columns,
        add '.png' to the image name, and keep only the given percent of rows.
        """
        df = pd.read_csv(csv_path)
        df = df[['image', 'MEL']]
        df['image'] = df['image'].astype(str) + extension  
        # Seleciona apenas a porcentagem desejada
        n = int(len(df) * percent/100)
        df = df.iloc[:n].reset_index(drop=True)
        return df

    def _load_images_processImages(self, paths: List[str], base_dir: Optional[str] = None, mask_list=None, perimeter_list=None) -> np.ndarray:
        images = []
        extractor = ImageFeatureExtractor(self.height_width)
        dataset_type = "Train" if base_dir == self.train_data_path else "Validation" if base_dir == self.validation_data_path else "Test"
        
        extraction_technique = None if not self.extraction_technique else self.extraction_technique[0].lower()
        for idx, path in enumerate(tqdm(paths, desc=f"[INFO] Processing images: {dataset_type}")):
            img_path = os.path.join(base_dir, path) if base_dir else path
            img = Image.open(img_path).convert('RGB')
            # Pegue a máscara e perímetro se necessário
            mask = mask_list[idx] if mask_list is not None else None
            perimeter = perimeter_list[idx] if perimeter_list is not None else None
            features = extractor.extract(
                img,
                extraction_technique,
                mask=mask,
                perimeter=perimeter
            )
            images.append(features)
        return np.stack(images)

    def _load_images_raw(self, paths: List[str], base_dir: Optional[str] = None) -> np.ndarray:
        images = []
        for path in tqdm(paths, desc="[INFO] Loading raw images"):
            img_path = os.path.join(base_dir, path) if base_dir else path
            img = Image.open(img_path)
            if self.height_width:  # height_width deve ser uma tupla (width, height)
                img = img.resize(self.height_width)
            images.append(np.array(img))
        return np.stack(images)

    def generate_hdf5(self, output_path: str, create_hdf5: bool, train_img_dir=None, val_img_dir=None, test_img_dir=None):
        train_img_dir = train_img_dir if train_img_dir is not None else self.train_data_path
        val_img_dir = val_img_dir if val_img_dir is not None else self.validation_data_path
        test_img_dir = test_img_dir if test_img_dir is not None else self.test_data_path

        
        X_train_raw = self._load_images_raw(self.train_df['image'].tolist(), train_img_dir)
        y_train = np.array(self.train_df['MEL'].tolist())

        augmenter = Augmentation(random_state=42)
        X_train_bal, y_train_bal = augmenter.balance_oversample(X_train_raw, y_train)

        extractor = ImageFeatureExtractor(self.height_width)
        extraction_technique = None if not self.extraction_technique else self.extraction_technique[0].lower()
        train_data = []
        for img in tqdm(X_train_bal, desc="[INFO] Extracting features from balanced train set"):
            features = extractor.extract(Image.fromarray(img), extraction_technique)
            train_data.append(features)
        train_data = np.stack(train_data)

        val_data = self._load_images_processImages(self.validation_df['image'].tolist(), val_img_dir)
        test_data = self._load_images_processImages(self.test_df['image'].tolist(), test_img_dir)
        val_labels = np.array(self.validation_df['MEL'].tolist())
        test_labels = np.array(self.test_df['MEL'].tolist())

        if not create_hdf5:
            return train_data, val_data, test_data, y_train_bal, val_labels, test_labels

        with h5py.File(output_path, 'w') as f:
            f.create_dataset('train_data', data=train_data, compression="gzip")
            f.create_dataset('train_label', data=y_train_bal, compression="gzip")
            f.create_dataset('validation_data', data=val_data, compression="gzip")
            f.create_dataset('validation_label', data=np.array(self.validation_df['MEL'].tolist()), compression="gzip")
            f.create_dataset('test_data', data=test_data, compression="gzip")
            f.create_dataset('test_label', data=np.array(self.test_df['MEL'].tolist()), compression="gzip")
    
    def load_hdf5(self, hdf5_path: str):
        """
        Load datasets from an HDF5 file.
        Returns a dictionary with keys:
        'train_data', 'train_label', 'validation_data', 'validation_label', 'test_data', 'test_label'
        """
        data = {}
        with h5py.File(hdf5_path, 'r') as f:
            data['train_data'] = f['train_data'][:]
            data['train_label'] = f['train_label'][:]
            data['validation_data'] = f['validation_data'][:]
            data['validation_label'] = f['validation_label'][:]
            data['test_data'] = f['test_data'][:]
            data['test_label'] = f['test_label'][:]
        return data