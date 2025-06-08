from math import e
import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from PIL import Image
import h5py
from tqdm import tqdm

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

    def _load_images(self, paths: List[str], base_dir: Optional[str] = None) -> np.ndarray:
        images = []
        dataset_type = "treino" if base_dir == self.train_data_path else "validação" if base_dir == self.validation_data_path else "teste"
        for path in tqdm(paths, desc=f"[INFO] Processing images: {dataset_type}"):
            img_path = os.path.join(base_dir, path) if base_dir else path
            img = Image.open(img_path).convert('RGB')
            img = img.resize(self.height_width)
            images.append(np.array(img))
        return np.stack(images)

    def generate_hdf5(self, output_path: str, train_img_dir=None, val_img_dir=None, test_img_dir=None):
        # Use the provided directory or default to the data path attributes
        train_img_dir = train_img_dir if train_img_dir is not None else self.train_data_path
        val_img_dir = val_img_dir if val_img_dir is not None else self.validation_data_path
        test_img_dir = test_img_dir if test_img_dir is not None else self.test_data_path

        train_data = self._load_images(self.train_df['image'].tolist(), train_img_dir)
        val_data = self._load_images(self.validation_df['image'].tolist(), val_img_dir)
        test_data = self._load_images(self.test_df['image'].tolist(), test_img_dir)

        with h5py.File(output_path, 'w') as f:
            f.create_dataset('train_data', data=train_data, compression="gzip")
            f.create_dataset('train_label', data=np.array(self.train_df['MEL'].tolist()), compression="gzip")
            f.create_dataset('validation_data', data=val_data, compression="gzip")
            f.create_dataset('validation_label', data=np.array(self.validation_df['MEL'].tolist()), compression="gzip")
            f.create_dataset('test_data', data=test_data, compression="gzip")
            f.create_dataset('test_label', data=np.array(self.test_df['MEL'].tolist()), compression="gzip")
