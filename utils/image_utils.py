import re
import numpy as np
from PIL import Image
import pyImageFeatures.pyfeats as pyfeats

class ImageFeatureExtractor:
    def __init__(self, resize_shape):
        self.resize_shape = resize_shape

    def extract(self, img: Image.Image, technique: str, **kwargs) -> np.ndarray:
        """
        Extracts features from the image using the specified technique.
        img: PIL.Image
        technique: name of the technique (string)
        kwargs: additional arguments for the extraction function
        """
        if self.resize_shape is not None:
            img = img.resize(self.resize_shape)
        
        img_np = np.array(img)
        if technique == None:
            return img_np       
        else:
            # Convert to grayscale if necessary
            if technique not in ["raw", "rgb"]:
                f = np.array(img.convert("L"))
            else:
                f = img_np

            # Mask and other arguments can be passed via kwargs
            mask = kwargs.get("mask", None)
            perimeter = kwargs.get("perimeter", None)
            # Dictionary of techniques
            if technique == "raw":
                return f
            elif technique == "fos":
                features, _ = pyfeats.fos(f, mask)
                return features
            elif technique == "glcm":
                features_mean, features_range, _, _ = pyfeats.glcm_features(f, ignore_zeros=True)
                return np.concatenate([features_mean, features_range])
            elif technique == "glds":
                features, _ = pyfeats.glds_features(f, mask, Dx=[0,1,1,1], Dy=[1,1,0,-1])
                return features
            elif technique == "ngtdm":
                features, _ = pyfeats.ngtdm_features(f, mask, d=1)
                return features
            elif technique == "sfm":
                features, _ = pyfeats.sfm_features(f, mask, Lr=4, Lc=4)
                return features
            elif technique == "lte":
                features, _ = pyfeats.lte_measures(f, mask, l=7)
                return features
            elif technique == "fdta":
                features, _ = pyfeats.fdta(f, mask, s=3)
                return features
            elif technique == "glrlm":
                features, _ = pyfeats.glrlm_features(f, mask, Ng=256)
                return features
            elif technique == "fps":
                features, _ = pyfeats.fps(f, mask)
                return features
            elif technique == "shape":
                features, _ = pyfeats.shape_parameters(f, mask, perimeter, pixels_per_mm2=1)
                return features
            elif technique == "glszm":
                features, _ = pyfeats.glszm_features(f, mask, connectivity=1)
                return features
            elif technique == "hos":
                features, _ = pyfeats.hos_features(f, th=[135,140])
                return features
            elif technique == "lbp":
                features, _ = pyfeats.lbp_features(f, mask, P=[8,16,24], R=[1,2,3])
                return features
            elif technique == "grayscale_morphology":
                pdf, cdf = pyfeats.grayscale_morphology_features(f, kwargs.get("N", 32))
                return np.concatenate([pdf, cdf])
            elif technique == "multilevel_binary_morphology":
                pdf_L, pdf_M, pdf_H, cdf_L, cdf_M, cdf_H = pyfeats.multilevel_binary_morphology_features(
                    f, mask, N=30, thresholds=[25, 50]
                )
                return np.concatenate([pdf_L, pdf_M, pdf_H, cdf_L, cdf_M, cdf_H])
            elif technique == "histogram":
                features, _ = pyfeats.histogram(f, mask, bins=32)
                return features
            elif technique == "multiregion_histogram":
                features, _ = pyfeats.multiregion_histogram(f, mask, kwargs.get("bins", 32), num_eros=3, square_size=3)
                return features
            elif technique == "correlogram":
                Hd, Ht, _ = pyfeats.correlogram(f, mask, bins_digitize=32, bins_hist=32, flatten=True)
                return np.concatenate([Hd, Ht])
            elif technique == "amfm":
                features, _ = pyfeats.amfm_features(f, bins=32)
                return features
            elif technique == "dwt":
                features, _ = pyfeats.dwt_features(f, mask, wavelet='bior3.3', levels=3)
                return features
            elif technique == "swt":
                features, _ = pyfeats.swt_features(f, mask, wavelet='bior3.3', levels=3)
                return features
            elif technique == "wp":
                features, _ = pyfeats.wp_features(f, mask, wavelet='cof1', maxlevel=3)
                return features
            elif technique == "gt":
                features, _ = pyfeats.gt_features(f, mask, deg=4, freq=[0.05, 0.4])
                return features
            elif technique == "zernikes":
                features, _ = pyfeats.zernikes_moments(f, radius=9)
                return features
            elif technique == "hu":
                features, _ = pyfeats.hu_moments(f)
                return features
            elif technique == "tas":
                features, _ = pyfeats.tas_features(f)
                return features
            elif technique == "hog":
                features, _ = pyfeats.hog_features(f, ppc=8, cpb=3)
                return features
            else:
                raise ValueError(f"Extraction technique '{technique}' is not supported.")