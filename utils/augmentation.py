import numpy as np
from collections import Counter
from PIL import Image, ImageOps, ImageEnhance
import random

class Augmentation:
    def __init__(self, random_state=None):
        self.random_state = random_state
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

    def augment_image(self, img):
        """Applies a random transformation: flip, rotation, noise, or brightness."""
        img = Image.fromarray(img)
        ops = [
            lambda x: x,
            ImageOps.mirror,
            ImageOps.flip,
            lambda x: x.rotate(90),
            lambda x: x.rotate(180),
            lambda x: x.rotate(270),
            self.add_noise,
            self.change_brightness
        ]
        op = random.choice(ops)
        # If the operation is noise or brightness, handle differently
        if op == self.add_noise or op == self.change_brightness:
            return op(np.array(img))
        else:
            return np.array(op(img))

    def add_noise(self, img):
        """Adds Gaussian noise to the image."""
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        noise = np.random.normal(0, 10, img.shape)
        img_noisy = img + noise
        img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)
        return img_noisy

    def change_brightness(self, img):
        """Changes the brightness of the image."""
        img_pil = Image.fromarray(img)
        enhancer = ImageEnhance.Brightness(img_pil)
        factor = random.uniform(0.6, 1.4)  # brightness between 60% and 140%
        img_bright = enhancer.enhance(factor)
        return np.array(img_bright)

    def balance_oversample(self, X, y):
        """
        Balances the classes by oversampling with augmentation.
        X: array (N, H, W, C) or (N, H, W)
        y: array (N,)
        Returns: X_bal, y_bal
        """
        counter = Counter(y)
        max_count = max(counter.values())
        classes = np.unique(y)
        X_bal = []
        y_bal = []
        for cls in classes:
            idx = np.where(y == cls)[0]
            X_cls = X[idx]
            n_to_add = max_count - len(idx)
            X_aug = []
            if n_to_add > 0:
                for _ in range(n_to_add):
                    i = np.random.choice(len(X_cls))
                    img_aug = self.augment_image(X_cls[i])
                    X_aug.append(img_aug)
            X_bal.append(np.concatenate([X_cls, np.array(X_aug)]) if X_aug else X_cls)
            y_bal.append(np.full(max_count, cls))
        X_bal = np.concatenate(X_bal, axis=0)
        y_bal = np.concatenate(y_bal, axis=0)
        # Shuffle
        perm = np.random.permutation(len(y_bal))
        return X_bal[perm], y_bal[perm]