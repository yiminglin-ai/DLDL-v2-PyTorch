from typing import Any, Dict
import os
import lmdb
import numpy as np
import cv2
import pyarrow as pa
import math
import albumentations as A
import ibug.roi_tanh_warping.reference_impl as ref

from data import *

class AgeLMDB:
    def __init__(
        self,
        db_path,
        transforms=None,
        **kwargs
    ) -> None:
        self.db_path = db_path
        # https://github.com/chainer/chainermn/issues/129
        # Delay loading LMDB data until after initialization to avoid "can't pickle Environment Object error"
        self._init_db()
        self.env = None
        self.transforms = transforms
        # Workaround to have length from the start for ImageNet since we don't have LMDB at initialization time

    def _init_db(self):
        self.env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pa.deserialize(txn.get(b'__len__'))
            self.keys = pa.deserialize(txn.get(b'__keys__'))

    def return_ages(self):
        from tqdm import tqdm
        if self.env is None:
            self._init_db()
        ages = []
        with self.env.begin(write=False) as txn:
            for k in tqdm(self.keys):
                byteflow = txn.get(k)
                unpacked = pa.deserialize(byteflow)
                (_, _, age, _, _, _) = unpacked
                age = float(age) if age < 101 else 100.
                ages.append(int(age))
        self.env = None
        return ages

    def __getitem__(self, index):

        # Delay loading LMDB data until after initialization: https://github.com/chainer/chainermn/issues/129
        if self.env is None:
            self._init_db()
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pa.deserialize(byteflow)
        (name, image, age, gender, bbox, landmark) = unpacked
        # print(name, bbox, age, gender)
        try:
            bbox = np.frombuffer(bbox, int)
            landmark = np.frombuffer(landmark, int).reshape(-1, 2)
        except:
            bbox = np.asarray(bbox, int)
            landmark = np.asarray(landmark, int).reshape(-1, 2)

        image = cv2.imdecode(image, cv2.IMREAD_COLOR)[:, :, ::-1]
        if age < 0:
            age = 0.
        age = float(age) if age < 101 else 100.
        dist = [normal_sampling(int(age), i) for i in range(101)]
        dist = np.array([i if i > 1e-15 else 1e-15 for i in dist])
        if self.transforms is not None:
            image = self.transforms(
                image=image, bboxes=[bbox], category_ids=[age])['image']
        return dict(
            image=image,
            bbox=bbox,
            age=age,
            label=dist
        )

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'
