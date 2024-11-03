import numpy as np


class build_transforms:
    """Class to transform the input data during training."""
    def __init__(self, preprocessor, transf_degree):
        self.mean = np.array(preprocessor['mean']).reshape(1, -1)
        self.std = np.array(preprocessor['std']).reshape(1, -1)
        self.transf_degree = transf_degree

    def __call__(self, feature):
        feature = (feature - self.mean) / self.std
        if self.transf_degree > 0:
            feature += self.transf_degree * 0.01 * np.random.randn(len(feature))
        return feature


class build_transforms_val:
    """Class to transform the input data for inference."""
    def __init__(self, preprocessor, transf_degree):
        self.mean = np.array(preprocessor['mean']).reshape(1, -1)
        self.std = np.array(preprocessor['std']).reshape(1, -1)
        self.transf_degree = transf_degree

    def __call__(self, feature):
        feature = (feature - self.mean) / self.std
        if self.transf_degree > 0:
            feature += self.transf_degree * 0.01 * np.random.randn(len(feature))
        return feature
