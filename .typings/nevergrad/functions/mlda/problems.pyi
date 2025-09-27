import nevergrad.common.typing as tp
import numpy as np
from . import datasets as datasets
from ..base import ExperimentFunction as ExperimentFunction
from _typeshed import Incomplete

def _kmeans_distance(points: np.ndarray, centers: np.ndarray) -> float:
    """Computes the distance between points and centers
    after affecting each points to the closest center.
    """

class Clustering(ExperimentFunction):
    """Cost function of a clustering problem.

    Parameters
    ----------
    points: np.ndarray
        k x n array where k is the number of points and n their coordinates
    num_clusters: int
        number of clusters to find
    """
    num_clusters: Incomplete
    _points: Incomplete
    def __init__(self, points: np.ndarray, num_clusters: int, rescale: bool = True) -> None: ...
    @classmethod
    def from_mlda(cls, name: str, num_clusters: int, rescale: bool = True) -> Clustering:
        '''Clustering problem defined in
        Machine Learning and Data Analysis (MLDA) Problem Set v1.0, Gallagher et al., 2018, PPSN
        https://drive.google.com/file/d/1fc1sVwoLJ0LsQ5fzi4jo3rDJHQ6VGQ1h/view

        Parameters
        ----------
        name: str
            name of the dataset ("Ruspini", or "German towns")
        num_clusters: int
            number of clusters to find

        Note
        ----
        The MLDA problems are P1.a Clustering("Ruspini", 5) and  P1.b Clustering("German towns", 10)
        '''
    def _compute_distance(self, centers: np.ndarray) -> float:
        """Sum of minimum squared distances to closest centroid
        centers must be of size num_clusters x n
        """

class Perceptron(ExperimentFunction):
    """Perceptron function

    Parameters
    ----------
    x: np.ndarray
        the input data
    y: np.ndarray
        the data to predict from the input data
    """
    _x: Incomplete
    _y: Incomplete
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None: ...
    @classmethod
    def from_mlda(cls, name: str) -> Perceptron:
        '''Perceptron problem defined in
        Machine Learning and Data Analysis (MLDA) Problem Set v1.0, Gallagher et al., 2018, PPSN
        https://drive.google.com/file/d/1fc1sVwoLJ0LsQ5fzi4jo3rDJHQ6VGQ1h/view

        Parameters
        ----------
        name: str
            name of the dataset (among "quadratic", "sine", "abs" and "heaviside")

        Note
        ----
        quadratic is coined P2.a, sine P2.b, abs P2.c and heaviside P2.d
        '''
    def apply(self, parameters: tp.ArrayLike) -> np.ndarray:
        """Apply the perceptron transform to x using the provided parameters

        Parameters
        ----------
        parameters: ArrayLike
            parameters of the perceptron

        Returns
        -------
        np.ndarray
            transformed data
        """
    def _compute_loss(self, x: tp.ArrayLike) -> float:
        """Compute perceptron"""

class SammonMapping(ExperimentFunction):
    """Sammon mapping function"""
    _proximity: Incomplete
    _proximity_2: Incomplete
    def __init__(self, proximity_array: np.ndarray) -> None: ...
    @classmethod
    def from_mlda(cls, name: str, rescale: bool = False) -> SammonMapping:
        '''Mapping problem defined in
        Machine Learning and Data Analysis (MLDA) Problem Set v1.0, Gallagher et al., 2018, PPSN
        https://drive.google.com/file/d/1fc1sVwoLJ0LsQ5fzi4jo3rDJHQ6VGQ1h/view

        Parameters
        ----------
        name: str
            name of the dataset (initially among "Virus", "Employees", but Employees dataset is not
            available online anymore)

        Notes
        -----
        - "Virus" dataset is P3.a and "Employees" dataset is P3.b
        - for "Employees", we use the online proximity matrix
        - for "Virus", we compute a proximity matrix from raw data (no normalization)
        '''
    @classmethod
    def from_2d_circle(cls, num_points: int = 12) -> SammonMapping:
        """Simple test case where the points are in a 2d circle."""
    def _compute_distance(self, x: np.ndarray) -> float:
        """Compute the Sammon mapping metric for the input data"""

class Landscape(ExperimentFunction):
    '''Planet Earth Landscape problem defined in
    Machine Learning and Data Analysis (MLDA) Problem Set v1.0, Gallagher et al., 2018, PPSN
    https://drive.google.com/file/d/1fc1sVwoLJ0LsQ5fzi4jo3rDJHQ6VGQ1h/view
    and reverted to look for the minimum value (0). This is problem P4.

    Parameters
    ----------
    transform: None, "gaussian" or "square"
        whether use the image [0, 4319]x[0, 2159] (None) or to use a Gaussian transform.

    Note
    ----
    - the initial image is 4320x2160
    - sampling outside yields a +inf value (except for Gaussian, since large values are mapped to border indices)
    - the image is actually a variant of the one proposed in the article. Indeed, this image
      has a much better z-resolution. It is not directly proportional to the altitude though
      since it is an artificial rescaling to greyscale of a color image.
    '''
    _image: Incomplete
    parametrization: Incomplete
    _max: Incomplete
    def __init__(self, transform: tp.Optional[str] = None) -> None: ...
    def _get_pixel_value(self, x: float, y: float) -> float: ...
