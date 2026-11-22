import numpy as np
import torch


class Dataset(torch.utils.data.Dataset):
    """Dataloder.

    Args_
        torch (_type_): _description_

    """

    def __init__(self, adata = [], groups = []):
        """_summary.

        Args:
            group (str, optional): _description_. Defaults to 'train'.
            cell_type (str, optional): _description_. Defaults to 'CD4T'.

        """
        self.data = torch.tensor(adata.X.toarray())
        self.resample_data = self.data
        self.w = np.zeros(shape=(self.data.shape[0], 1))
        self.groups = groups


    def update_training_sample_probabilities(self, z, bins=100, smoothing_fac=0.0001):
        """Updates the sampling probabilities for cells within a batch based on the maximum of
        latent variables' probablities.
        Adapted from MIT 6.S191 Lab 2 (DB-VAE), MIT License,
        © MIT Introduction to Deep Learning, http://introtodeeplearning.com
        (https://github.com/MITDeepLearning/introtodeeplearning/commit/9e6df30),
        via https://goodboychan.github.io/python/tensorflow/mit/2021/02/27/Debiasing.html
        
        Args:
            z (torch.Tensor): Latent variables.
            bins (int, optional): Number of bins to use for histogram. Defaults to 100.
            smoothing_fac (float, optional): Smoothing factor for histogram. Defaults to 0.0001."""

        training_sample_p = np.zeros(z.shape[0])
        
        # consider the distribution for each latent variable 
        for i in range(z.shape[1]):

            latent_distribution = z[:,i]
            # generate a histogram of the latent distribution
            hist_density, bin_edges =  np.histogram(latent_distribution, density=True, bins=bins)

            # find which latent bin every data sample falls in 
            bin_edges[0] = -float('inf')
            bin_edges[-1] = float('inf')

            # call the digitize function to find which bins in the latent distribution 
            #    every data sample falls in to
            # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.digitize.html
            bin_idx = np.digitize(latent_distribution, bin_edges) 

            # smooth the density function
            hist_smoothed_density = hist_density + smoothing_fac
            hist_smoothed_density = hist_smoothed_density / np.sum(hist_smoothed_density)

            # invert the density function 
            p = 1.0/(hist_smoothed_density[bin_idx-1])

            # normalize all probabilities
            p = p / np.sum(p)

            # update sampling probabilities by considering whether the newly
            # computed p is greater than the existing sampling probabilities.
            training_sample_p = np.maximum(p, training_sample_p)

        # final normalization
        self.w = training_sample_p/np.sum(training_sample_p)

        return


    def set_uniform_probabilities(self):
        """Set uniform sampling probabilities (w = 1/N). Used by the
        uniform-resampling control arm, which shares the AR code path but
        draws cells without density-based weights."""

        self.w = np.full(self.data.shape[0], 1.0 / self.data.shape[0])

        return


    def resample_epoch_indices(self, rng):
        """Draw the resampled training set for one epoch (Algorithm 1, step 16):
        N indices sampled with replacement according to the current weights.

        Args:
            rng (np.random.Generator): generator seeded once per run

        Returns:
            np.ndarray: N resampled indices, to be split into minibatches"""

        return rng.choice(
            self.data.shape[0],
            size=self.data.shape[0],
            replace=True,
            p=self.w)


    def __len__(self):
        """Denotes the total number of samples.

        Returns:
            int: Shape of the sample data passed to the class.

        """
        return self.data.shape[0]


    def __getitem__(self, idx):
        """Generate one sample of data.

        Args:
            index (int): index of the item to be returned.

        Returns:
            torch.Tensor: Tensor of the resampled data.

        """
        data_dict = {}
        data_dict['X'] = self.data[idx]
        return data_dict