import numpy as np
import numpy.random as npr
import pymc3 as pm
import theano.tensor as tt
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import idct
import pickle

class Horseshoe_img_comp():

    def __init__(self, block_size):
        self.block_size = block_size
        self.intensity_basis = None
        
    @staticmethod
    def true_idct2D(img, t=2, norm="ortho"):
        """
        2D inverse DCT from scipy IDCT
        """
        return 128 + 2.5 * idct(idct(img, axis=0, type=t, norm=norm), axis=1, type=t, norm=norm)
        
        
    def display_intensity_basis(self):
        """
        Computing and displaying the intensity basis as 25 5 x 5 blocks
        """
        fig, ax = plt.subplots(self.block_size, self.block_size)
        fig.set_size_inches(8, 8)

        for i in range(self.block_size):
            for j in range(self.block_size):
                ## Computing elementary DCT
                A_DCT = np.zeros((self.block_size, self.block_size))
                A_DCT[i, j] = 255
                ## Coming back to intensity domain with an inverse transform
                A_intensity = self.true_idct2D(A_DCT)
                ## Showing the result
                ax[i, j].imshow(A_intensity, aspect='equal', cmap='gray', vmin=0, vmax=255)
                ax[i, j].set_axis_off()
        plt.show()
    
    
    def compute_basis_intensity_matrix(self):
        """
        Computing the intensity basis as a 25 x 25 matrix. Each line of this matrix is 
        a flattened basis block.
        """
        intensity_basis = np.zeros((self.block_size ** 2, self.block_size ** 2), dtype="int_")

        for i in range(self.block_size):
            for j in range(self.block_size):
                ## Computing elementary DCT
                elementary_DCT = np.zeros((self.block_size, self.block_size))
                elementary_DCT[j, i] = 255
                ## Coming back to intensity domain with an inverse transform
                elementary_intensity = self.true_idct2D(elementary_DCT)
                ## Saving the result in the matrix
                intensity_basis[i * self.block_size + j,:] = elementary_intensity.flatten()
        
        self.intensity_basis = intensity_basis
        return self.intensity_basis
        
        
    
    def get_mcmc_sample_img_comp(self, X, y, sigma_block, samples=300):
        """
        This should return a pymc3 Trace object
        Computes the Horseshoe method via MCMC sampling.
        """
        
        regression = pm.Model()
        
        with regression:
            dim = X.shape[1]
            sigma = 1 + sigma_block * 0.3 #20 #pm.HalfFlat("sigma")
            
            # Defining the horseshoe prior
            lmbda_vect = pm.HalfCauchy("lambda", 1, shape=dim)
            #tau = pm.HalfCauchy("tau", 1)
            tau = pm.HalfNormal('tau', sigma=1)
            beta_vect = pm.Normal("Beta", mu=0, sigma=tt.dot(tau, lmbda_vect), shape=dim)
            
            # Defining the likelihood
            likelihood = pm.Normal('likelihood', mu=tt.dot(X, beta_vect), sigma=sigma, observed = y)
            
            
            trace = pm.sample(samples, target_accept=.9)
            
        return trace
    
    
    def compress_list_of_blocks(self, list_of_blocks, line_nb, nb_of_non_zero_coefs):
        """
        To compress an image line by line. 
        list_of_blocks: list of the blocks of a line of the image to compress.
        line_nb: index of the line (to name the save file).
        nb_of_non_zero_coefs: desired number of coefs to keep.
        """
        X = self.intensity_basis[1:]     # The mean component is dealt with apart
        compressed_blocks_line = []
        coefs_save = []

        for block in list_of_blocks:
            y = block.flatten()
            avg_block = np.mean(y)       # Mean component
            sigma_block = np.std(block)  # The choice of sigma in the Horseshoe is based on the standard deviation of the block
            trace = self.get_mcmc_sample_img_comp(X.T, y - avg_block, sigma_block)
            coefs_hs = np.mean(trace.get_values("Beta"), axis=0)
            coefs_save.append(coefs_hs.copy())
            thres = np.sort(np.abs(coefs_hs))[-nb_of_non_zero_coefs] # Keep only the greatest coefficients (in abs. value).
            coefs_hs[np.abs(coefs_hs) < thres] = 0                   # Putting the rest to 0.
            compressed_blocks_line.append((coefs_hs @ X).reshape(self.block_size, self.block_size) + avg_block)
        
        ### Saving result in pickle file
        print("\nSaving...")
        with open(f"saves/compressed_blocks_line{line_nb}.pckl", "wb") as save_file:
            pckl = pickle.Pickler(save_file)
            pckl.dump(compressed_blocks_line)
        with open(f"saves/coefs_line{line_nb}.pckl", "wb") as save_file:
            # Remark: the coefs saved are not thresholded.
            pckl = pickle.Pickler(save_file)
            pckl.dump(coefs_save)
        print("Done")

        return compressed_blocks_line, coefs_save


if __name__ == '__main__':
    print("Hello Mr. Bardenet!")