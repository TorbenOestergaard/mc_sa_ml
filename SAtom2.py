"""
TOM sensitivity analysis using Kolmogorov-Smirnov 2-sample test.

TOM performs global sensitivity analysis for a single or multiple
outputs by repeated random splits of the output(s) which are compared
using the maximum distances, K, between the cumulative distributions
of the initial and filtered sets.
"""

# Authors: Torben Østergård <torbeniha@gmail.com>
# Version: 1.2.1
#
# License: MIT License

import timeit
import pandas as pd
import numpy as np
import math


class TOM:
    """
    TOM sensitivity analysis using Kolmogorov-Smirnov 2-sample test.
    
    TOM performs global sensitivity analysis for a single or multiple
    outputs by repeated random splits of the output(s) which are compared
    using the maximum distances, K, between the cumulative distributions
    of the initial and filtered sets.
    
    Parameters
    ----------
    X : pandas.DataFrame
        Input/feature matrix with values for the independent variables.
    
    Y : pandas.DataFrame
        Output/target matrix with dependent variables.
    
    J : int, default=100
        Number of repeated randomly selected samples. 
        
    Attributes:
    -----------
    K_ : array of shape(J, n_features)
        K distances for all J samples and all features (inputs).
        
    References
    ----------
    Østergård, T., Jensen, R.L., and Maagaard, S.E. (2017)
        Interactive Building Design Space Exploration Using Regionalized 
        Sensitivity Analysis, 15th conference of the International Building 
        Performance Simulation Association, 7-9 August 2017, San Francisco, USA
    
    Examples
    --------
    
    """
    def __init__(self, X, Y, J=100, dummy=True, verbose=True):
        self.X = X.copy()
        self.Y = Y.copy()
        if dummy:
            self.X['Dummy'] = np.random.permutation(self.X.shape[0])
#             self.X = self.add_dummy()
        self.J = J
        self.verbose = verbose
        self._validate_arguments()
        self.K = self.perform_SA(self.X, self.Y, J, verbose)

    def _validate_arguments(self):
        """
        Validation of class arguments.
        
        If a single output is passed as pandas.Series it is converted to DataFrame.
        
        Output columns with no variance will be removed.
        """        
        try:
            self.X = self.X.astype('float64')
        except:
            raise Exception('X contains non-numeric values.') from None
        try:
            self.Y = self.Y.astype('float64')
        except:
            raise Exception('Y contains non-numeric values.') from None         
        
        if isinstance(self.Y, pd.Series):
            self.Y = self.Y.to_frame()

        for col in self.Y.columns:
            if self.is_unique(self.Y[col]):
                self.Y.drop(columns=col, inplace=True)
                print(f'Removed the output {col} which have equal values and thus no variance.')
        
    def add_dummy(self):
        """
        Adds a dummy variable with random integers.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features) input data

        Returns
        -------
        X : including the "Dummy" column with random integers
        """
        
#         self.X['Dummy'] = np.random.permutation(self.X.shape[0])
#         return X
    
    def perform_SA(self, X, Y, J, verbose):
        """
        Performs TOM sensitivity analysis.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features) input data

        y : pandas.DataFrame of shape (n_samples, n_targets) output data
        
        J : int, number of randomly selected samples

        Returns
        -------
        self : returns an instance of self.
        """
        
        starttime = timeit.default_timer()

        N = X.shape[0]
        n_inputs = X.shape[1]
        n_outputs = Y.values.ndim if Y.values.ndim == 1 else Y.shape[1]
        Q_size = math.floor(0.5 ** (1 / n_outputs) * N) # Size of random samples, see research paper

        Xarr = X.values
        Yarr = Y.values.reshape(N, n_outputs)

        arrA = np.array(list(range(N))).reshape(N, 1)

        Ytemp1 = np.array([], dtype=np.int64).reshape(N, 0)
        YtempA = np.array([], dtype=np.int64).reshape(N, 0)

        rep = 0
        halton = np.array(TOM.halton_sequence(J, n_outputs)) * N
        KS_j = np.zeros(shape=(1, n_inputs))
        KS = np.zeros(shape=(J, n_inputs))
        KS_means = np.zeros(shape=(J, n_inputs))

        if verbose:
            print(f'{"N, simulations:" :>20}  {N}')
            print(f'{"N, inputs:" :>20}  {n_inputs}')
            print(f'{"N, outputs:" :>20}  {n_outputs}')
            print(f'{"J:" :>20}  {J}')
            print(f'{"Q, size:" :>20}  {Q_size}')

        # Create array with simulation indices ranked for each output
        for output in range(n_outputs):
            arr_temp = Yarr[:, output].reshape(len(arrA), 1)  # Turn ith output vector to 2D numpy array
            arr = np.concatenate((arr_temp, arrA), axis=1)  # Add simulation index vector as second column
            arr = arr[
                arr[:, 0].argsort()]  # Sort by ith values # arr = np.sort(arr, axis=0, kind='stable') # alternative
            arrSorted = arr[:, 1].astype(int)
            arrSorted = arrSorted.reshape(N, 1)
            YtempA = np.concatenate((YtempA, arrSorted),
                                    axis=1)  # Add 2-columns array for ith input with previous arraYtemp1
        # Double the array (to enable the random sequence to go 'out-of-bounds')
        YtempB = np.concatenate((YtempA, YtempA), axis=0)

        # Loop until random split has been performed J times
        while rep < J:
            # Redim array to hold accumulated indices for random selected sequences
            idx_random_all_outputs = np.array([], dtype=np.int64).reshape(Q_size, 0)

            for idx_output in range(n_outputs):
                #             idxStart = random.randint(0, N)
                idxStart = math.floor(halton[idx_output, rep])
                idx_random_ith_output = YtempB[idxStart:(idxStart + Q_size), idx_output].reshape(Q_size, 1)
                idx_random_all_outputs = np.concatenate((idx_random_all_outputs, idx_random_ith_output), axis=1)
                idxBBefore = idx_random_all_outputs[:, idx_output]

                if idx_output == 0:
                    idxBAfter = idxBBefore
                else:
                    idxBAfter = list(set(idxBAfter) & set(idx_random_all_outputs[:, idx_output]))

            for i_input in range(n_inputs):
                ks2_stats = TOM.get_max_delta_of_EDFs(Xarr[:, i_input], Xarr[[int(i) for i in idxBAfter], i_input])
                KS_j[0, i_input] = ks2_stats

            # Add j'th ks2-statistics to J x n_inputs array
            KS[rep, :] = KS_j 
            KS_means[rep, :] = np.mean(KS[:rep + 1, :], axis=0) # J'th means for convergence plot

            rep += 1

        self.KS = KS # Array with all K's 
        self.KS_means = KS_means # Array with K's averaged for each J step
        self.KS_df = pd.DataFrame(self.KS_means, columns=X.columns)
        
        # Plot convergence plot and bar plot
        if verbose:
            print(f'{"Calc. time:" :>20}  {(timeit.default_timer() - starttime):.3f} s')
            self.ax_conv = TOM.plot_convergence(self)  
            self.ax_barh = TOM.plot_SA_as_bar(self, self.KS_df)
        
        return self

    def get_max_delta_of_EDFs(x1, x2):
        """
        Perform Kolmogorov-Smirnov two-sample test and return maximum
        distance, K, between the two cumulative distributions.

        Parameters
        ----------
        x1 : array with values for first vector

        x2 : array with values for second vector

        Returns
        -------
        self : returns maximum distance K as float
        """        
        
        # Create bins
        binEdges = np.concatenate((x1, x2), axis=0)
        binEdges.sort()
        #     binEdges = np.sort(binEdges, axis=-1, kind='mergesort')
        binEdges = np.insert(binEdges, 0, float('-inf'))
        binEdges = np.append(binEdges, float('inf'))

        # Compute histograms for both datasets
        binCounts1 = np.histogram(x1, bins=binEdges)[0]
        binCounts2 = np.histogram(x2, bins=binEdges)[0]

        # Calculate empirical distribution functions, EDFs
        sampleCDF1 = np.cumsum(binCounts1) / np.sum(binCounts1)
        sampleCDF2 = np.cumsum(binCounts2) / np.sum(binCounts2)

        # Return maximum distance between EDFs
        max_delta_EDFs = np.max(abs(sampleCDF1 - sampleCDF2))
        #     print(max_delta_EDFs)

        return max_delta_EDFs

    def plot_convergence(self):
        """
        Plot the convergence of the averaged K's for all columns.

        Returns
        -------
        ax : returns figure axes
        """        
        last_row = self.KS_means[-1,:] # Obtain the last, hopefully converged, averaged K's
        last_row_sorted = [sorted(last_row).index(x) for x in last_row] # List with indices sorted by averaged K's
        cols_ranked = [''] * self.KS_means.shape[1] # Empty list to contain reordered column names
        
        for i in range(self.KS_means.shape[1]):
            cols_ranked[last_row_sorted[i]] = list(self.KS_df.columns.values)[i]
        cols_ranked.reverse()
        
        ax = self.KS_df[np.array(cols_ranked)].iloc[10:].plot.line(); # Plot while ignoring 10 first, erratic and non-converged, rows
        legend_ncol = math.ceil(self.KS.shape[1] / 10) # ncol = 1 for 1-10 inputs, 2 for 11-20 inputs, etc.
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=legend_ncol)
        ax.set_xlabel('J');
        ax.set_ylabel('Mean Ks');
        
        return ax

    def plot_SA_as_bar(self, df, metric='percentage', sort=True):
        """
        Create horizontal bar plot for SAtom results.

        Parameters
        ----------
        df : Pandas.DataFrame with mean K values of shape(J, n_features)

        metric : string determining how to aggregate the converged, 
            K values, default='percentage'

        sort : boolean to determine whether to sort the inputs by
            SA rank, default=True

        Returns:
        --------
        ax : returns figure axes
        """
        df_SA = pd.DataFrame(columns=['Input', 'SA, tom'])
        SA_score = df.iloc[-1,:].values
        x_label = 'K, average'

        if metric == 'percentage':
            SA_score = np.array([(val / sum(SA_score)) * 100 for val in SA_score])
            x_label = 'Percentage'

        for i, (col, score) in enumerate(zip(df.columns, SA_score)):
            df_SA.loc[i] = [col, score]

        if sort == True : df_SA = df_SA.sort_values(by="SA, tom", ascending=True) 

        ax = df_SA.plot.barh(x='Input', y='SA, tom', figsize=(6,i/3+1), width=0.8);

        ax.set_xlabel(x_label);

        # Add labels
        for i, v in enumerate(df_SA['SA, tom'].values):
            ax.text(v+0.1, i-0.2 , str(round(v,1)),)
        # Add width to x-axis to make room for labels (corresponding to 2 percentage point) 
        ax_axis = list(ax.axis())
        ax_axis[1] = ax_axis[1]+2
        ax.axis(ax_axis)
        
        return ax    
    
    def next_prime():
        def is_prime(num):
            # Checks if num is a prime value
            for i in range(2, int(num ** 0.5) + 1):
                if (num % i) == 0: return False
            return True

        prime = 3
        while (1):
            if is_prime(prime):
                yield prime
            prime += 2

    def vdc(n, base=2):
        vdc, denom = 0, 1
        while n:
            denom *= base
            n, remainder = divmod(n, base)
            vdc += remainder / float(denom)
        return vdc

    def halton_sequence(size, dim):
        """
        Create Halton sequence.
        
        Parameters
        ----------
            size : integer with number of rows
            
            dim : integer with number of columns

        Returns:
        --------
            array with Halton sequences
        """
        seq = []
        primeGen = TOM.next_prime()
        next(primeGen)
        for d in range(dim):
            base = next(primeGen)
            seq.append([TOM.vdc(i, base) for i in range(size)])
        return seq
    
    def is_unique(self, s):
        """
        Test if all values are the same, i.e. no variance
        
        Parameters:
        -----------
            s : Pandas.Series
        
        Returns:
        --------
            boolean : True if values are equal, and vice versa
        """
        a = s.to_numpy()
        return (a[0] == a).all()