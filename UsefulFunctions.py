'''
coding: utf-8
----------------------------------------------------
Useful functions for Applied Statistics Exam

Authors:
 - Ulrik Friis-Jensen (lgb543@alumni.ku.dk)
 
Co-authors:
 - Christian Noes Petersen (lbc622@alumni.ku.dk)
 - David Harding-Larsen (pfl888@alumni.ku.dk)
 - Lars Erik Skjegstad (zfj803@alumni.ku.dk)
 - Marcus Frahm Nygaard (nwb154@alumni.ku.dk)
 - Lasse Skjoldborg Krog (cxq235@alumni.ku.dk)

Date:
 - Exam 2019 version from 15-01-2020
 - Latest update: 19-01-2020
-----------------------------------------------------
'''
################################################################################################################### Imports 
import numpy as np
from numpy.linalg import inv
from scipy import stats
from iminuit import Minuit     

################################################################################################### Functions for ChiSquare
def constant(x, const):
    return const

def linear(x, x0, x1):
    return x1 * x + x0

def polynomial_2(x, x0, x1, x2):
    return x2* x**2 + x1 * x + x0

def polynomial_3(x, x0, x1, x2, x3):
    return x3 * x**3 + x2 * x**2 + x1 * x + x0

def binomial(x, n, p, N = 1):
    return N * stats.binom.pmf(x,n,p)

def poisson(x, mu, N = 1) :
    return N * stats.poisson.pmf(x, mu)

def gaussian(x, N = 1.0, mu = 0.0, sigma = 1.0, binwidth = 1.0) :
    return binwidth * N * stats.norm.pdf(x, mu, sigma)

def gaussian_x2(x, N1, mu1, sigma1, N2, mu2, sigma2, binwidth1 = 1.0, binwidth2 = 1.0):
    return gaussian(x, N1, mu1, sigma1, binwidth=binwidth1) + gaussian(x, N2, mu2, sigma2, binwidth=binwidth2)

def gaussian_x3(x, N1, mu1, sigma1, N2, mu2, sigma2, N3, mu3, sigma3, binwidth1 = 1.0, binwidth2 = 1.0, binwidth3 = 1.0):
    return gaussian(x, N1, mu1, sigma1, binwidth=binwidth1) + gaussian(x, N2, mu2, sigma2, binwidth=binwidth2) + gaussian(x, N3, mu3, sigma3, binwidth=binwidth3)

def exponential_decay(x, C, k):
    return C * np.exp(-x/k)

def exponential_growth(x, C, k):
    return C * np.exp(x/k)

def sigmoid(x, L, x0):
    return L * ((x - x0) / np.sqrt(1 + (x - x0)**2))

########################################################################################################## Simple functions

def gauss_prob(sig1, sig2, mu=0, sig=1, two_tailed=True):
    '''
    Calculates the probability for a gaussian value to lie
    in a specified interval of sigmas away from the mean.
    '''
    prob1 = stats.norm.sf(sig1, loc=mu, scale=sig)
    prob2 = stats.norm.sf(sig2, loc=mu, scale=sig)
    if two_tailed:
        prob = 2 * (prob1 - prob2)
        print(f'Probability (two tailed) for a Gaussian value to lie between {sig1} and {sig2} sigma away from the mean: {prob:.2%}')
    else:
        prob = prob1 - prob2
        print(f'Probability (one tailed) for a Gaussian value to lie between {sig1} and {sig2} sigma away from the mean: {prob:.2%}')
    return None

def gauss_percentile(prob, mu=0, sig=1, digits=3, tail=1, return_values=False):
    '''
    Determines the value of a gaussianly distributed variable at which the integral 
    from the variable and away from the mean is equal to a given percentile.
    
    If no input for mu and sig is given the result is returned in sigma away from the mean.
    
    The optional argument tail determines if the value(s) are for the lower tail, higher tail or both.
      - 0 is lower tail
      - 1 is upper tail (default)
      - 2 is both tails
    '''
    result = None
    
    if tail==0:
        lower_val = stats.norm.ppf(prob, loc=mu, scale=sig)
        print(f'The lower tail contains {prob:.1%} if the integral is taken from {lower_val:.{digits}f} and down.')
        if return_values: result = lower_val
        return result
    
    if tail==1:
        upper_val = stats.norm.isf(prob, loc=mu, scale=sig)
        print(f'The upper tail contains {prob:.1%} if the integral is taken from {upper_val:.{digits}f} and up.')
        if return_values: result = upper_val
        return result
    
    if tail==2:
        lower_val = stats.norm.ppf(prob/2, loc=mu, scale=sig)
        upper_val = stats.norm.isf(prob/2, loc=mu, scale=sig)
        print(f'The two tails contains {prob:.1%} combined if the integral is taken from {lower_val:.{digits}f} and down and from {upper_val:.{digits}f} and up.')
        if return_values: result = np.array([lower_val, upper_val])
        return result

def mean_no_unc(data, digits=4,get_values=False):
    '''
    Calculates the mean, RMS and uncertainty on mean for a data sample w/o uncertainties.
    '''
    mean = data.mean()
    unc_on_data = np.sqrt(np.sum((data-mean)**2)/(len(data)-1))
    unc_on_mean = unc_on_data / np.sqrt(len(data))
    print(f'''
    ____________________________________________________
    ----------------------------------------------------
    Mean of data set:  {mean:.{digits}f} +/- {unc_on_mean:.{digits}f} (RMS = {unc_on_data:.{digits}f})
    ____________________________________________________''')
    if get_values:
        return mean, unc_on_data, unc_on_mean
    else:
        return None
    
def bin_data(data, Nbins, xmin, xmax):
    '''
    Converts a list or array to a histogram.
    Returns bin_centers, counts, error on counts and binwidth.
    '''
    counts, bin_edges = np.histogram(data, bins=Nbins, range=(xmin, xmax))
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
    s_counts = np.sqrt(counts) 
    
    x = bin_centers[counts>0]
    y = counts[counts>0]
    sy = s_counts[counts>0]
    
    binwidth = (xmax-xmin) / Nbins
    return x, y, sy, binwidth

####################################################################################################################################### Advanced functions

def chi2_test_uniform(bin_centers, counts, get_values=False):
    '''
    Tests if a histogram is uniformly distributed.
    '''
    data = counts
    expected = data.sum() / len(bin_centers)
    chi2 = np.sum( (data - expected)**2 / data )
    Ndof = len(bin_centers)-1
    p_chi2 = stats.chi2.sf(chi2, Ndof) 

    print(f'''
    _____________________________
    -----------------------------
    ChiSquare test (uniform dist)
    -----------------------------
    Chi2-value = {chi2:.3f}
    Ndof       = {Ndof}
    Chi2-prob  = {p_chi2:.2%}
    _____________________________''')
    if get_values:
        return chi2, Ndof, p_chi2
    else:
        return None
    
def pearsons_chi2(obs_counts, exp_counts_or_dist, use_dist=False, A_and_B = True, get_values=False):
    '''
    Pearson's ChiSquare test for comparing a histogram to another histogram or distribution.
    Input arguments are the observed counts and the expected binomial/poisson.
    
    A_and_B determines if the denominator is A+B or just B.
    '''  
    chi2 = 0
    events = 0
    if use_dist:
        exp_counts = obs_counts.sum()*exp_counts_or_dist
    else:
        exp_counts = exp_counts_or_dist
        
    for A, B in zip(obs_counts, exp_counts):
        if A_and_B:
            denom = A + B
        else:
            denom = B
        if A != 0 and B != 0:
            chi2 += (A - B)**2 / (denom)
            events += 1

    Ndof = events
    p_chi2 = stats.chi2.sf(chi2, Ndof) 

    print(f'''
    ___________________________
    ---------------------------
     Pearson's ChiSquare test
    ---------------------------
    Chi2-value = {chi2:.3f}
    Ndof       = {Ndof}
    Chi2-prob  = {p_chi2:.2%}
    ___________________________''')
    if get_values:
        return chi2, Ndof, p_chi2
    else:
        return None

def ks_comparison(data1, data2, get_values=False, **kwdarg):
    '''
    Kolmogorov-Smirnov test for comparing to datasets.
    Returns the test statistic, critical value and p-value either as a string or numbers.
    Alternative hypothesis can be:
        'two-sided'
        'less'
        'greater'
    '''
    D, p = stats.ks_2samp(data1, data2, **kwdarg)
    d = D * np.sqrt(len(data1))
    print(f'''
    ____________________________________________________________
    ------------------------------------------------------------
    Result of Kolmogorov-Smirnov comparison between two datasets
    ------------------------------------------------------------
    KS statistic   :    {D:.4f}
    Critical value :    {d:.4f}
    p-value        :    {p:.2%}
    ____________________________________________________________
    ''')
    if get_values:
        return D, d, p
    else:
        return None
    
def ks_test(data1, cdf, get_values=False, **kwdarg):
    '''
    Kolmogorov-Smirnov test for comparing data to a continous distribution.
    Returns the test statistic, critical value and p-value either as a string or numbers.
    Look at the documentation for scipy.stats.ks_test for further details.
    '''
    D, p = stats.kstest(data1, cdf, **kwdarg)
    d = D * np.sqrt(len(data1))
    print(f'''
    _____________________________________________
    ---------------------------------------------
          Result of Kolmogorov-Smirnov test
    ---------------------------------------------
    KS statistic   :    {D:.4f}
    Critical value :    {d:.4f}
    p-value        :    {p:.2%}
    _____________________________________________
    ''')
    if get_values:
        return D, d, p
    else:
        return None
    
def chi2_fit(func, x, y, yerr, get_values=False, print_result=True, pedantic = False, print_level = 0, digits=5, latex_format=False, **kwdarg):
    '''
    ChiSquare fit of a given function to a given data set.
    
    Returns the fitted parameters for further plotting.
    
    **kwdarg allows the user to specify initial parameter 
    values and fix values using the syntax from Minuit.
    
    The digits variable controls the amount of digits 
    given in the printed result.
    
    The latex_format command allows the user to generate a 
    table for latex with the fitted parameter values.
    '''
    chi2obj = Chi2Regression(func, x, y, yerr)
    minuit_obj = Minuit(chi2obj, pedantic=pedantic, print_level=print_level, **kwdarg)

    minuit_obj.migrad()   

    if (not minuit_obj.get_fmin().is_valid) :                                   # Check if the fit converged
        print("    WARNING: The ChiSquare fit DID NOT converge!!!")

    Chi2_value = minuit_obj.fval                                             # The Chi2 value
    NvarModel = len(minuit_obj.args)
    Ndof = len(x) - NvarModel
    ProbChi2 = stats.chi2.sf(Chi2_value, Ndof)
    if not print_result:
        return minuit_obj.args, minuit_obj.errors
    if latex_format:
        print(r'''----------------------------------------------------------------------------------
NB! This is not a perfect formatting.
Units, caption, label and sometimes parameter names must be changed in LaTex.
----------------------------------------------------------------------------------

\begin{table}[b]
    \centering
    \begin{tabular}{lrr}
    \hline
    \hline
        Parameter & Value (Unit) & Unc. (Unit) \\
    \hline''')
        for name in minuit_obj.parameters:
            print(f'        ${name}$ & ${minuit_obj.values[name]:.{digits}f}$ & ${minuit_obj.errors[name]:.{digits}f}$ \\\ ')
        print(r'''    \hline
    \hline''')
        print(r'        $\chi^2$-value = {0:.3f} & Ndof = {1} & $\chi^2$-prob = {2:.3f} \\'.format(Chi2_value,Ndof,ProbChi2))
        print(r'''    \hline
    \hline
    \end{tabular}
    \caption{Results of $\chi^2$-fit.}
    \label{tab:chi2_fit}
\end{table}''')
    else:
        print(f'''
    _____________________________________________________
    -----------------------------------------------------
                    ChiSquare Fit Results
    -----------------------------------------------------
    Chi2-value = {Chi2_value:.3f}
    Ndof       = {Ndof}
    Chi2-prob  = {ProbChi2:.2%}
    -----------------------------------------------------''')
        for name in minuit_obj.parameters:
            print(f'\n    Chi2 Fit result:    {name} = {minuit_obj.values[name]:.{digits}f} +/- {minuit_obj.errors[name]:.{digits}f}')
        print('    _____________________________________________________')
    if get_values:
        return minuit_obj.args, minuit_obj.errors, Chi2_value, Ndof, ProbChi2
    else:
        return minuit_obj.args, minuit_obj.errors

def UnbinnedLH_reg(func, x, extended=True, pedantic = False, print_level = 0, digits=5, latex_format=False, **kwdarg):
    '''
    ChiSquare fit of a given function to a given data set.
    
    Returns the fitted parameters for further plotting.
    
    **kwdarg allows the user to specify initial parameter 
    values and fix values using the syntax from Minuit.
    
    The digits variable controls the amount of digits 
    given in the printed result.
    
    The latex_format command allows the user to generate a 
    table for latex with the fitted parameter values.
    '''
    ulreg = UnbinnedLH(func, x, extended = extended)
    minuit_obj = Minuit(ulreg, pedantic=pedantic, print_level=print_level, **kwdarg)

    minuit_obj.migrad()   

    if (not minuit_obj.get_fmin().is_valid) :                                   # Check if the fit converged
        print("    WARNING: The ChiSquare fit DID NOT converge!!!")

    if latex_format:
        print(r'''----------------------------------------------------------------------------------
NB! This is not a perfect formatting.
Units, caption, label and sometimes parameter names must be changed in LaTex.
----------------------------------------------------------------------------------

\begin{table}[b]
    \centering
    \begin{tabular}{lrr}
    \hline
    \hline
        Parameter & Value (Unit) & Unc. (Unit) \\
    \hline''')
        for name in minuit_obj.parameters:
            print(f'        ${name}$ & ${minuit_obj.values[name]:.{digits}f}$ & ${minuit_obj.errors[name]:.{digits}f}$ \\\ ')
        print(r'''    \hline
    \hline
    \hline
    \end{tabular}
    \caption{Results of unbinned likelihood regression.}
    \label{tab:ulreg}
\end{table}''')
    else:
        print(f'''
    __________________________________________________________
    ----------------------------------------------------------
                 UnbinnedLH Regression Results
    ----------------------------------------------------------''')
        for name in minuit_obj.parameters:
            print(f'\n    UnbinnedLH reg. result:    {name} = {minuit_obj.values[name]:.{digits}f} +/- {minuit_obj.errors[name]:.{digits}f}')
        print('    __________________________________________________________')
    return minuit_obj.args, minuit_obj.errors

def log_LH_sweep(data_list, log_func, N_steps, start, end, show_values = False, digits = 3):
    '''
    Log likelihood sweep of variable of choice
    
    Data list should be a 1D array
    
    log_func should be the logarithm of the pdf times -2,
    as function of x and the variable of choice
    
    N_steps is number of steps in the sweep,
    while start and end is the initial and final values of the sweep
    
    The digits variable controls the amount of digits given in the printed result.
    '''
    ullh_minval = 999999.9
    ullh_minpos = 0.0
    step = (end-start) / N_steps

    ullh = np.zeros(N_steps+1)
    var  = np.zeros(N_steps+1)

    for i in range(N_steps+1):
        var_hypo = start + i*step         
        var[i] = var_hypo
        ullh[i] = 0

        for x in data_list:     
            ullh[i] +=  log_func(x,var[i]) # Unbinned LLH function

        if show_values and i % 10 == 0:
            print(f" {i:3d}:  p = {var_hypo:4.{digits}f}   log(ullh) = {ullh[i]:6.{digits}f}")

        # Search for minimum values of ullh:

        if (ullh[i] < ullh_minval) :
            ullh_minval = ullh[i]
            ullh_minpos = var_hypo
    
    if show_values:
        print(f'''
    ________________________________________________________
    --------------------------------------------------------
                 Log Likelihood Sweep Results
    --------------------------------------------------------
    Minimum value of sweep:      {ullh_minval:.{digits}f}
    Variable value at minimum:   {ullh_minpos:.{digits}f}
    ________________________________________________________''')

    return var, ullh, ullh_minval, ullh_minpos

def MonteCarlo(func, N_points, xmin = 0, xmax = 1, ymin = 0, ymax = 1, print_result=True, **kwdarg):
    '''
    Generate random number according to a pdf using Monte Carlo.
    Inputs are:
        - the pdf
        - the number of points to be generated
        - Ranges of the x and y values (optional)
        - any additional arguments for the pdf (optional)    
    '''
    N_try = 0
    x_accepted = np.zeros(N_points)
    for i in range(N_points):

        while True:
            
            # Count the number of tries, to get efficiency/integral
            N_try += 1   

            # Range that f(x) is defined/wanted in:
            x_test = np.random.uniform(xmin, xmax)  

            # Upper bound for function values:
            y_test = np.random.uniform(ymin, ymax)

            if (y_test <= func(x_test, **kwdarg)):
                break

        x_accepted[i] = x_test
        
    # Efficiency
    eff = N_points / N_try                        

    # Error on efficiency (binomial)
    eff_error = np.sqrt(eff * (1-eff) / N_try) 

    # Integral
    integral =  eff * (xmax-xmin) * (ymax-ymin)

    # Error on integral
    eintegral = eff_error * (xmax-xmin) * (ymax-ymin)  
    if print_result:
        print(f'''
    _____________________________________________________________
    -------------------------------------------------------------
                             Monte Carlo 
    -------------------------------------------------------------
    Generation of random numbers according to the given pdf.
    -------------------------------------------------------------
    Intervals used to sample random numbers:
    x in [{xmin}, {xmax}]
    y in [{ymin}, {ymax}]
    
    Integral of the pdf is:  {integral:.4f} +/- {eintegral:.4f}
    
    Efficiency of the Accept/Reject method is:  {eff:.2%} +/- {eff_error:.2%}
    _____________________________________________________________''')
    return x_accepted

def array_of_MC_sums(func, N_values, N_points, xmin = 0, xmax = 1, ymin = 0, ymax = 1, print_result=False, **kwdarg):
    '''
    Generate an array of N_values elements.
    All elements are the sum of N_points random numbers generated according to a pdf using Monte Carlo.
    Inputs are:
        - the pdf
        - the number of elements to be generated
        - the number of points to be generated
        - Ranges of the x and y values (optional)
        - any additional arguments for the pdf (optional) 
    '''
    u = np.zeros(N_values)
    for i in range(N_values):
        u[i] = MonteCarlo(func, N_points, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, print_result=print_result, **kwdarg).sum()
    return u

def MC_errorpropagation(func, N_exp, mu1, sig1, mu2, sig2, rho=0.0, get_data=False, get_values=False):
    '''
    Monte Carlo simulation of the error propagation of a function with 2 variables.
    
    mu1 and mu2 is the value of the variable. sig1 and sig2 is the uncertainties on the values.
    
    If both uncertainties have the same value the function can't do correlated 
    parameters as it would result in division by 0.
    '''
    result = None
    if not (-1.0 <= rho <= 1.0): 
        raise ValueError(f"Correlation factor not in interval [-1,1], as it is {rho12:6.2f}")
    if sig1 == sig2:
        theta = 0
    else:
        theta = 0.5 * np.arctan( 2.0 * rho * sig1 * sig2 / ( np.square(sig1) - np.square(sig2) ) )

    sigu = np.sqrt( np.abs( ((sig1*np.cos(theta)**2) - (sig2*np.sin(theta))**2 ) / ( (np.cos(theta))**2 - np.sin(theta)**2) ) )
    sigv = np.sqrt( np.abs( ((sig2*np.cos(theta)**2) - (sig1*np.sin(theta))**2 ) / ( (np.cos(theta))**2 - np.sin(theta)**2) ) )

    sigu = np.sqrt( np.abs( (((sig1*np.cos(theta))**2) - (sig2*np.sin(theta))**2 ) / ( (np.cos(theta))**2 - np.sin(theta)**2) ) )
    sigv = np.sqrt( np.abs( (((sig2*np.cos(theta))**2) - (sig1*np.sin(theta))**2 ) / ( (np.cos(theta))**2 - np.sin(theta)**2) ) )

    u = np.random.normal(0.0, sigu, N_exp)
    v = np.random.normal(0.0, sigv, N_exp)

    x1_all = mu1 + np.cos(theta)*u - np.sin(theta)*v
    x2_all = mu2 + np.sin(theta)*u + np.cos(theta)*v

    y_all = func(x1_all, x2_all)

    y_mean, y_rms, y_unc = mean_no_unc(y_all, get_values=True)
    
    if get_data:
        result = y_all
    elif get_values:
        result = y_mean, y_unc, y_rms
    elif get_data and get_values:
        result = y_all, y_mean, y_unc, y_rms
    return result

def correlation_matrix(x, y=None, rowvar=True, print_level=True):
    '''
    Calculates the correlation matrix between any number of variables 
    for a given data set.
    
    rowvar determines the direction the data set is read in. True if 
    each variable is a row, False if each variable is a column.
    '''
    corr_matrix = np.corrcoef(x, y=y, rowvar=rowvar)
    if print_level == True:
        print(f'''
    ______________________________________________
    ----------------------------------------------
                 Correlation matrix
    ----------------------------------------------''')
        for row in corr_matrix:
            print(f'    {row}')
        print('    ______________________________________________')
    
    return corr_matrix

def calc_separation(x, y):
    '''
    Calculate the seperation between two histograms.
    '''
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    std_x = np.std(x, ddof=1)
    std_y = np.std(y, ddof=1)
    d = np.abs((mean_x - mean_y)) / np.sqrt(std_x**2 + std_y**2)
    return d

def calc_ROC(hist1, hist2) :
    '''
    Calculate ROC curve from two histograms (hist1 is signal, hist2 is background).
    '''

    # First we extract the entries (y values) and the edges of the histograms
    y_sig, x_sig_edges, _ = hist1 
    y_bkg, x_bkg_edges, _ = hist2
    
    # Check that the two histograms have the same x edges:
    if np.array_equal(x_sig_edges, x_bkg_edges) :
        
        # Extract the center positions (x values) of the bins (both signal or background works - equal binning)
        x_centers = 0.5*(x_sig_edges[1:] + x_sig_edges[:-1])
        
        # Calculate the integral (sum) of the signal and background:
        integral_sig = y_sig.sum()
        integral_bkg = y_bkg.sum()
    
        # Initialize empty arrays for the True Positive Rate (TPR) and the False Positive Rate (FPR):
        TPR = np.zeros_like(y_sig) # True positive rate (sensitivity)
        FPR = np.zeros_like(y_sig) # False positive rate ()
        
        # Loop over all bins (x_centers) of the histograms and calculate TN, FP, FN, TP, FPR, and TPR for each bin:
        for i, x in enumerate(x_centers): 
            
            # The cut mask
            cut = (x_centers < x)
            
            # True positive
            TP = np.sum(y_sig[~cut]) / integral_sig    # True positives
            FN = np.sum(y_sig[cut]) / integral_sig     # False negatives
            TPR[i] = TP / (TP + FN)                    # True positive rate
            
            # True negative
            TN = np.sum(y_bkg[cut]) / integral_bkg      # True negatives (background)
            FP = np.sum(y_bkg[~cut]) / integral_bkg     # False positives
            FPR[i] = FP / (FP + TN)                     # False positive rate            
            
        return FPR, TPR
    
    else:
        AssertionError("Signal and Background histograms have different bins and ranges")

def get_covariance_offdiag(X, Y):
    '''
    Calculate the off-diagonal value [var_i, var_j] in the covariance matrix.
    '''
    return np.cov(X, Y, ddof=1)[0, 1]

def calc_covar_matrix(data_list1, data_format=1, printlevel = False, n = 0):
    '''
    Calculate the covariance matrix for a data_list.
    
    The data_list could have two possible formats of the type [rows, columns].
    Format type 1 is [data object, variable data].
    Format type 2 is [variable data, data object].
    
    A data object is a flat array with one value of each variable.
    Variable data is a flat array with all values measured for that variable.
    
    n dictates the text printed. Use n=0 to print the covariance matrix.
    Use n=1 and n=2 to differentiate between covariance matrices when calculating Fisher coefficients.
    '''
    if data_format == 2:
        data_list1 = data_list1.T
        
    cov_mat = np.zeros((len(data_list1[0]),len(data_list1[0])))
    for ivar in range(len(data_list1[0])):
        for jvar in range(len(data_list1[0])):
            cov_mat[ivar, jvar] = get_covariance_offdiag(data_list1[:, ivar],data_list1[:, jvar])
    if printlevel == True and n == 0:
        print(f'''
    ______________________________________________
    ----------------------------------------------
             Covariance matrix
    ----------------------------------------------''')
        for row in cov_mat:
            print(f'    {row}')
        print('    ______________________________________________')
    if printlevel == True and n == 1:
        print(f'''
    ______________________________________________
    ----------------------------------------------
             Covariance matrix #1
    ----------------------------------------------''')
        for row in cov_mat:
            print(f'    {row}')
        print('    ______________________________________________')
    if printlevel == True and n == 2:
        print(f'''
    ______________________________________________
    ----------------------------------------------
             Covariance matrix #2
    ----------------------------------------------''')
        for row in cov_mat:
            print(f'    {row}')
        print('    ______________________________________________')
    return cov_mat

def fisher_coef(data_list1, data_list2, data_format=1, printlevel = False):
    '''
    Calculates Fisher coefficients for two data lists.
    
    The data_lists could have two possible formats of the type [rows, columns].
    Format type 1 is [data object, variable data].
    Format type 2 is [variable data, data object].
    
    A data object is a flat array with one value of each variable.
    Variable data is a flat array with all values measured for that variable.
    '''
    if data_format == 2:
        data_list1 = data_list1.T
        data_list2 = data_list2.T
        
    if len(data_list1[0]) == len(data_list2[0]):
        cov_mat1 = calc_covar_matrix(data_list1, printlevel = printlevel, n = 1)
        cov_mat2 = calc_covar_matrix(data_list2, printlevel = printlevel, n = 2)
        covmat_comb_inv = inv(cov_mat1 + cov_mat2)
        mu_list1 = np.zeros(len(data_list1[0]))
        mu_list2 = np.zeros(len(data_list1[0]))
        for ivar in range(len(data_list1[0])):
            var_list1 = data_list1[:, ivar]
            var_list2 = data_list2[:, ivar]
            mu_list1[ivar] = var_list1.mean()
            mu_list2[ivar] = var_list2.mean()
        wf = covmat_comb_inv.dot(mu_list1-mu_list2)
        if printlevel == True:
            print(f'''
    _______________________________________________
    -----------------------------------------------
                 Fisher coefficients
    -----------------------------------------------''')
            for row in wf:
                print(f'    {row}')
            print('    _______________________________________________')
        return wf
    else:
        print('Data lists do not have same dimensions')

def fisher_descri(data_list1, data_list2, data_format=1, printlevel = False):
    '''
    Calculates Fisher discriminants for data_list1 and data_list2.
    
    The data_lists could have two possible formats of the type [rows, columns].
    Format type 1 is [data object, variable data].
    Format type 2 is [variable data, data object].
    
    A data object is a flat array with one value of each variable.
    Variable data is a flat array with all values measured for that variable.
    '''
    if data_format == 2:
        data_list1 = data_list1.T
        data_list2 = data_list2.T
    
    prik1 = []
    prik2 = []
    wf = fisher_coef(data_list1, data_list2, printlevel = printlevel)
    for ivar in range(len(data_list1[0])):
        var_list1 = data_list1[:, ivar]
        var_list2 = data_list2[:, ivar]
        wf_element = wf[ivar]
        prik1.append(wf_element*var_list1)
        prik2.append(wf_element*var_list2)
    fisher_descri1 = np.zeros(len(prik1[0]))
    fisher_descri2 = np.zeros(len(prik2[0]))
    for i in range(len(prik1[0])):
        fisher_element1 = 0
        for j in range(len(prik1)):
            if len(prik1[0]) == len(prik1[j]):
                fisher_element1 += prik1[j][i]
            else:
                print('Variable lists do not have same dimensions')
                return None
        fisher_descri1[i] = fisher_element1
    for k in range(len(prik2[0])):
        fisher_element2 = 0
        for l in range(len(prik2)):
            if len(prik2[0]) == len(prik2[l]):
                fisher_element2 += prik2[l][k]
            else:
                print('Variable lists do not have same dimensions')
                return None
        fisher_descri2[k] = fisher_element2
    fisher_descri1 = np.array(fisher_descri1)
    fisher_descri2 = np.array(fisher_descri2)
    return fisher_descri1, fisher_descri2

        
###########################################################################################################################
###########################################################################################################################
'''
Integration of External_Functions to make this library independant of non-standard imports.
Author: Christian Michelsen, NBI, 2018
'''

def format_value(value, decimals):
    """ 
    Checks the type of a variable and formats it accordingly.
    Floats has 'decimals' number of decimals.
    """
    
    if isinstance(value, (float, np.float)):
        return f'{value:.{decimals}f}'
    elif isinstance(value, (int, np.integer)):
        return f'{value:d}'
    else:
        return f'{value}'

def values_to_string(values, decimals):
    """ 
    Loops over all elements of 'values' and returns list of strings
    with proper formating according to the function 'format_value'. 
    """
    
    res = []
    for value in values:
        if isinstance(value, list):
            tmp = [format_value(val, decimals) for val in value]
            res.append(f'{tmp[0]} +/- {tmp[1]}')
        else:
            res.append(format_value(value, decimals))
    return res

def len_of_longest_string(s):
    """ Returns the length of the longest string in a list of strings """
    return len(max(s, key=len))

def nice_string_output(d, extra_spacing=5, decimals=3):
    """ 
    Takes a dictionary d consisting of names and values to be properly formatted.
    Makes sure that the distance between the names and the values in the printed
    output has a minimum distance of 'extra_spacing'. One can change the number
    of decimals using the 'decimals' keyword.  
    """
    
    names = d.keys()
    max_names = len_of_longest_string(names)
    
    values = values_to_string(d.values(), decimals=decimals)
    max_values = len_of_longest_string(values)
    
    string = ""
    for name, value in zip(names, values):
        spacing = extra_spacing + max_values + max_names - len(name) - 1 
        string += "{name:s} {value:>{spacing}} \n".format(name=name, value=value, spacing=spacing)
    return string[:-2]

def add_text_to_ax(x_coord, y_coord, string, ax, fontsize=12, color='k'):
    """ Shortcut to add text to an ax with proper font. Relative coords."""
    ax.text(x_coord, y_coord, string, family='monospace', fontsize=fontsize,
            transform=ax.transAxes, verticalalignment='top', color=color)
    return None

# =============================================================================
#  Probfit replacement
# =============================================================================

from iminuit.util import make_func_code
from iminuit import describe #, Minuit,

def set_var_if_None(var, x):
    if var is not None:
        return np.array(var)
    else: 
        return np.ones_like(x)
    
def compute_f(f, x, *par):
    
    try:
        return f(x, *par)
    except ValueError:
        return np.array([f(xi, *par) for xi in x])

class Chi2Regression:  # override the class with a better one
    
    def __init__(self, f, x, y, sy=None, weights=None):
        
        self.f = f  # model predicts y for given x
        self.x = np.array(x)
        self.y = np.array(y)
        
        self.sy = set_var_if_None(sy, self.x)
        self.weights = set_var_if_None(weights, self.x)
        self.func_code = make_func_code(describe(self.f)[1:])

    def __call__(self, *par):  # par are a variable number of model parameters
        
        # compute the function value
        f = compute_f(self.f, self.x, *par)
        
        # compute the chi2-value
        chi2 = np.sum(self.weights*(self.y - f)**2/self.sy**2)
        
        return chi2

def simpson38(f, edges, bw, *arg):
    
    yedges = f(edges, *arg)
    left38 = f((2.*edges[1:]+edges[:-1]) / 3., *arg)
    right38 = f((edges[1:]+2.*edges[:-1]) / 3., *arg)
    
    return bw / 8.*( np.sum(yedges)*2.+np.sum(left38+right38)*3. - (yedges[0]+yedges[-1]) ) #simpson3/8

def integrate1d(f, bound, nint, *arg):
    """
    compute 1d integral
    """
    edges = np.linspace(bound[0], bound[1], nint+1)
    bw = edges[1] - edges[0]
    
    return simpson38(f, edges, bw, *arg)

class UnbinnedLH:  # override the class with a better one
    
    def __init__(self, f, data, weights=None, badvalue=-100000, extended=False, extended_bound=None, extended_nint=100):
        
        self.f = f  # model predicts PDF for given x
        self.data = np.array(data)
        self.weights = set_var_if_None(weights, self.data)
        self.bad_value = badvalue
        
        self.extended = extended
        self.extended_bound = extended_bound
        self.extended_nint = extended_nint
        if extended and extended_bound is None:
            self.extended_bound = (np.min(data), np.max(data))

        
        self.func_code = make_func_code(describe(self.f)[1:])

    def __call__(self, *par):  # par are a variable number of model parameters
        
        logf = np.zeros_like(self.data)
        
        # compute the function value
        f = compute_f(self.f, self.data, *par)
    
        # find where the PDF is 0 or negative (unphysical)        
        mask_f_positive = (f>0)

        # calculate the log of f everyhere where f is positive
        logf[mask_f_positive] = np.log(f[mask_f_positive]) * self.weights[mask_f_positive] 
        
        # set everywhere else to badvalue
        logf[~mask_f_positive] = self.bad_value
        
        # compute the sum of the log values: the LLH
        llh = -np.sum(logf)
        
        if self.extended:
            extended_term = integrate1d(self.f, self.extended_bound, self.extended_nint, *par)
            llh += extended_term
        
        return llh
    
    def default_errordef(self):
        return 0.5

class BinnedLH:  # override the class with a better one
    
    def __init__(self, f, data, bins=40, weights=None, weighterrors=None, bound=None, badvalue=1000000, extended=False, use_w2=False, nint_subdiv=1):
        
        self.weights = set_var_if_None(weights, data)


        self.f = f
        self.use_w2 = use_w2
        self.extended = extended

        if bound is None: 
            bound = (np.min(data), np.max(data))

        self.mymin, self.mymax = bound

        h, self.edges = np.histogram(data, bins, range=bound, weights=weights)
        
        self.bins = bins
        self.h = h
        self.N = np.sum(self.h)

        if weights is not None:
            if weighterrors is None:
                self.w2, _ = np.histogram(data, bins, range=bound, weights=weights**2)
            else:
                self.w2, _ = np.histogram(data, bins, range=bound, weights=weighterrors**2)
        else:
            self.w2, _ = np.histogram(data, bins, range=bound, weights=None)


        
        self.badvalue = badvalue
        self.nint_subdiv = nint_subdiv
        
        
        self.func_code = make_func_code(describe(self.f)[1:])
        self.ndof = np.sum(self.h > 0) - (self.func_code.co_argcount - 1)
        

    def __call__(self, *par):  # par are a variable number of model parameters

        # ret = compute_bin_lh_f(self.f, self.edges, self.h, self.w2, self.extended, self.use_w2, self.badvalue, *par)
        ret = compute_bin_lh_f2(self.f, self.edges, self.h, self.w2, self.extended, self.use_w2, self.nint_subdiv, *par)
        
        return ret


    def default_errordef(self):
        return 0.5

import warnings

def xlogyx(x, y):
    
    #compute x*log(y/x) to a good precision especially when y~x
    
    if x<1e-100:
        warnings.warn('x is really small return 0')
        return 0.
    
    if x<y:
        return x*np.log1p( (y-x) / x )
    else:
        return -x*np.log1p( (x-y) / y )

#compute w*log(y/x) where w < x and goes to zero faster than x
def wlogyx(w, y, x):
    if x<1e-100:
        warnings.warn('x is really small return 0')
        return 0.
    if x<y:
        return w*np.log1p( (y-x) / x )
    else:
        return -w*np.log1p( (x-y) / y )

def compute_bin_lh_f2(f, edges, h, w2, extended, use_sumw2, nint_subdiv, *par):
    
    N = np.sum(h)
    n = len(edges)

    ret = 0.
    
    for i in range(n-1):
        th = h[i]
        tm = integrate1d(f, (edges[i], edges[i+1]), nint_subdiv, *par)
        
        if not extended:
            if not use_sumw2:
                ret -= xlogyx(th, tm*N) + (th-tm*N)

            else:
                if w2[i]<1e-200: 
                    continue
                tw = w2[i]
                factor = th/tw
                ret -= factor*(wlogyx(th,tm*N,th)+(th-tm*N))
        else:
            if not use_sumw2:
                ret -= xlogyx(th,tm)+(th-tm)
            else:
                if w2[i]<1e-200: 
                    continue
                tw = w2[i]
                factor = th/tw
                ret -= factor*(wlogyx(th,tm,th)+(th-tm))

    return ret

def compute_bin_lh_f(f, edges, h, w2, extended, use_sumw2, badvalue, *par):
    
    mask_positive = (h>0)
    
    N = np.sum(h)
    midpoints = (edges[:-1] + edges[1:]) / 2
    b = np.diff(edges)
    
    midpoints_pos = midpoints[mask_positive]
    b_pos = b[mask_positive]
    h_pos = h[mask_positive]
    
    if use_sumw2:
        warnings.warn('use_sumw2 = True: is not yet implemented, assume False ')
        s = np.ones_like(midpoints_pos)
        pass
    else: 
        s = np.ones_like(midpoints_pos)

    
    E_pos = f(midpoints_pos, *par) * b_pos
    if not extended:
        E_pos = E_pos * N
        
    E_pos[E_pos<0] = badvalue
    
    ans = -np.sum( s*( h_pos*np.log( E_pos/h_pos ) + (h_pos-E_pos) ) )

    return ans