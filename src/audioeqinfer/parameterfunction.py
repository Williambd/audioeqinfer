from scipy import stats
import numpy as np
import time
import scipy
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pedalboard.io import AudioFile
from patsy import dmatrix,bs,build_design_matrices

class ParameterFunction:
    def __init__(self, n_parameters = 502, domain= (0., 22050.)):
        '''
        This class implements a cubic spline function for audio equalization.
        n_parameters: number of parameters for the spline
        domain: domain of frequencies for the spline function
        '''
        self.n_parameters = n_parameters
        self.domain = domain

        # Set up knots based on number of desired parameters
        n_internal_knots = max(n_parameters - 4, 1)  # Cubic splines need degree + 1
        self.knots = np.linspace(domain[0], domain[1], n_internal_knots + 2)[1:-1]  # exclude endpoints

        self.design_info = dmatrix(f"bs(x, knots={list(self.knots)}, degree=3, include_intercept=True) - 1", 
                         {"x": np.linspace(0., 22050., 500)}).design_info
    
    def set_coefficients(self, coefficients:np.ndarray):
        '''
        Set the coefficients for the spline.
        coefficients: coefficients for the spline
        '''
        self.coefficients = coefficients

    def number_of_coefficients_needed(self):
        '''
        Returns the number of coefficients needed for the spline.
        This is equal to the number of knots + 4 (for cubic splines).
        '''
        return len(self.knots) + 4
    
        # Build the design matrix for this specific x value
    def __call__(self, x:np.ndarray) -> np.ndarray:
        '''
        Evaluate the spline function at a specific x values.
        x: x values to evaluate the spline at
        '''
        basis_at_x = build_design_matrices([self.design_info], {"x": x})[0]
        y_value = np.dot(basis_at_x, self.coefficients)
        return y_value

    def evaluate(self,x:np.ndarray) -> np.ndarray:
        '''
        Evaluate the spline function at a specific x values.
        x: x values to evaluate the spline at
        '''
        basis_at_x = build_design_matrices([self.design_info], {"x": x})[0]
        y_value = np.dot(basis_at_x, self.coefficients)
        return y_value
    
    def graph(self, show_basis = False):
        '''
        Graph the spline function.
        show_basis: if True, show the basis functions as well
        '''
        x = np.linspace(0., 22000., 100)
        y = self.evaluate(x)

        if show_basis:
            basis = build_design_matrices([self.design_info], {"x": x})[0]
            for i in range(basis.shape[1]):
                plt.plot(x, basis[:,i])
        
        plt.plot(x,y)
        plt.show()
    
    def fit_to_data(self, frequency:np.ndarray, magnitude:np.ndarray):
        '''
        Fit the spline to the given frequency and magnitude data.
        frequency: Frequency values (x-axis)
        magnitude: Magnitude values (y-axis)
        '''
        basis_matrix = build_design_matrices([self.design_info], {"x": frequency})[0]

        # Fit the GLM model
        fit = sm.GLM(magnitude, basis_matrix).fit()
        self.coefficients = fit.params