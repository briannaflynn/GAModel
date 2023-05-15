import numpy as np
import pandas as pd
from pygam import LinearGAM, s, l
import matplotlib.pyplot as plt

class GAModel:
    def __init__(self, data, target, spline_index, n_splines, linear_indices=None):
        self.data = data
        self.target = target
        self.spline_index = spline_index
        self.n_splines = n_splines
        self.linear_indices = linear_indices if linear_indices else []
        self.gam = None
        self.pred = None
        self.pred_intervals = None

    def fit_model(self):
        # define the model
        terms = s(self.spline_index, n_splines=self.n_splines)
        for index in self.linear_indices:
            terms += l(index)
        self.gam = LinearGAM(terms).fit(self.data, self.target)

        # predict y and calculate confidence intervals
        self.pred = self.gam.predict(self.data)
        self.pred_intervals = self.gam.prediction_intervals(self.data, width=0.95)

    def plot_results(self):
        # plot the results
        plt.figure(figsize=(8, 6))
        plt.scatter(self.data[:, self.spline_index], self.target, color='k', s=10, label='Data')
        plt.scatter(self.data[:, self.spline_index], self.pred, color='r', s=10, label='Prediction')
        plt.fill_between(np.sort(self.data[:, self.spline_index]), 
                         self.pred_intervals[np.argsort(self.data[:, self.spline_index]), 0], 
                         self.pred_intervals[np.argsort(self.data[:, self.spline_index]), 1], 
                         color='b', alpha=0.2, label='95% CI')
        plt.xlabel(f'Feature {self.spline_index}')
        plt.ylabel('Target')
        plt.legend()
        plt.show()
