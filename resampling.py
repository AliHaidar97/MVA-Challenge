import numpy as np
import pandas as pd

from binning.equal_frequency import EqualFrequencyBinner
from pykliep import DensityRatioEstimator 
from adapt.instance_based import KLIEP

class DistributionResampler():

    def __init__(self, column=0, sample_frac=1, n_bins=100, seed=None):
        super().__init__()
        self.column = column
        self.sample_frac = sample_frac
        self.binner = EqualFrequencyBinner(n_bins=n_bins)
        self.seed = 1
        self.weight = None

    def fit(self, X, y=None, **fit_params):

        if isinstance(X, pd.DataFrame):
            self.binner.fit(X[[self.column]])
        else:
            self.binner.fit(X[:, self.column].values.reshape(-1, 1))

    def transform_fit(self, X_train, X_test, y=None):
        '''
        X = np.array(X_train)
        Y = np.array(X_test)
        x2 = np.sum(X**2,axis=1)
        y2 = np.sum(Y**2,axis=1)
        xy = X@Y.T

        k = y2 - 2*xy + x2.reshape(-1,1)
        sigma = 0.1
        
        k = np.exp(-k/(2*sigma**2))
        k = np.sum(k,axis=1)
        k = np.abs(k) 
        '''
        print(np.array(X_test[self.column]).shape)
        model = KLIEP(Xt=np.array(X_test[self.column]), kernel="rbf", gamma=[0.01], random_state=0)
        weights= model.fit_weights(np.array(X_train[self.column]), np.array(X_test[self.column]))
    

        self.weight = weights
                
        self.weight = self.weight / np.sum(self.weight)   
        X = X_train
        if isinstance(X, pd.DataFrame):
            return X.sample(
                frac=self.sample_frac,
                weights=self.weight,
                replace=True,
                random_state=self.seed
            )
        else:
            
            return pd.DataFrame(X).sample(
                frac=self.sample_frac,
                weights=self.weight,
                replace=True,
                random_state=self.seed
            ).values


#ewb = xam.preprocessing.EqualFrequencyBinner(n_bins=300)
#ewb.fit(test.reshape(-1, 1))
#train_bins = ewb.transform(train.reshape(-1, 1))[:, 0]
#train_bin_counts = Counter(train_bins)
#weights = np.array([1 / train_bin_counts[x] for x in train_bins])
#weights_norm = weights / np.sum(weights)

# Sample from the training set
#np.random.seed(0)
#sample = np.random.choice(train, size=30000, p=weights_norm, replace=False)

#with plt.xkcd():
#    fig, ax = plt.subplots(figsize=(14, 8))
#    sns.kdeplot(train, ax=ax, label='Train');
#    sns.kdeplot(test, ax=ax, label='Test');
#    sns.kdeplot(sample, ax=ax, label='Train resampled');
#    ax.legend();
