from Classifiers.components.rm_outliers_encoding import outliers_encode
from Classifiers.components.data_balancing import balancing
from Classifiers.components.data_normalization import normalize
from Classifiers.components.distribution_normalization import features_distribution


if __name__ == '__main__':

    data_cleaned = outliers_encode()
    data_balanced = balancing(data_cleaned)
    data_normalized = normalize(data_balanced)
    features_distribution(data_normalized)