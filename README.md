# DEEP LEARNING FOR CLUSTERING OF WEATHER AND ATMOSPHERIC DISPERSION FEATURES


**ABSTRACT**

Dimensionality reduction alleviates computational burdens faced in various tasks
from design optimization to predictive modelling. Hence, popular dimensionality
reduction approaches aim at projecting the data onto a subspace spanned by linear and
non-linear functions obtained from the compression of a dataset of solution snapshots.

This thesis approaches the problem of obtaining meaningful low-dimensionality
representations of high-dimensional data, referred to as dimensionality reduction, by
the design and application of suitable deep neural network configurations. Moreover,
in the applications targeted by this dissertation it is crucial for the algorithms to learn
the underlying data manifold in order to facilitate further processing. The novelty of
the proposed technique is to design auto encoders that make use of weather variables
and temporality, in order to learn enhanced data representations for weather data
clustering. We examine a plethora of end-to-end trainable models improving
clustering procedure on multiple years of weather data in the European continent.

A number of real-world applications could benefit from such models, for instance
anomaly detection in renewable energy sources utilizing weather features from wind-
flow and solar energy in order to develop monitoring systems. Furthermore, scenarios
including environment and climate change could be developed for atmospheric
inverse analysis systems to estimate emissions or gas flux sources. Our knowledge in
meteorology for agricultural purposes could be enhanced from such frameworks by
collecting dominant meteorological patterns.

Among the various ways of learning representations, this thesis focuses on deep
learning methods: those that are formed by the composition of multiple non-linear
transformations, with the goal of yielding more abstract – and ultimately more useful
– representations. We present alternative non-linear representation learning methods,
such as Convolutional and LSTM (Long Short Term Memory) networks and discuss
advantages and disadvantages, focusing on extracting reusable and robust deep
features. These methods are known to be able to disentangle the data manifolds, i.e.,
to uncover the latent dimension(s) along which the temporal and/or spatio-temporal
3aspect of the data. However, their application on the clustering of weather data has not
been sufficiently studied before.

The models are evaluated and compared to alternative methods demonstrating
competitive results. They are shown to map input data into a new space where the
data are linear separable. Hence, the extracted representations can be applicable in a
plethora of domains, such as in discovering robust weather patterns, developing
emergency response applications in atmospheric physics, radiology, environmental
research, healthcare and in other application areas.
Keywords: Unsupervised learning, Dimensionality reduction, manifold learning,
Deep learning, weather clustering



![Text](https://github.com/Giwrgosyea/Weather-Analysis/blob/master/DEEP%20LEARNING%20FOR%20CLUSTERING%20OF%20WEATHER%20AND%20ATMOSPHERIC%20DISPERSION%20FEATURES/DNNs%20Results.png)

Results from evaluating the our proposed models for 10 and 30
simulated locations reporting accuracy as a function of the dimensionality
reduction configuration and the choice of the proposed cluster descriptor 
