# ee046202-unsupervised-learning-data-analysis

<h1 align="center">
  <br>
Technion EE 046202 - Unsupervised Learning and Data Analysis
  <br>
  <img src="https://github.com/taldatech/ee046202-unsupervised-learning-data-analysis/blob/master/assets/tut_xx_mnist_anim.gif" width="200"><img src="https://github.com/taldatech/ee046202-unsupervised-learning-data-analysis/blob/master/assets/tut_xv_vae_anim.gif" width="200">
</h1>

<p align="center">
    <a href="https://taldatech.github.io">Tal Daniel</a> •
    <a href="https://ronmeir.net.technion.ac.il/">Ron Meir</a>
  </p>

Jupyter Notebook tutorials for the Technion's EE 046202 course "Unsupervised Learning and Data Analysis"

<h4 align="center">
    <a href="https://colab.research.google.com/github/taldatech/ee046202-unsupervised-learning-data-analysis"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <a href="https://nbviewer.jupyter.org/github/taldatech/ee046202-unsupervised-learning-data-analysis/tree/master/"><img src="https://raw.githubusercontent.com/taldatech/ee046211-deep-learning/main/assets/nbviewer_badge.svg" alt="Open In NBViewer"/></a>
    <a href="https://mybinder.org/v2/gh/taldatech/ee046202-unsupervised-learning-data-analysis/master"><img src="https://mybinder.org/badge_logo.svg" alt="Open In Binder"/></a>

</h4>



For the old tutorials, please see `winter20` branch.

- [ee046202-unsupervised-learning-data-analysis](#ee046202-unsupervised-learning-data-analysis)
  * [Running The Notebooks](#running-the-notebooks)
    + [Running Online](#running-online)
    + [Running Locally](#running-locally)
  * [Agenda](#agenda)
  * [Installation Instructions](#installation-instructions)
    + [Libraries to Install](#libraries-to-install)

## Running The Notebooks
You can view the tutorials online or download and run locally.

### Running Online

|Service      | Usage |
|-------------|---------|
|Jupyter Nbviewer| Render and view the notebooks (can not edit) |
|Binder| Render, view and edit the notebooks (limited time) |
|Google Colab| Render, view, edit and save the notebooks to Google Drive (limited time) |


Jupyter Nbviewer:

[![nbviewer](https://raw.githubusercontent.com/taldatech/ee046211-deep-learning/main/assets/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/taldatech/ee046202-unsupervised-learning-data-analysis/tree/master/)


Press on the "Open in Colab" button below to use Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/taldatech/ee046202-unsupervised-learning-data-analysis)

Or press on the "launch binder" button below to launch in Binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/taldatech/ee046202-unsupervised-learning-data-analysis/master)

Note: creating the Binder instance takes about ~5-10 minutes, so be patient

### Running Locally

Press "Download ZIP" under the green button `Clone or download` or use `git` to clone the repository using the 
following command: `git clone https://github.com/taldatech/ee046202-unsupervised-learning-data-analysis.git` (in cmd/PowerShell in Windows or in the Terminal in Linux/Mac)

Open the folder in Jupyter Notebook (it is recommended to use Anaconda). Installation instructions can be found in `Setting Up The Working Environment.pdf`.



## Agenda

|File       | Topics Covered |
|----------------|---------|
|`Setting Up The Working Environment.pdf`| Guide for installing Anaconda locally with Python 3 and PyTorch, integration with PyCharm and using GPU on Google Colab |
|`ee046202_tutorial_00_probability_optimization.ipynb\pdf`| Probability basics, random variables, Bayes rule, expectancy, PDF and CDF, smoothing theorem, Multivariate Normal Distribution, Largrange Multipliers, Useful inequalities: Markov, Chebyshev, Hoeffding |
|`ee046202_tutorial_01_classic_statistics_point_estimation.ipynb\pdf`| Classical statistical inference (frequentist), Point Estimation, Evaluating estimators: Bias, Variance, Mean Squared Error (MSE), Consistency, The Tail Sum formula (non-parametric estimation), Maximum Likelihood Estimation (MLE), Vector/Matrix derivatives, KL-Divergence, Entropy, The Weak Law of Large Numbers|
|`ee046202_tutorial_02_classic_statistics_confidence_intervals.ipynb\pdf`| Confidence Intervals and Interval Estimation, Bootstrap, PPF (Inverse of the CDF), Empirical CDF, Dvoretzky–Kiefer–Wolfowitz (DKW) Inequality|
|`ee046202_tutorial_03_classic_statistics_hypothesis_testing_1.ipynb\pdf`| Hypothesis Testing, Null and Alternative Hypotheses, Test statistic, z-stat, p-value, Significance level, Error types (type 1 and type 2), The central limit theorem (CLT)|
|`ee046202_tutorial_04_classic_statistics_hypothesis_testing_2.ipynb\pdf`| Hypothesis Testing recap, t-test (t-statistic), Pearson Chi-squared test, Uniformly Most Powerful (UMP) Test, How to choose statitical test|
|`ee046202_tutorial_05n_dim_reduction_pca_kernels.ipynb\pdf`| Dimensionality reduction, Principle Component Analysis (PCA), PCA for compression, Relation to SVD, The Breast Cancer Dataset, Eigenvectors, Eigenvalues, The Transpose Trick, Kernels motivation, Feature extraction, Kernels, The Kernel Trick, Mercer condition, Radial Basis Function (RBF), Kernel PCA (KPCA)|
|`ee046202_tutorial_06_dim_reduction_tsne.ipynb\pdf`| Stochastic Neighbor Embedding (SNE), t-SNE, The crowding problem, Student t-distribution, KL-divergence|
|`ee046202_tutorial_07_deep_learn_pytorch_ae.ipynb\pdf`| PyTorch, MNIST, Fashion-MNIST, MULTI-layer Perceptron (MLP), Fully-Connected (FC), Convolutional Networks (CNN), Autoencoders|
|`ee046202_tutorial_08_deep_unsupervised_vae_1.ipynb\pdf`| Implicit and Explicit Generative models, GANs, Variational Inference (VI), Variational Autoencoder (VAE), Evidence Lower Bound (ELBO), Reparameterization Trick|
|`ee046202_tutorial_09_deep_unsupervised_vae_2.ipynb\pdf`| 	VAE implementation, interpolation in the latent space, saving and loading models in PyTorch|
|`ee046202_tutorial_10_generative_adversarial_networks_gan.ipynb\pdf`|Generative Adversarial Network (GAN), Explicit/Implicit density estimation, Nash Equilibrium with Proof, Mode Collapse, Vanisihng/Diminishing Gradient, Conditional GANs, WGAN, EBGAN, BEGAN, Tips for Training GANs|
|`ee046202_tutorial_11_expectation_maximization.ipynb\pdf`|Clustering, K-Means, Gaussian Mixture Model (GMM), Expectation Maximization (EM) algorithm, Bernoulli Mixture Model (BMM)|
|`ee046202_tutorial_12_spectral_clustering.ipynb\pdf`|Spectral Clustering (Graph Clustering), Degree matrix, Weighted Adjacency matrix, Similarity graph, epsilon-neighborhood graph, KNN graph, Fully connected graph, Graph Laplacian, GraphCut, MinCut, RatioCut|

## Installation Instructions

For the complete guide, with step-by-step images, please consult `Setting Up The Working Environment.pdf`

1. Get Anaconda with Python 3, follow the instructions according to your OS (Windows/Mac/Linux) at: https://www.anaconda.com/distribution/
2. Create a new environment for the course:
In Windows open `Anaconda Prompt` from the start menu, in Mac/Linux open the terminal and run `conda create --name torch`. Full guide at https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands
3. To activate the environment, open the terminal (or `Anaconda Prompt` in Windows) and run `conda activate torch`
4. Install the required libraries according to the table below (to search for a specific library and the corresponding command you can also look at https://anaconda.org/)

### Libraries to Install

|Library         | Command to Run |
|----------------|---------|
|`Jupyter Notebook`|  `conda install -c conda-forge notebook`|
|`numpy`|  `conda install -c conda-forge numpy`|
|`matplotlib`|  `conda install -c conda-forge matplotlib`|
|`pandas`|  `conda install -c conda-forge pandas`|
|`scipy`| `conda install -c anaconda scipy `|
|`scikit-learn`|  `conda install -c conda-forge scikit-learn`|
|`seaborn`|  `conda install -c conda-forge seaborn`|
|`pytorch` (cpu)| `conda install pytorch torchvision cpuonly -c pytorch` |
|`pytorch` (gpu)| `conda install pytorch torchvision cudatoolkit=10.0 -c pytorch` |


5. To open the notbooks, open Anancinda Navigator or run `jupyter notebook` in the terminal (or `Anaconda Prompt` in Windows) while the `torch` environment is activated.
