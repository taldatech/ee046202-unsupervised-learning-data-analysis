# ee046202-unsupervised-learning-data-analysis

Jupyter Notebook tutorials for the Technion's EE 046202 course "Unsupervised Learning and Data Analysis"

[![nbviewer](https://jupyter.org/assets/main-logo.svg)](https://nbviewer.jupyter.org/github/taldatech/ee046202-unsupervised-learning-data-analysis/tree/master/) 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/taldatech/ee046202-unsupervised-learning-data-analysis) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/taldatech/ee046202-unsupervised-learning-data-analysis/master)

<img src="https://github.com/taldatech/ee046202-unsupervised-learning-data-analysis/blob/master/assets/tut_xx_mnist_anim.gif" width="200"><img src="https://github.com/taldatech/ee046202-unsupervised-learning-data-analysis/blob/master/assets/tut_xv_vae_anim.gif" width="200">

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

[![nbviewer](https://jupyter.org/assets/main-logo.svg)](https://nbviewer.jupyter.org/github/taldatech/ee046202-unsupervised-learning-data-analysis/tree/master/)


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
