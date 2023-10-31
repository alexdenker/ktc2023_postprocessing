# Submission to the KTC 2023

For this submission we developed a **postprocessing UNet** for EIT segmentation. The postprocessing network is trained on a dataset of synthetic phantoms and simulated measurements.

# Table of contents 
1. [Usage](#usage)
2. [Method](#method)
3. [Examples](#examples)
4. [Evaluation](#evaluation)
5. [Authors](#authors)

## Usage

We provide the `enviroment.yml` file to restore the conda enviroment used for the submission. You can create the enviroment using the following command:

```
conda env create -f environment.yml
```

The network weights are stored at **upload files here**. The script `main.py` can be used reconstruct phantoms: 

```
python main.py /path_to_input_folder /path_to_ouput_folder difficulty_level
```


### Enviroment


## Method

Our goal is to train a postprocessing UNet, 

$$ f_\theta(\mathcal{R}(U)) \approx \sigma_\text{segmentation} $$

to predict the segmentation of the conductivty $\sigma$ based on some initial reconstruction $\mathcal{R}(U)$. We directly represent this semgentation map on the $256 \times 256$ pixel grid. Further, the initial reconstruction is also  interpolated to the $256 \times 256$ pixel grid. This has the practical advantage that we can implement the diffusion model as a convolutional neural network and are independent of the underlying mesh. 

### Initial Reconstructions

We use linearised time-difference reconstructions for the conditional input. Computing this linearised time-difference reconstruction amounts to solving a regularised least squares problem

$$ \Delta \sigma = (J_{\sigma_0} \Gamma_e^{-1} J + \alpha_1 R_\text{SM} + \alpha_2 R_\text{Laplace} + \alpha_3 R_\text{NOSER})^{-1} J_{\sigma_0}^T \Gamma_e^{-1} \Delta U, $$

where $J_{\sigma_0}$ is the Jacobian w.r.t. to a constant background conductivity, $\Gamma_e^{-1}$ is the noise precision matrix and $\Delta U = U^\delta - F(\sigma_0)$ is the difference of the measurements. For $F(\sigma_0)$ we make use of the provided measurements of the empty water tank. 

We use a combination of three different priors, where $R_\text{SM}$ denotes a [smoothness prior](https://www.fips.fi/KTC2023_Instructions_v3_Oct12.pdf), $R_\text{Laplace}= - \bigtriangleup$ the graph laplacian and $R_\text{NOSER} = diag(J_{\sigma_0} \Gamma_e^{-1} J)$ the [NOSER](https://pubmed.ncbi.nlm.nih.gov/36909677/) prior. 

In total, we use five different combinations of $\alpha_1, \alpha_2$ and $\alpha_3$ as each regularisation results in different artifacts in the reconstruction. This means, that our conditional input, interpolated to the pixel grid, is of the size $5 \times 256 \times 256$. 


### Training

### Forward Operator 

### Synthetic Training Data


## Examples

## Evaluation


We evaluate the postprocessing UNet w.r.t. the [score function](https://www.fips.fi/KTC2023_Instructions_v3_Oct12.pdf) used in the challenge. 


| Level         |    Score       |
|---------------|----------------|
| 1            | $X.XXX$       |
| 2            | $X.XXX$       |
| 3            | $X.XXX$       |
| 4            | $X.XXX$       |
| 5            | $X.XXX$       |
| 6            | $X.XXX$       |
| 7            | $X.XXX$       |


## Authors

- Alexander Denker<sup>1</sup>, Tom Freudenberg<sup>1</sup>, Željko Kereta<sup>2</sup>, Imraj RD Singh<sup>2</sup>, Tobias Kluth<sup>1</sup>, Simon Arridge <sup>2</sup>

<sup>1</sup>Center of Industrial Mathematics (ZeTeM), University of Bremen

<sup>2</sup>Department of Computer Science, University College London