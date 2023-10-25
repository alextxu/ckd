# Comparative Knowledge Distillation

This repository covers the implementation of the following paper submitted to ICLR 2024: "Comparative Knowledge Distillation" (CKD).
This repository is based on the official implementation of the "Contrastive Representation Distillation" paper (https://github.com/HobbitLong/RepDistiller).


This repository implements (using PyTorch):
- The original knowledge distillation seen in "Distilling the Knowledge in a Neural Network"
- 5 state-of-the-art relational knowledge distillation methods:
    - (mixup) - "mixup: Beyond Empirical Risk Minimization"
    - (mixup3) - A version of mixup that mixes three samples instead of two
    - (rkd) - "Relational Knowledge Distillation"
    - (dist) - "Knowledge Distillation from A Stronger Teacher"
    - (crd) - "Contrastive Representation Distillation"
- 3 white-box knowledge distillation methods:
    - (hint) - "Fitnets: Hints for Thin Deep Nets"
    - (vid) - "Variational Information Distillation for Knowledge Transfer"
    - (correlation) - "Correlation Congruence for Knowledge Distillation"

<!-- All white-box methods from CRD repo:
(FitNet) - Fitnets: hints for thin deep nets  
(AT) - Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks
    via Attention Transfer  
(SP) - Similarity-Preserving Knowledge Distillation  
(CC) - Correlation Congruence for Knowledge Distillation  
(VID) - Variational Information Distillation for Knowledge Transfer  
(PKT) - Probabilistic Knowledge Transfer for deep representation learning  
(AB) - Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons  
(FT) - Paraphrasing Complex Network: Network Compression via Factor Transfer  
(FSP) - A Gift from Knowledge Distillation:
    Fast Optimization, Network Minimization and Transfer Learning  
(NST) - Like what you like: knowledge distill via neuron selectivity transfer  -->

## Installation and Setup

This repo was tested with Ubuntu 22.04.6 LTS, Python 3.10.12, PyTorch 2.0.1, and CUDA 11.8. More details about the Python environment can be found in `requirements.txt`.

*Note that the distillation program uses Weights and Biases to log validation and test accuracies, so be sure to change the project ID and username in `train_student.py` before running.*

<!-- -------------- NOT YET CHANGED --------------- -->

## Running

### Pretraining Teacher Models
There are two options here:
1. Fetch the pretrained teacher models (same as in the original CRD repository) by:

    ```
    sh scripts/fetch_pretrained_teachers.sh
    ```
   which will download and save the models to `save/models`
2. To pretrain teacher models locally, run `train_teacher.py` with a specified teacher model architecture. An example would be:
    ```
    python train_teacher.py --model=resnet110
    ```
    **Note: before running distillation, please be sure to rename the folders for the relevant teacher saves to `{model_name}_vanilla/`**

### Distillation
There are also two options here:
1. To perform distillation with default hyperparameter searching, there are sweep files that can be run with Weights and Biases:
    - CKD: `wandb sweep ckd.yml`
    - mixup and mixup3: `wandb sweep mixup.yml`
    - Other relational methods: `wandb sweep relational.yml`
    - White-box methods with CKD: `wandb sweep white_box_ckd.yml`
    - White-box methods without CKD: `wandb sweep white_box.yml`
2. To test from other customized hyperparameters, run `train_student.py` and be sure to specify the student model architectures, as well as the method to use. The teacher model used will be the last save from the teacher pretraining in step 1. An example of running CKD (without combining with white-box methods)is as follows:

    ```
    python train_student.py --model_s resnet32 --distill kd --relational ckd --learning_rate 0.1 --subset_size 2000 --trial 1
    ```
    where the flags are explained as:
    - `--model_s`: the architecture of the student model, see `models/__init__.py` to check the available model types.
    - `--distill`: the white-box distillation method to use (or 'kd' for black-box)
    - `--relational`: the relational method to use (if enabling white-box methods, this should be left as None or "ckd_inter")
    - `--subset_size`: the number of samples to use for training and validation (80% of this number is used for training, and hence passed through the teacher)
    - `--learning_rate`: the starting learning rate for training the student, default: `0.05`
    - `--trial`: the experimental ID to differentiate between multiple runs

    Some other options are listed below:
    - `--model_t`: the architecture of the teacher model (please ensure pretraining for this teacher model has finished), defaults to corresponding teacher from paper results
    - `--w_cls`: the weight of the classification loss between logit and ground truth, default: `1`
    - `--w_kd`: the weight of the KD loss, default: `0`
    - `--w_rel_scale`: constant factor to be multiplied by the default weighting of relational loss when combining loss components, default: `1`
    - `--w_inter_scale`: constant factor to be multiplied by the default weighting of the white-box method loss when combining loss components, default: `1`