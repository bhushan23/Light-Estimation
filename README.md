# Light-Estimation
## Introduction
Lighting estimation is difficult task. Many optimization based methods estimates lighting using Spherical Harmonics.
Spherical harmonics cannot model all types of shading.
Unfortunately, due to lack of ground truth for real images, lighting estimation becomes even more difficult task. Hence, we have used adversarial networks to bridge the gap between synthetic and real images.  

We have - 
1. Implemented [LDAN](https://arxiv.org/abs/1709.01993) which uses Generative Adversarial Networks for domain adaptation
2. Implemented Denoising Autoencoder to compare against Adversarial method for lighting estimation

## Structure
1. LDAN folder contains domain adaptation implementation 
    * models.py       - FeatureNet, LightingNet, Generator
    * train.py        - Training FeatureNet, LightingNet and Generator
    * shading.py      - Utility methods for generating Shading from Face normal and Spherical harmonics
    * utils.py        - Misc utility methods required for saving images and other
    * dataLoading.py  - loading CelebA data, SfSNet synthetic data and LDAN synthetic data
    * LDAN/exp_1.py                         - Experiement 1 below
    * LDAN/exp_2_training_with_sirfs_sh.py  - Experiement 2 below
    * LDAN/exp_2_training_with_true_sh.py   - Experiment 3 below
    * AutoLighting/resNet_CelebA.py         - Experiment 4 with CelebA
    * AutoLighting/resNet_sfsNet.py         - Experiment 5 with SfSNet dataset
    
## Experiements performed
  1. Estimating lighting of CelebA 
     * Trained with LDAN Synthetic dataset
     * Real Image dataset: CelebA
     * SIRFS SH for training GAN: Using SIRFS method to generate SH for CelebA dataaset
     * Ground truth for training feature net and lighting net - provided in LDAN dataset
     * Conclusion- Synthetic images space was adapted by the net
     
  2. Estimating lighting of SfSNet dataset with SIRFS SH for GAN training 
     * This is to verify estimated shading with ground truth shading
     * Training on LDAN synthetic dataset
     * Used SIRFS SH for training GAN: Using SIRFS method to generate SH for SfSNet dataset
     * Ground truth Normal, Shading for SfSNet dataset provided
     * MSE plot with respect to SIRFS SH goes down and with respect to ground truth SH increases verifying that SIRFS domain is being adapted
     * Conclusion- Although results are not better than SIRFS estimated shading, we can see domain being adapted
     
  3. Estimating lighting of SfSNet dataset with ground truth SH for GAN training 
     * This is to verify estimated shading with ground truth shading
     * Training on LDAN synthetic dataset
     * Used ground truth SH for training GAN
     * Ground truth Normal, Shading for SfSNet dataset provided
     * MSE plot with respect to SIRFS SH increases and with respect to ground truth decrease i.e. opposite to Experiment 2
     * Conclusion- Although results are not better than SIRFS estimated shading, we can see domain being adapted
     
   4. AutoLighting
      * This was to compare domain adaptaion with respect to denoising autoencoder
      * Steps - 
         1. Generate Noisy SH using SIRFS method on SfSNet data
         2. Train denoising autoencoder to denoise SH
         3. Use trained denoising autoencoder to remove noise from SH on real images SH
         4. Use Use feature net and lighting net to estimate lighting on real images
       * This approach does not out performs domain adaptation
       * Conclusion - Synthetic and real images spaces are different and adversarial approach performs well to understand and estimate lighting for real images trained on synthetic images.
     
## Report
[https://github.com/bhushan23/Light-Estimation/blob/master/Lighting_Estimation_Report.pdf](Project Report)
