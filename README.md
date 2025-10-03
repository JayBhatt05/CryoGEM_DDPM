# CryoGEM_DDPM
**Reproduction of the DDPM for Retinal Images paper [1] to generate a simulated CryoGEM [2] dataset**  
1. Used the model proposed in [1] as foundation and adapted it to generated the grayscale simulated CryoGEM dataset.
2. Evaluated the generated data using the Fréchet Inception Distance (FID) metric [3].    

<img width="1894" height="886" alt="image" src="https://github.com/user-attachments/assets/2e25c51e-01c1-443f-92ce-cbf431a08e59" />    

Generated Data:  
<img width="1308" height="488" alt="image" src="https://github.com/user-attachments/assets/9b5524a7-4ae7-4492-a502-20a2dfc86a4f" />

  
Real Data:  
<img width="1310" height="472" alt="image" src="https://github.com/user-attachments/assets/8b0c3f96-68fa-4474-b249-f08ee6fb8390" />

     
[1] Alimanov, Alnur & Islam, Md Baharul. (2023). Denoising Diffusion Probabilistic Model for Retinal Image Generation and Segmentation. 1-12. 10.1109/ICCP56744.2023.10233841.  
[2] Zhang, J., Chen, Q., Zeng, Y., Gao, W., He, X., Liu, Z., & Yu, J. (2024). CryoGEM: Physics-Informed Generative Cryo-Electron Microscopy. Advances in Neural Information Processing Systems, 37, 63222-63249.  
[3] Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2017). Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30.
