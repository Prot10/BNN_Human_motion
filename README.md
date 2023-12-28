# Human Motion Forecasting with Bayesian Neural Networks

## Introduction

This repository presents a project on Human Motion Forecasting using Bayesian Neural Networks (BNNs). The goal is to predict human poses in dynamic scenarios, such as human-robot interaction and autonomous driving, while accounting for uncertainty.

## Repository Structure

The project is organized into distinct components, each housed in its respective directory:

- **data:** Contains datasets relevant to the project.
- **funcs:** Houses utility functions used across the project.
- **autoformer:** Implementation of the Autoformer model.
  - **weights:** Stores model weights.
  - **AutoCorrelation.py:** Auto-correlation implementation.
  - **Autoformer_EncDec.py:** Autoformer encoder-decoder implementation.
  - **Autoformer_Enc_only.py:** Autoformer encoder-only implementation.
  - **Embed.py:** Embedding-related functionality.
  - **Light_Autoformer.py:** Lightweight or specific configurations of the Autoformer.
- **gated_recurrent_nn:** Implementation of the Gated Recurrent Neural Network (GRNN).
  - **checkpoints:** Stores model checkpoints.
  - **decoder.py:** Decoder implementation for the GRNN.
  - **encoder.py:** Encoder implementation for the GRNN.
  - **model.py:** Overall GRNN model implementation.
  - **pos_embed.py:** Positional embedding implementation.
  - **prova.ipynb:** Jupyter notebook for testing purposes.
  - **testing.py:** Script or module for testing the GRNN.
  - **training.py:** Script or module for training the GRNN.
- **transformer:** Implementation of the Transformer model.
  - **pos_embed.py:** Positional embedding implementation.
  - **sta_block.py:** Spatio-Temporal Attention (STA) block implementation.
  - **sttformer.py:** Overall Spatio-Temporal Transformer implementation.
  - **transformer.py:** Overall Transformer model implementation.
- **utils:** Utility functions and modules.
  - **dataloader.py:** Data loading functionality.
  - **gated_recurrent_nn.py:** Utility functions related to the GRNN.
  - **loss.py:** Loss functions implementation.
  - **train.py:** Training-related functionality.
  - **visualizations.py:** Functions for creating visualizations.

## Process

1. **Data Preparation:** Relevant datasets are stored in the `data` directory.
2. **Model Implementations:** Autoformer, GRNN, and Transformer models are implemented in their respective directories.
3. **Training and Testing:** Utilize Jupyter notebooks (`training_autoformer.ipynb`, `training_grnn.ipynb`, `training_transformer.ipynb`, `testing.ipynb`) for model training and testing.
4. **Utility Functions:** `utils` directory contains utility functions for data loading, training, and visualization.
5. **Analysis:** Jupyter notebook (`analysis.ipynb`) for further analysis and evaluation.

## Findings

- Comparative analysis shows that the Autoformer model is the most effective for Human Pose Forecasting.
- Bayesian Neural Networks provide uncertainty estimation, crucial for understanding model confidence.
- Training times are optimized through a hybrid approach in the Autoformer implementation.
- Future work includes refining loss functions and improving computational efficiency.

For detailed information, refer to the specific Jupyter notebooks and source code in each directory.

Feel free to explore, contribute, and provide feedback!

### Contact
- [Bellaroba Albachiara](mailto:bellaroba.1892618@studenti.uniroma1.it)
- [Leoni Paolo](mailto:leoni.1894985@studenti.uniroma1.it)
- [Migliarini Matteo](mailto:migliarini.1886186@studenti.uniroma1.it)
- [Protani Andrea](mailto:protani.1860126@studenti.uniroma1.it)
