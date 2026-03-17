🚀 Key Features
Dual-Discriminator Architecture: Utilizes two distinct discriminator networks (netD1 and netD2) to evaluate the authenticity of generated images.
Adversarial Stability: The generator is trained to simultaneously satisfy both discriminators, reducing the likelihood of mode collapse.
Comprehensive Evaluation: Includes built-in scripts to calculate Inception Score (IS) and Fréchet Inception Distance (FID) for objective performance benchmarking.
Custom Data Pipeline: Configured to work with specialized datasets, including categories like "Black Snapper" and "Pink Perch".

🏗️ Architecture Details
Generator (netG)
The generator takes a 100-dimensional latent vector and projects it into a 64x64 RGB image.
Layers: Uses ConvTranspose2d layers for learned upsampling.
Activations: ReLU for hidden layers and Tanh for the final output layer to keep pixel values in the $[-1, 1]$ range.
Discriminators (netD1 & netD2)
The model features two discriminators with identical or similar CNN architectures designed to distinguish between real and synthetic data.
Structure: Strided convolutions (Conv2d) replace traditional pooling layers.
Normalization: Employs Batch Normalization to stabilize the training of deep convolutional layers.
Non-linearity: Uses LeakyReLU activations (slope 0.2) to prevent dead neurons during backpropagation.

⚖️ Dual-Adversarial Training Logic
Unlike a standard GAN where the Generator ($G$) faces one Discriminator ($D$), this project implements a joint loss function:
Discriminator Update: $D_1$ and $D_2$ are updated independently using Binary Cross Entropy (BCE) loss on separate batches of real and fake images.
Generator Update: $G$ is updated based on the average adversarial error from both $D_1$ and $D_2$. The Generator wins only when it successfully fools both critics.

📈 Evaluation Metrics
This project uses industry-standard metrics to quantify the quality of the generated fish images:
Inception Score (IS): Measures the clarity and diversity of the generated images based on an Inception-v3 classifier.
Fréchet Inception Distance (FID): Compares the distribution of generated images with the distribution of real images in feature space. A lower FID indicates higher similarity and better performance.

🛠️ Requirements
torch, torchvision (Deep Learning Framework)
numpy, scipy (Scientific Computing)
matplotlib (Visualization)
tqdm (Progress bars)

🖼️ Visualizing Results
The repository includes a visualization suite that generates:
Loss Curves: Real-time tracking of Loss_G, Loss_D1, and Loss_D2.
Image Grids: Progress snapshots showing the transition from random noise to structured imagery.
Training Animation: A frame-by-frame evolution of the generator's performance over epochs.
