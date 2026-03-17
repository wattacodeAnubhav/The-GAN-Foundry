# The GAN Foundry: Dual-Adversarial Image Synthesis 🎨🐟

## 📌 Project Overview
**The GAN Foundry** is an advanced generative modeling project that implements a **Dual-Discriminator DCGAN** architecture. Unlike standard GANs that rely on a single critic, this system utilizes two independent discriminator networks to guide the generator. This "two-player" critique system significantly enhances training stability and helps the model capture the intricate textures of specialized marine datasets, such as **Black Snapper** and **Pink Perch**.

## 🚀 Key Features
* **Dual-Discriminator Architecture:** Utilizes two distinct discriminator networks (`netD1` and `netD2`) to evaluate the authenticity of generated images from different perspectives.
* **Adversarial Stability:** The generator is trained to simultaneously satisfy both discriminators, reducing the likelihood of **Mode Collapse**—a common failure in generative modeling.
* **Comprehensive Evaluation:** Includes built-in scripts to calculate **Inception Score (IS)** and **Fréchet Inception Distance (FID)** for objective performance benchmarking.
* **Custom Data Pipeline:** Configured to process specialized biological categories, including high-resolution imagery of "Black Snapper" and "Pink Perch."

---

## 🏗️ Architecture Details

### Generator (`netG`)
The generator maps a 100-dimensional latent noise vector ($z$) into a $64 \times 64$ RGB image.
* **Layers:** Employs `ConvTranspose2d` layers for learned upsampling.
* **Activations:** **ReLU** for hidden layers and **Tanh** for the final output layer to normalize pixel values within the $[-1, 1]$ range.

### Discriminators (`netD1` & `netD2`)
The model features two discriminators with symmetric CNN architectures designed to distinguish between real and synthetic data.
* **Structure:** Strided convolutions (`Conv2d`) replace traditional pooling layers to maintain spatial information.
* **Normalization:** Employs **Batch Normalization** to stabilize the training of deep convolutional layers.
* **Non-linearity:** Uses **LeakyReLU** activations (slope 0.2) to prevent "dead neurons" during backpropagation.

---

## ⚖️ Dual-Adversarial Training Logic
In this framework, the Generator ($G$) faces a joint challenge from two critics ($D_1$ and $D_2$):

1.  **Discriminator Update:** $D_1$ and $D_2$ are updated independently using **Binary Cross Entropy (BCE)** loss on separate batches of real and fake images.
2.  **Generator Update:** $G$ is updated based on the **average adversarial error** from both $D_1$ and $D_2$. The Generator "wins" only when it successfully fools both critics simultaneously.

$$L_G = \frac{1}{2} [ \mathbb{E}_{z \sim p_z(z)}[\log D_1(G(z))] + \mathbb{E}_{z \sim p_z(z)}[\log D_2(G(z))] ]$$

---

## 📈 Evaluation Metrics
This project uses industry-standard quantitative metrics to measure the realism and diversity of the generated fish species:

* **Inception Score (IS):** Measures the clarity and diversity of the generated images based on an Inception-v3 classifier.
* **Fréchet Inception Distance (FID):** Compares the distribution of generated images with real images in feature space. **A lower FID** indicates higher similarity to the real dataset.

---
