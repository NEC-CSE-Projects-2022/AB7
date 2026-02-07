# EfficientSINet-B4: Enhancing Crowd Counting with EfficientNet and Shunting Inhibition Mechanism

## Author
**Yerragunta Mahesh Kumar Reddy**

---

## Team Info
- **22471A570 — Yerragunta Mahesh Kumar Reddy**  
  *Work Done:* Complete project implementation including dataset preprocessing, EfficientNet-B4 integration, model training, evaluation, result analysis, and documentation.

- **22471A555 — Latika Charan Mani**  
  *Work Done:* Density map generation, preprocessing support, experimentation.

- **22471A508 — Chepuri Chaitanya Venkat**  
  *Work Done:* Literature survey, performance comparison, result interpretation.

- **22471A565 — Joshi Vinukonda**  
  *Work Done:* Visualization, graphs, report formatting.

---

## Abstract
Crowd counting from images is vital for applications such as security monitoring, crowd control, and smart video surveillance systems. The Shunting Inhibition Network (SINet) is a biologically inspired approach that incorporates segmentation mechanisms; however, its shallow encoder limits feature extraction capability in highly crowded and complex scenes. To address this limitation, this work proposes **EfficientSINet-B4**, an enhanced crowd counting model that replaces the original SINet encoder with **EfficientNet-B4**, a deeper and more expressive backbone pretrained on large-scale image datasets. While retaining the original decoding structure, the stronger encoder enables improved feature representation and robustness in dense crowd scenarios. Experimental evaluation on the **ShanghaiTech Part-A dataset** demonstrates that EfficientSINet-B4 achieves a **Mean Absolute Error (MAE) of 36.4** and a **Root Mean Squared Error (RMSE) of 85.4**, significantly outperforming the original SINet, which records **52.3 MAE** and **87.6 RMSE**.

---

## Paper Reference (Inspiration)
**EfficientSINet-B4: Enhancing Crowd Counting with EfficientNet and Shunting Inhibition Mechanism**  
**Author:** Yerragunta Mahesh Kumar Reddy  

This work is inspired by the original Shunting Inhibition Network (SINet) and improves it using a deeper EfficientNet-B4 backbone for enhanced feature extraction.

---

## Improvements Over Existing Work
- Replaced the shallow SINet encoder with EfficientNet-B4
- Enhanced multi-scale feature extraction in dense crowd scenes
- Faster convergence and stable training behavior
- Significant reduction in MAE and RMSE
- Lightweight architecture suitable for real-time and edge deployment

---

## Project Overview
This project estimates the number of people present in a crowd image by predicting a **density map**, where each pixel represents the probability of a person’s presence.

### Workflow
Input Image
↓
Preprocessing
↓
EfficientNet-B4 Encoder
↓
SINet Decoder
↓
Density Map
↓
Crowd Count

---

## Dataset Used
**ShanghaiTech Part-A Dataset**  
https://www.kaggle.com/datasets/tthien/shanghaitech

### Dataset Details
- 482 high-resolution crowd images
- 300 training images and 182 testing images
- Head annotations stored in `.mat` files
- Highly dense and complex crowd scenes

---

## Dependencies
- Python
- TensorFlow
- NumPy
- OpenCV
- SciPy
- Matplotlib
- CUDA (for GPU acceleration)

---

## EDA & Preprocessing
- All images resized to 256×256 resolution
- Head annotations converted into density maps using Gaussian kernels
- Image normalization using ImageNet statistics
- Tensor conversion for GPU-based training

---

## Model Training
- Backbone: EfficientNet-B4 (pretrained on ImageNet)
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam
- Learning Rate: 0.0001
- Batch Size: 4
- Epochs: 50–100
- Learning rate scheduling using ReduceLROnPlateau

---

## Model Testing & Evaluation
- Evaluation Metrics:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
- Crowd count obtained by summing predicted density map values
- Density maps resized back to original image resolution during testing

---

## Results
- Achieved MAE of **36.4** and RMSE of **85.4** on ShanghaiTech Part-A
- Outperformed baseline SINet
- Stable convergence and improved accuracy in dense crowd scenes

---

## Limitations & Future Work
### Limitations
- Evaluated only on ShanghaiTech Part-A dataset
- Performance may vary in unseen environments

### Future Work
- Extend experiments to UCF-QNRF and JHU-CROWD++ datasets
- Integrate attention mechanisms
- Improve adaptive density estimation
- Optimize for real-time edge deployment

---

## Deployment
The trained EfficientSINet-B4 model can be deployed in:
- Smart surveillance systems
- Crowd monitoring at public events
- Real-time safety and disaster management systems

Deployment can be implemented using **Flask** or **FastAPI** with GPU or edge-based inference.

---

## License
This project is intended for academic and research purposes only.
