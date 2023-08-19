# Integrating-Spatial-Transform-Networks-with-PaDim-Architecture

Authors: Renato-Castro, Cristian Lazo Quispe.

Anomaly detection, an essential component of industrial quality control and surveillance, plays a crucial role in identifying deviations from normal patterns. This extended abstract explores a preliminary research focused on enhancing the anomaly detection capabilities of the PaDim architecture—a state-of-the-art solution for anomaly detection on the MVTEC dataset—through the integration of Spatial Transform Networks (STNs). The aim is to improve performance, particularly in challenging classes such as "zipper" and "screw," where the PaDim architecture achieves lower metrics of performance. Notably, these challenging scenarios often involve objects in non-fixed positions, making anomaly detection intricate in real-world complex scenarios. Through experimentation involving the integration of a Spatial Transform Network using self-supervised training, the performance of this innovative approach is evaluated and it sheds light on both the strengths and limitations of this integration, providing insights into the benefits of leveraging Spatial Transform Networks to handle real-world complexity.

# Project based On
This project is based on the Github PaDiM-Anomaly-Detection-Localization-master: [https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master/tree/main]

# Requirements

<p>Libraries:</p>
  <li>Pytorch</li>
  <li>NumPy</li>
  <li>sklearn</li>
  <li>Pillow</li>
  <li>Scipy</li>
  <li>Skimage</li>
  <li>Skimage</li>
  <li>Matplotlib</li>

-------------------- 

# Downloading Dataset

Follow the next steps to download the dataset:

<li>Go to the link: https://www.mvtec.com/company/research/datasets/mvtec-ad</li>
<li>Check the neccesary boxes for downloading the dataset.</li>
<li>Click on SUBMIT AND CONTINUE TO DOWNLOAD PAGE.</li>
<li>Click on DOWNLOAD THE WHOLE DATASET.</li>

# Test

<p>Command Linux Terminal or Anaconda Prompt for Resnet-18:</p> 

```python
  python src/resnet_18_spatial_transform.py
```

<p>Command Linux Terminal or Anaconda Prompt for Wideresnet:</p> 

```python
  python src/wideresnet_spatial_transform.py
```
