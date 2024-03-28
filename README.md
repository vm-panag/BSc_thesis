[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# <p align="center">Facial Inpainting Methods for Robust Face Recognition</p>

<p float="left">
  <img src="/thesis_images/small_inpainting.PNG" width="400" />
  <img src="/thesis_images/medium_inpainting.PNG" width="400" /> 
</p>

:mailbox_with_mail: <b>For further details about the dataset or the implementation you can contact me on vassilis.panagakis@gmail.com</b> 

### Objective
The objective of this thesis is the restoration of occluded face images to a non-occluded form, in order to facilitate their identification. To achieve that, we investigate a number of inpainting models and we evaluate them on our own face recognition task. The models are based on two principal face inpainting methodologies. The first, supervised method, known as **Generative Landmark Guided Face Inpainting** (or [**LaFIn**](https://github.com/YaN9-Y/lafin)) exploits some of the most innovative and state-of-the-art tools, in the machine learning field, the deep neural networks. LaFIn’s architecture benefits from the integration of facial landmarks and accomplishes the desired face restoration. The second, unsupervised method known as **Principal Component Pursuit using Side Information, Features and Missing Values** (or [**PCPSFM**](https://ieeexplore.ieee.org/document/8657767)) is a variation of the famous Robust Principal Component Analysis (RPCA) method. PCPSFM utilizes domain dependent prior knowledge and manages to recover a low-rank matrix *L<sub>0</sub>*, containing the inpainted face. At the same time, it isolates the occlusions in a separate, sparse matrix *S<sub>0</sub>*.

To evaluate the proposed methods, we worked on a portion of the popular [**CelebA**](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset, which contains face representations of numerous celebrities. For the purpose of our experiments, we created occlusions of different sizes and shapes, in order to test the models under multiple scenarios. Concerning the evaluation process, three different models were employed to detect the dominant inpainting method, based on the percentage of successful matches between the inpainted faces and the clean faces of all the celebrity identities in the dataset.

### Requirements
* Python 3.7
* Pytorch (&ge; 1.5)
* NVIDIA GPU & CUDA cuDNN

### Citation
```bib
@thesis{vmpanagakis2021,
    author      = {Panagakis Vasileios-Marios},
    title       = {Facial Inpainting Methods for Robust Face Recognition},
    type        = {bscthesis},
    url         = {https://github.com/vm-panag/BSc_thesis},
    institution = {Department of Informatics and Telecommunications, University of Athens},
    year        = {2021}
}
```
### Acknowledgements 
Supervisor: [Prof. Yannis Panagakis](https://scholar.google.com/citations?user=z1bkjU8AAAAJ&hl=en)<br />

