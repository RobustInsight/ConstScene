# ConstScene: A Dataset and Model for Advancing Robust Semantic Segmentation in Construction Environments


<p float="left">
  <img src="assets/orig.jpg" width="40%" />
  <img src="assets/over.jpg" width="40%" />
</p>


## Overview
This repository accompanies the paper "ConstScene: A Dataset and Model for Advancing Robust Semantic Segmentation in Construction Environments" authored by Maghsood Salimi, Mohammad Loni, Sara Afshar, Marjan Sirjani, and Antonio Cicchetti from Mälardalen University and Volvo Construction Equipment.

## Abstract
The increasing demand for autonomous machines in construction environments necessitates the development of robust object detection algorithms that can perform effectively across various weather and environmental conditions. This paper introduces the ConstScene dataset, a semantic segmentation dataset tailored for construction sites, addressing challenges posed by diverse weather conditions. The dataset enhances the training and evaluation of object detection models, fostering adaptability and reliability in real-world construction applications.

## Dataset
The ConstScene dataset is available in the `dataset` folder. It includes annotated images captured under a wide range of weather conditions, such as sunny days, rainy periods, foggy atmospheres, and low-light situations. Additionally, environmental factors like dirt/mud on the camera lens are integrated into the dataset through both actual captures and synthetic generation, simulating complex conditions in construction sites.

### Dataset Structure
- `images/`: Contains raw images.
- `annotations/`: Contains semantic segmentation masks.
- `metadata/`: Includes metadata files with additional information.
- `synthetic/`: Contains synthetically generated images and masks.
- ...

```
/data/D1/
        ├── train/
        │   ├ _classes.csv
        │   ├ image1.jpg
        │   ├ image1_mask.png
        │   ├ image2.jpg
        │   └ image2_mask.png
        ├── valid/
        │   ├ image1.jpg
        │   ├ image1_mask.png
        │   ├ image2.jpg
        │   └ image2_mask.png
        └── test/
            ├ image1.jpg
            ├ image1_mask.png
            ├ image2.jpg
            └ image2_mask.png        
```


For detailed information on dataset structure and annotation format, refer to the [Dataset Documentation](dataset/README.md).

## Code
The code is available in the `code` folder. It includes scripts for training, evaluation, and other relevant tasks.

### Dependencies
Ensure you have the following dependencies installed:
- List of dependencies with versions.
- Instructions on installing dependencies.

### Usage
1. **Training:** Train the model using the following command:
    ```bash
    python train.py --options
    ```
2. **Evaluation:** Evaluate the model's performance with:
    ```bash
    python evaluate.py --options
    ```
3. ...

For more detailed instructions on code usage and customization, refer to the [Code Documentation](code/README.md).

## Results
To demonstrate the dataset's utility, we evaluated state-of-the-art object detection algorithms on our proposed benchmark. Detailed results and comparisons can be found in the [Results](results/) folder.

## License
This dataset and code are released under the [MIT License](LICENSE).

## Citation
If you use this dataset or code in your research, please cite our paper. For citation details, refer to [CITATION.md](CITATION.md).

## Contact
For questions or inquiries, please contact:
- Maghsood Salimi (maghsood.salimi@mdu.se)
- Mohammad Loni (mohammad.loni@volvo.com)
- ...

## Acknowledgments
We would like to express our gratitude to...


