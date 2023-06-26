# Intel Image Classification
![Intel Image Classification](https://github.com/Rohitkushwaha79/Intel_Image_Classification/assets/118690283/e162f7a3-699d-4e4a-ae4c-31cd32b33733)


This repository contains the code and resources for the Intel Image Classification project. The project aims to develop a deep learning model for accurately classifying images from the Intel Image Classification dataset.

## Dataset

The Intel Image Classification dataset is a collection of images representing various natural scenes, including buildings, forests, mountains, glaciers, and street views. The dataset is divided into training and testing sets, with labeled images for different scene categories.

The dataset can be obtained from [link-to-dataset](insert_dataset_link](https://storage.googleapis.com/kaggle-data-sets/111880/269359/upload/seg_pred.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230626%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230626T065408Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=78003ee388acba8ea64a3a71323d0cdebabac4537f8c91a6ddd37f8186b4ea7df75d7ab2141e7779867305e282a36efc3836fc9e952fecb328007b5f7d24b503e1fe7662b21b3ea0923d3f5e98efc202c2859adec32c2f7aba59c2ab5f512205e376b073357bd2feaa8105257b5a6ba5086ee5d4f637105dcae98c7dd483bfa93c8e0a3323a692ee503a5698e47ff97e0e2b3f4e20fc4547ae0495ff0cd06e6f78d965534d88d70c9c6d84b80902e31bc5389e80099da18051390ca446168542d1c4778101ab9fdf98985d7db87472f9b278443b25809fb5a24022a2eb30f3560674bfc54f67704c2908bc18eab0a054e055fe40742699b136731c902c0dcaf3)https://storage.googleapis.com/kaggle-data-sets/111880/269359/upload/seg_pred.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230626%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230626T065408Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=78003ee388acba8ea64a3a71323d0cdebabac4537f8c91a6ddd37f8186b4ea7df75d7ab2141e7779867305e282a36efc3836fc9e952fecb328007b5f7d24b503e1fe7662b21b3ea0923d3f5e98efc202c2859adec32c2f7aba59c2ab5f512205e376b073357bd2feaa8105257b5a6ba5086ee5d4f637105dcae98c7dd483bfa93c8e0a3323a692ee503a5698e47ff97e0e2b3f4e20fc4547ae0495ff0cd06e6f78d965534d88d70c9c6d84b80902e31bc5389e80099da18051390ca446168542d1c4778101ab9fdf98985d7db87472f9b278443b25809fb5a24022a2eb30f3560674bfc54f67704c2908bc18eab0a054e055fe40742699b136731c902c0dcaf3). Please refer to the dataset's documentation for more details on its structure and usage.

## Installation

To run the code in this repository, follow these steps:

1. Clone the repository:

2. Install the required dependencies:

3. Download and preprocess the dataset:
- Download the Intel Image Classification dataset from [link-to-dataset](insert_dataset_link](https://storage.googleapis.com/kaggle-data-sets/111880/269359/upload/seg_pred.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230626%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230626T065408Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=78003ee388acba8ea64a3a71323d0cdebabac4537f8c91a6ddd37f8186b4ea7df75d7ab2141e7779867305e282a36efc3836fc9e952fecb328007b5f7d24b503e1fe7662b21b3ea0923d3f5e98efc202c2859adec32c2f7aba59c2ab5f512205e376b073357bd2feaa8105257b5a6ba5086ee5d4f637105dcae98c7dd483bfa93c8e0a3323a692ee503a5698e47ff97e0e2b3f4e20fc4547ae0495ff0cd06e6f78d965534d88d70c9c6d84b80902e31bc5389e80099da18051390ca446168542d1c4778101ab9fdf98985d7db87472f9b278443b25809fb5a24022a2eb30f3560674bfc54f67704c2908bc18eab0a054e055fe40742699b136731c902c0dcaf3)https://storage.googleapis.com/kaggle-data-sets/111880/269359/upload/seg_pred.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230626%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230626T065408Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=78003ee388acba8ea64a3a71323d0cdebabac4537f8c91a6ddd37f8186b4ea7df75d7ab2141e7779867305e282a36efc3836fc9e952fecb328007b5f7d24b503e1fe7662b21b3ea0923d3f5e98efc202c2859adec32c2f7aba59c2ab5f512205e376b073357bd2feaa8105257b5a6ba5086ee5d4f637105dcae98c7dd483bfa93c8e0a3323a692ee503a5698e47ff97e0e2b3f4e20fc4547ae0495ff0cd06e6f78d965534d88d70c9c6d84b80902e31bc5389e80099da18051390ca446168542d1c4778101ab9fdf98985d7db87472f9b278443b25809fb5a24022a2eb30f3560674bfc54f67704c2908bc18eab0a054e055fe40742699b136731c902c0dcaf3).
- Preprocess the dataset using the provided preprocessing script (`preprocess.py`). Refer to the script's documentation for usage instructions.

## Usage

To train and evaluate the image classification model, follow these steps:

1. Run the training script:

2. Monitor the training progress and performance metrics.

3. Evaluate the trained model on the test dataset:

4. Generate predictions for new images using the trained model:

## Results

After training my model i got 92% validation accuracy .
![graph](https://github.com/Rohitkushwaha79/Intel_Image_Classification/assets/118690283/52890b06-8a35-4ac2-88c2-22c44f96a5ce)


## Contributing

Contributions to this project are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request. 

## Contact

For questions, feedback, or collaborations, please contact [Rohit Chandara Maurya] at [rm9093036@gmail.com].
