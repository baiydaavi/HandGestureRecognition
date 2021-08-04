# Real Time American Sign Language Recognition 

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#hand-landmark-model">Hand Landmark Model</a></li>
        <li><a href="#cnn-model">CNN Model</a></li>
      </ul>
    </li>
    <li><a href="#requirements">Requirements</a></li>
    <li><a href="#organization">Organization</a></li>
    <li><a href="#execution">Execution</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
In this project, we build machine learning models to detect american sign language in real time.

### Hand Landmark Model

### CNN Model
 

<!-- Requirements -->
## Requirements
<!--
This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```
-->


<!-- Organization -->
## Organization
This repository is organized into the following folders and files:-

Folders:
1. asl_alphabet_test - This folder contains the test images that were created by us to test the model on entirely unseen images.
2. asl_alphabet_train - This folder contains the train images that we obtained from kaggle.
3. reference_images - This folder contains the reference images for each sign for the purpose of showing a reference image for a given sign during real-time inference.
4. trained_cnn_models - This folder contains the trained CNN models.
5. trained_landmarks_models - This folder contains the trained hand landmark models.

Files:
1. ColabCNNTrain.ipynb - This colab file is used to train the CNN model.
2. ColabDNNLandmarksTrain.ipynb - This colab file is used to train the hand landmark model.
3. asl_recog.mov - This is a movie showing the real-time performance of the hand landmark model.
4. create_landmark_csv.py - This file is used to generate the hand landmarks data for both training and testing images.
5. gesture_detect.py - This file is used to run real-time inference using either the hand landmark model or the CNN model.
6. model.py - This file contains the model architecture for all the models.
7. utils.py - This file contains the utility functions required by the models.


<!-- Execution -->
## Execution

To execute the model please run - 
python gesture_detect.py.

The model takes 1 argument - model. You can choose amongst 3 model options
1. 'landmark' - denoting the hand landmark model.
2. 'mobilenet' - denoting the CNN model using MobileNet backbone.
3. 'resnet' - denoting the CNN model using ResNet backbone.

'landmark' is selected by default and has the highest accuracy. It works close to 100% of the time!! 

To run the script using model of your choice run either 
1. python gesture_detect.py -m model_name 
2. python gesture_detect.py --model model_name

https://user-images.githubusercontent.com/14941840/126101775-9fd5b9e1-2927-447f-9263-f1a8b8dc9671.mp4


<!-- CONTRIBUTING -->
## Contributing

Any contributions you make are **greatly appreciated**. Please follow the steps below to make a contribution.

1. Fork the Project
2. Create your Feature Branch 
3. Commit your Changes 
4. Push to the Branch 
5. Open a Pull Request


<!-- CONTACT -->
## Contact

* Avinash - aavinash@ucdavis.edu - [@Linkedin](https://www.linkedin.com/in/baidyaavinash/)
* Ankita Sinha - asinha4@uci.edu -  [@Linkedin](https://www.linkedin.com/in/anki08/)


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [Kaggle ASL dataset](https://www.kaggle.com/grassknoted/asl-alphabet)
* [Github readme template](https://github.com/othneildrew/Best-README-Template)
