# Real Time American Sign Language Recognition 

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
In this project, we build machine learning models to detect american sign language in real time.

### Built With



<!-- GETTING STARTED -->
## Getting Started
<!--
This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.
-->

### Prerequisites
<!--
This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```
-->


<!-- USAGE EXAMPLES -->
## Usage

https://user-images.githubusercontent.com/14941840/126101775-9fd5b9e1-2927-447f-9263-f1a8b8dc9671.mp4

<!--
Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.
-->

<!-- CONTRIBUTING -->
## Contributing

Any contributions you make are **greatly appreciated**. Please follow the steps below to make a contribution.

1. Fork the Project
2. Create your Feature Branch 
3. Commit your Changes 
4. Push to the Branch 
5. Open a Pull Request

<!-- Execution -->
## Execution

To execute the model please run
python gesture_detect.py 

The model takes 1 args - model. You can choose amongst 3 models
1. Landmark model
2. MobileNet
3. ResNet

Landmark is selected by default and has the highest accuracy. 100% !! 

To run the script using model of your choice run either 
1. python gesture_detect.py -m mobilenet 
2. python gesture_detect.py --model mobilenet
<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- CONTACT -->
## Contact

* Avinash - aavinash@ucdavis.edu - [@Linkedin](https://www.linkedin.com/in/baidyaavinash/)
* Ankita Sinha - asinha4@uci.edu -  [@Linkedin](https://www.linkedin.com/in/anki08/)

Project Link: [https://github.com/baiydaavi/HandGestureRecognition](https://github.com/baiydaavi/HandGestureRecognition)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [Kaggle ASL dataset](https://www.kaggle.com/grassknoted/asl-alphabet)
* [Github readme template](https://github.com/othneildrew/Best-README-Template)
