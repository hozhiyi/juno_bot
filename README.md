
# GYM Counting Device (GYMcD)

University of Malaya, WID3005 Intelligent Robotics, Semester 2 2022/2023

Lecturer: Dr. Zati Hakim Binti Azizul Hasan

Group Members: 

| NAME                           | MATRIC NUMBER  | FAVOURITE EXERCISE  |
|--------------------------------|----------------|---------------------|
| Lawrence Leroy Chieng Tze Yao  | S2018935       | Sit-ups             |
| Pang Chong Wen                 | U2005402       | Pilates             |
| Tan Jia Xuan                   | U2005407       | Hiking              |
| Ho Zhi Yi                      | U2005261       | Weight Training     |

## Project Details 

Posture is a prominent factor that makes or breaks an effective gym workout, be it cardio or hypertrophy. In the case of training solo, it is often challenging to fully self-assess if a person is having the correct pose, yet not everyone can afford a personal trainer.

GYMcD, being the proposed solution to the aforementioned predicament, hopes to deliver a personalised, efficient and tech-savvy experience to solo trainers. This device hopes to cover the needs of a gym assistant by capturing the real time exercises performed by the user and displaying via image and audio the current reps for each workout. Incorrect actions will not be counted into the workout to enforce integrity, just like how a human trainer would. 

The project offers to detect three kinds of posture - curl, push, and squat using a customized Posture Detection model. For each correct posture being made, Juno will count for the user and read the count out loud.

From a technical viewpoint, data captured from the JUNO lens is published to a rostopic to be subscribed by the AI machine learning algorithm, which then publishes the calculated results to the display node and text-to-speech node respectively. The model in this project was largely trained by its group members – minimum 150 reps per exercise in front of the camera. Beneath the hood, the pose estimation machine learning model determines the correctness of the detected posture.   

We hope that GYMcD can be a considerable alternative to budget solidarity individuals who desire a propped physique without needing to hire a trainer. With that inspiration in mind, we hope GYMcD could be one of the small steps of man in the giant leap of robotics.

## Project Demo 

Video link: https://www.youtube.com/watch?v=yrsDZUI_-h4 

## Getting Started

### Prerequisites
- Have the Juno bot with you! Our project requires Juno with vision and speech functions only.

### Clone this repo
```shell
cd [workspace]/src
git clone https://github.com/hozhiyi/juno_bot.git
```


### Model
- Create a folder named "models" inside of src folder.
- Download the model from our Google Drive - [LSTM Attention.h5](https://drive.google.com/file/d/1V4iLpShTlPDDALkrWmv_q0v3R33gB5kg/view?usp=sharing).
- Place the LSTM Attention.h5 in the models folder.

### Installation 

1. To create a virtual environment, we'll be using [pyenv](https://github.com/pyenv/pyenv), a tool that allows you to manage multiple versions of Python on your system. That's because our project requires both Python 2 and 3 running. Here's a guide to do so on Ubuntu/Debian.
- Build pyenv dependencies.
    ```shell
    sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
    libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl
    ```
- Install pyenv
    ```bash
    curl https://pyenv.run | bash
    ```
- Install Python.
    ```shell
    pyenv install -v 3.9.2
    ```
- Create virtual environment using pyenv
    ```shell
    pyenv virtualenv <python_version> <environment_name>
    ```
- Activate the python version
    ```shell
    pyenv local <environment_name>
    ```
- Activate the virtual environment
    ```shell
    pyenv activate <environment_name>
    ```
- Install the required packages using 
    ```shell
    pip install -r requirements.txt
    ```

### To build CVBridge 
To ensure that the Juno robot is able to read the CV2 image, a library named cv_bridge, a ROS package that provides a bridge between ROS image messages and OpenCV image formats is required. We'll be manually building it because catkin_make only compiles Python 2 scripts. Please follow these steps: 
```shell
mkdir -p ~/cvbridge_build_ws/src
cd ~/cvbridge_build_ws/src

cd ~/cvbridge_build_ws
catkin config -DPYTHON_EXECUTABLE=/home/mustar/.pyenv/versions/env-w15/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/home/mustar/.pyenv/versions/env-w15/lib/python3.9/site-packages
catkin config --install

catkin install -DCMAKE_BUILD_TYPE=Release -DSETUPTOOLS_DEB_LAYOUT=OFF

catkin build cv_bridge

source /home/mustar/cvbridge_build_ws/install/setup.bash [change to correct directory**] --extend
```

### To Start GYMcD

- Four terminals are required to run four sets of commands in parallel. 

1. Terminal 1
    - To build the project with Catkin and start roscore.
    ```shell
    cd ..
    catkin_make
    cd src/juno_bot/src
    chmod +x *.py
    roscore
    ```
2. Terminal 2
    - To start Juno's vision and capture your exercise posture.
    ```shell
    rosrun juno_bot camera_node.py 
    ```
3. Terminal 3
    - To start the text-to-speech node so that Juno can read the counts out loud.
    ```shell
    rosrun juno_bot gtts_node.py
    ```
4. Terminal 4
    - To launch the posture detection task.
    ```shell
    cd catkin_ws/src/juno_bot/
    rosrun juno_bot exercise_detection.py 
    ```
