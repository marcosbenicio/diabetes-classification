# Diabetes Classification Project

## Project Description




## Virtual Environment Setup

To run this project on your system, follow the steps below after downloading the code:

If pipenv is not already installed on your system, use this command to install it:

    pip install pipenv

In the root of the project directory, install the required dependencies with:

    pipenv install

To activate the virtual environment created by pipenv, run:

    pipenv shell

Then, to host the service locally using the virtual environment, run the python script `src/models/predict_model.py` :

    python3 src/models/predict_model.py

With the Flask application running, we can make HTTP requests to port 9696. For example, in the Jupyter notebook located in `notebooks/diabetes-indicators.ipynb`, run the following code at the section 5.2:

```python
url_local = "http://127.0.0.1:9696/predict"
diabetes_indicators = {'GenHlth':2, "BMI" :29.0,'Age': 1, 'HighChol':0, 'HighBP':0 }
requests.post(url_local, json = diabetes_indicators).json()
```


## Docker Deployment

Open a terminal or command prompt. Navigate to the directory containing the Dockerfile. Run the following command to build the Docker image named diabetes_prediction (you can give a different name to the image if you prefer):

    $ sudo docker build -t diabetes_prediction .

or simple download the image from the Docker Hub ([Docker image](https://hub.docker.com/r/marcosbenicio/diabetes_prediction/tags)):

    sudo docker pull marcosbenicio/diabetes_prediction:latest

To list all the Docker images on your system and verify that the image is there, use:

    $ sudo docker images

After the image is built or pushed, run a container from it with the following command:

    $ sudo docker run -p 9696:9696 diabetes_prediction

With the Flask application running inside Docker, we can make HTTP requests to port 9696. For example, in the Jupyter notebook located in `notebooks/diabetes-indicators.ipynb`, run the following code cell at the section 5.2: 

```python
url_local = "http://127.0.0.1:9696/predict"
diabetes_indicators = {'GenHlth':2, "BMI" :29.0,'Age': 1, 'HighChol':0, 'HighBP':0 }
requests.post(url_local, json = diabetes_indicators).json()
```



### Run Docker On Clound

The model was deployed to the cloud using Render, to interact with it via HTTP requests. Go to the jupyter notebook located in `notebooks/diabetes-indicators.ipynb`, run the following code cell at the section 5.2: 

```python
url_clound = "https://diabetes-indicators.onrender.com/predict"
diabetes_indicators = {'GenHlth':2, "BMI" :29.0,'Age': 1, 'HighChol':0, 'HighBP':0 }
requests.post(url_clound, json = diabetes_indicators).json()
```




