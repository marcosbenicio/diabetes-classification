# Diabetes Classification Project

## Project Description



# Project Setup


## Cloning the Repository

To get started with this project, clone the repository to your local machine:

```sh
git clone https://github.com/marcosbenicio/diabetes-classification.git
cd diabetes-classification
```


## Virtual Environment and Dependencies

To run the Jupyter notebooks in this project or a web service, is required to set up a virtual environment and install the dependencies.

1. Clone the repository to your local machine and navigate to the project directory (root):

    ```sh
    git clone https://github.com/marcosbenicio/diabetes-classification.git
    cd path/diabetes-classification
    ```

2. Ensure that `pipenv` is installed. If it is not installed, use the following command:

    ```sh
    pip install pipenv
    ```

3. In the root directory of the project, where the `Pipfile` is located, set up the virtual environment and install the dependencies:

    ```sh
    pipenv install
    ```

4. In the root directory of the project, where the `setup.py` is located, install the local package used on the notebook in editable mode. This will make the packages available within the Jupyter notebook:

    ```sh
    pipenv install -e .
    ```

    To see a list of the packages installed, you can use

    ```sh
    pipenv run pip freeze
    ```


5. To start the Jupyter notebook within the `pipenv` environment, you can run:

    ```sh
    pipenv shell
    jupyter notebook
    ```
    
And it will be started the Jupyter Notebook in the virtual environment context on the browser. Just go to the `notebooks/diabetes-indicators.ipynb` and run.

# Web Service Deployment

## Virtual Environment Setup

To host the service locally using the virtual environment, run the python script `src/models/predict_model.py`  to start the Flask application:

```sh
python3 src/models/predict_model.py
```

With the Flask application running, we can make HTTP requests to port 9696. For example, in the Jupyter notebook located in `notebooks/diabetes-indicators.ipynb`, run the following code at the section 5.2:

```python
url_local = "http://127.0.0.1:9696/predict"
diabetes_indicators = {'GenHlth':2, "BMI" :29.0,'Age': 1, 'HighChol':0, 'HighBP':0 }
requests.post(url_local, json = diabetes_indicators).json()
```

## Docker Deployment

1. Open a terminal or command prompt. Navigate to the directory containing the Dockerfile. Run the following command to build the Docker image named diabetes_prediction (you can give a different name to the image if you prefer):

        sudo docker build -t diabetes_prediction .

    or simple download the image from the Docker Hub ([Docker image](https://hub.docker.com/r/marcosbenicio/diabetes_prediction/tags)):

        sudo docker pull marcosbenicio/diabetes_prediction:latest

2. To list all the Docker images on your system and verify that the image is there, use:

     sudo docker images

3. After the image is built or pushed, run a container from it with the following command:

     sudo docker run -p 9696:9696 diabetes_prediction

    With the Flask application running inside Docker, we can make HTTP requests to port 9696. For example, in the Jupyter notebook located in `notebooks/diabetes-indicators.ipynb`, run the following code cell at the section 5.2: 

    ```python
    url_local = "http://127.0.0.1:9696/predict"
    diabetes_indicators = {'GenHlth':2, "BMI" :29.0,'Age': 1, 'HighChol':0, 'HighBP':0 }
    requests.post(url_local, json = diabetes_indicators).json()
    ```

### Cloud Deployment

The model was deployed to the cloud using Render, to interact with it via HTTP requests. Go to the jupyter notebook located in `notebooks/diabetes-indicators.ipynb`, run the following code cell at the section 5.2: 

```python
url_clound = "https://diabetes-indicators.onrender.com/predict"
diabetes_indicators = {'GenHlth':2, "BMI" :29.0,'Age': 1, 'HighChol':0, 'HighBP':0 }
requests.post(url_clound, json = diabetes_indicators).json()
```




