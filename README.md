# **Outline**
- [Diabetes Classification Project](#diabetes-classification-project)
    - [Project Description](#project-description)
    - [Conclusions](#conclusions)
- [Project Setup](#project-setup)
    - [Cloning the Repository](#cloning-the-repository)
    - [Virtual Environment and Dependencies](#virtual-environment-and-dependencies)
- [Web Service Deployment](#web-service-deployment)
    - [Virtual Environment Setup](#virtual-environment-setup)
    - [Docker Deployment](#docker-deployment)
    - [Cloud Deployment](#cloud-deployment)

# **Diabetes Classification Project**

<center><img src = "reports/figures/readme-image.png" width="970" height="270"/></center>



## Project Description

Diabetes is among the most prevalent chronic diseases in the United States, impacting millions of Americans each year. Diabetes is a chronic disease in which individuals lose the ability to effectively regulate levels of glucose in the blood, and can lead to reduced quality of life and life expectancy.

This project uses the Diabetes Health Indicators dataset, available on [kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/discussion). For detailed dataset insights, go to the `references/diabetes-indicator-dictionary.md` in the project folder.

The project seeks answer the following questions:

1. Which risk factors most strongly predict diabetes?

2. Can a subset of the risk factors to accurately predict whether an individual has diabetes?
   
## Conclusions

From the exploratory data analysis (EDA), we identified the possibility of dropping some redundant features from the dataset out of the initial 22, leading to two subsets: the 'small feature set' (with features agreed upon by all correlation metrics) and the 'large feature set' (a union of features selected by both Mutual Information and Pearson Correlation).

In the Model Training and Validation section we showed that both feature sets produced similar outcomes across Logistic Regression, Decision Tree, and Random Forest models, with only slight variances in accuracy and other metrics such as F1 Score, Precision, Recall, and AUC. The Features selected by the models shows to be a even smaller set of feature, with only five features, that demonstrates also a high predictive power with only a marginal difference from the other features set.

The challenge in reducing the false negative rate (individuals wrongly classified with diabetes) from the different models may be due the possible class imbalances of the features **CholCHeck**, **Stroke**, **HeartDiseaseorAttack**, **Vaggies**, **HvyAlcoholConsump**,  **AnyHealthcare**, **NoDocbcCost**, as noted previously in the EDA. One idea for improvement for the dataset in a future work may be focus on adding another tables from [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?select=diabetes_012_health_indicators_BRFSS2015.csv) that can be used complementary to the one used here. 

A important information from our analysis is the indicative that the features **BMI**, **GenHlth**, **Age**, **HighBP**, and **HighChol** are the most predictive risk factors for diabetes, consistent with established clinical insights. Our project successfully reduce the feature space from 22 possible risk factors to just a subset of five, facilitating a simplified predictive model without compromising accuracy. 

The **Random Forest model**, utilizing these five risk factors, proved to be the most balanced in terms of complexity and predictive capacity, and was selected for the final model deployment.


# **Project Setup**


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

# **Web Service Deployment**

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

## Cloud Deployment

The model was deployed to the cloud using Render, to interact with it via HTTP requests. Go to the jupyter notebook located in `notebooks/diabetes-indicators.ipynb`, run the following code cell at the section 5.2: 

```python
url_clound = "https://diabetes-indicators.onrender.com/predict"
diabetes_indicators = {'GenHlth':2, "BMI" :29.0,'Age': 1, 'HighChol':0, 'HighBP':0 }
requests.post(url_clound, json = diabetes_indicators).json()
```




