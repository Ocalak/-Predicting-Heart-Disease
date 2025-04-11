{
  "metadata": {
    "kernelspec": {
      "language": "python",
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "pygments_lexer": "ipython3",
      "nbconvert_exporter": "python",
      "version": "3.6.4",
      "file_extension": ".py",
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "name": "python",
      "mimetype": "text/x-python"
    },
    "colab": {
      "name": "A Guide to any Classification Problem",
      "provenance": [],
      "include_colab_link": true
    }
  },
  "nbformat_minor": 0,
  "nbformat": 4,
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Ocalak/-Predicting-Heart-Disease/blob/main/A_Guide_to_any_Classification_Problem.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "source": [
        "# THEN FEEL FREE TO DELETE THIS CELL.\n",
        "# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON\n",
        "# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR\n",
        "# NOTEBOOK.\n",
        "import kagglehub\n",
        "andrewmvd_heart_failure_clinical_data_path = kagglehub.dataset_download('andrewmvd/heart-failure-clinical-data')\n",
        "fedesoriano_heart_failure_prediction_path = kagglehub.dataset_download('fedesoriano/heart-failure-prediction')\n",
        "\n",
        "print('Data source import complete.')\n"
      ],
      "metadata": {
        "id": "8UOxxFnucMcS",
        "outputId": "1408ad7e-d570-4116-c2e7-eb6bfdc1ed35",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "cell_type": "code",
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Downloading from https://www.kaggle.com/api/v1/datasets/download/andrewmvd/heart-failure-clinical-data?dataset_version_number=1...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 3.97k/3.97k [00:00<00:00, 1.50MB/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Extracting files...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Data source import complete.\n"
          ]
        }
      ],
      "execution_count": 1
    },
    {
      "cell_type": "markdown",
      "source": [
        "# HI!!!\n",
        "\n",
        "<img src=\"https://media.giphy.com/media/WpIPS0DWNpMm4kfMVr/giphy.gif\">\n",
        "\n",
        "## Lets get started !!!\n",
        "\n",
        "# When dealing with machine learning problems, there are generally two types of data (and machine learning models):\n",
        "- Supervised data: always has one or multiple targets associated with it.\n",
        "- Unsupervised data: does not have any target variable.\n",
        "\n",
        "A supervised problem is considerably easier to tackle than an unsupervised one. A problem in which we are required to predict a value is known as a supervised problem. For example, if the problem is to predict house prices given historical house prices, with features like presence of a hospital, school or supermarket, distance to nearest public transport, etc. is a upervised problem. Similarly, when we are provided with images of cats and dogs, and we know beforehand which ones are cats and which ones are dogs, and if the task is to create a model which predicts whether a provided image is of a cat or a dog, the problem is considered to be\n",
        "supervised.\n",
        "\n",
        "Here in this Dataset we have a Supervised Machine Learning Problem, For Heart Failure Prediction"
      ],
      "metadata": {
        "id": "YsOfvk_FcMcU"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Importing all the libraries needed"
      ],
      "metadata": {
        "id": "xhIS653FcMcV"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import warnings\n",
        "import seaborn as sns\n",
        "import matplotlib.pyplot as plt\n",
        "import plotly.express as px\n",
        "warnings.filterwarnings(\"ignore\")\n",
        "pd.set_option(\"display.max_rows\",None)\n",
        "from sklearn import preprocessing\n",
        "import matplotlib\n",
        "matplotlib.style.use('ggplot')\n",
        "from sklearn.preprocessing import LabelEncoder"
      ],
      "metadata": {
        "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
        "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
        "execution": {
          "iopub.status.busy": "2021-12-13T06:24:51.202064Z",
          "iopub.execute_input": "2021-12-13T06:24:51.202723Z",
          "iopub.status.idle": "2021-12-13T06:24:53.617972Z",
          "shell.execute_reply.started": "2021-12-13T06:24:51.202619Z",
          "shell.execute_reply": "2021-12-13T06:24:53.617111Z"
        },
        "trusted": true,
        "id": "0scJaSOocMcV"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Context\n",
        "\n",
        "<img src=\"https://media.giphy.com/media/8cBhJBU2wlq6H6qY4W/giphy.gif\">\n",
        "\n",
        "Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Four out of 5CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. Heart failure is a common event caused by CVDs and this dataset contains 11 features that can be used to predict a possible heart disease.\n",
        "\n",
        "People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-11T18:28:55.671258Z",
          "iopub.execute_input": "2021-12-11T18:28:55.671715Z",
          "iopub.status.idle": "2021-12-11T18:28:56.739043Z",
          "shell.execute_reply.started": "2021-12-11T18:28:55.671682Z",
          "shell.execute_reply": "2021-12-11T18:28:56.738347Z"
        },
        "id": "vHKUfltCcMcV"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df=pd.read_csv(\"/kaggle/input/heart-failure-prediction/heart.csv\")\n",
        "df.head()"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:24:53.620027Z",
          "iopub.execute_input": "2021-12-13T06:24:53.620345Z",
          "iopub.status.idle": "2021-12-13T06:24:53.663223Z",
          "shell.execute_reply.started": "2021-12-13T06:24:53.620313Z",
          "shell.execute_reply": "2021-12-13T06:24:53.662567Z"
        },
        "trusted": true,
        "id": "DcJYkNlecMcV",
        "outputId": "47f73948-c701-4fb1-c94c-2f8e9de759f6",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 224
        }
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   Age Sex ChestPainType  RestingBP  Cholesterol  FastingBS RestingECG  MaxHR  \\\n",
              "0   40   M           ATA        140          289          0     Normal    172   \n",
              "1   49   F           NAP        160          180          0     Normal    156   \n",
              "2   37   M           ATA        130          283          0         ST     98   \n",
              "3   48   F           ASY        138          214          0     Normal    108   \n",
              "4   54   M           NAP        150          195          0     Normal    122   \n",
              "\n",
              "  ExerciseAngina  Oldpeak ST_Slope  HeartDisease  \n",
              "0              N      0.0       Up             0  \n",
              "1              N      1.0     Flat             1  \n",
              "2              N      0.0       Up             0  \n",
              "3              Y      1.5     Flat             1  \n",
              "4              N      0.0       Up             0  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-676b020d-2cc2-41f5-8c9d-168dc9d5ed24\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Age</th>\n",
              "      <th>Sex</th>\n",
              "      <th>ChestPainType</th>\n",
              "      <th>RestingBP</th>\n",
              "      <th>Cholesterol</th>\n",
              "      <th>FastingBS</th>\n",
              "      <th>RestingECG</th>\n",
              "      <th>MaxHR</th>\n",
              "      <th>ExerciseAngina</th>\n",
              "      <th>Oldpeak</th>\n",
              "      <th>ST_Slope</th>\n",
              "      <th>HeartDisease</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>40</td>\n",
              "      <td>M</td>\n",
              "      <td>ATA</td>\n",
              "      <td>140</td>\n",
              "      <td>289</td>\n",
              "      <td>0</td>\n",
              "      <td>Normal</td>\n",
              "      <td>172</td>\n",
              "      <td>N</td>\n",
              "      <td>0.0</td>\n",
              "      <td>Up</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>49</td>\n",
              "      <td>F</td>\n",
              "      <td>NAP</td>\n",
              "      <td>160</td>\n",
              "      <td>180</td>\n",
              "      <td>0</td>\n",
              "      <td>Normal</td>\n",
              "      <td>156</td>\n",
              "      <td>N</td>\n",
              "      <td>1.0</td>\n",
              "      <td>Flat</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>37</td>\n",
              "      <td>M</td>\n",
              "      <td>ATA</td>\n",
              "      <td>130</td>\n",
              "      <td>283</td>\n",
              "      <td>0</td>\n",
              "      <td>ST</td>\n",
              "      <td>98</td>\n",
              "      <td>N</td>\n",
              "      <td>0.0</td>\n",
              "      <td>Up</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>48</td>\n",
              "      <td>F</td>\n",
              "      <td>ASY</td>\n",
              "      <td>138</td>\n",
              "      <td>214</td>\n",
              "      <td>0</td>\n",
              "      <td>Normal</td>\n",
              "      <td>108</td>\n",
              "      <td>Y</td>\n",
              "      <td>1.5</td>\n",
              "      <td>Flat</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>54</td>\n",
              "      <td>M</td>\n",
              "      <td>NAP</td>\n",
              "      <td>150</td>\n",
              "      <td>195</td>\n",
              "      <td>0</td>\n",
              "      <td>Normal</td>\n",
              "      <td>122</td>\n",
              "      <td>N</td>\n",
              "      <td>0.0</td>\n",
              "      <td>Up</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-676b020d-2cc2-41f5-8c9d-168dc9d5ed24')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-676b020d-2cc2-41f5-8c9d-168dc9d5ed24 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-676b020d-2cc2-41f5-8c9d-168dc9d5ed24');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-a70bd333-f7fa-41f9-b55a-b757d5af8180\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-a70bd333-f7fa-41f9-b55a-b757d5af8180')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-a70bd333-f7fa-41f9-b55a-b757d5af8180 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "df",
              "summary": "{\n  \"name\": \"df\",\n  \"rows\": 918,\n  \"fields\": [\n    {\n      \"column\": \"Age\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 9,\n        \"min\": 28,\n        \"max\": 77,\n        \"num_unique_values\": 50,\n        \"samples\": [\n          44,\n          68,\n          66\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Sex\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 2,\n        \"samples\": [\n          \"F\",\n          \"M\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"ChestPainType\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 4,\n        \"samples\": [\n          \"NAP\",\n          \"TA\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"RestingBP\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 18,\n        \"min\": 0,\n        \"max\": 200,\n        \"num_unique_values\": 67,\n        \"samples\": [\n          165,\n          118\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Cholesterol\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 109,\n        \"min\": 0,\n        \"max\": 603,\n        \"num_unique_values\": 222,\n        \"samples\": [\n          305,\n          321\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"FastingBS\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"RestingECG\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 3,\n        \"samples\": [\n          \"Normal\",\n          \"ST\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"MaxHR\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 25,\n        \"min\": 60,\n        \"max\": 202,\n        \"num_unique_values\": 119,\n        \"samples\": [\n          132,\n          157\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"ExerciseAngina\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 2,\n        \"samples\": [\n          \"Y\",\n          \"N\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Oldpeak\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1.0665701510493233,\n        \"min\": -2.6,\n        \"max\": 6.2,\n        \"num_unique_values\": 53,\n        \"samples\": [\n          1.3,\n          0.6\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"ST_Slope\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 3,\n        \"samples\": [\n          \"Up\",\n          \"Flat\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"HeartDisease\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 3
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "The describe() function in pandas is very handy in getting various summary statistics.This function returns the count, mean, standard deviation, minimum and maximum values and the quantiles of the data."
      ],
      "metadata": {
        "id": "fW173hxUcMcV"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df.dtypes"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:24:53.664122Z",
          "iopub.execute_input": "2021-12-13T06:24:53.664766Z",
          "iopub.status.idle": "2021-12-13T06:24:53.672573Z",
          "shell.execute_reply.started": "2021-12-13T06:24:53.664733Z",
          "shell.execute_reply": "2021-12-13T06:24:53.671665Z"
        },
        "trusted": true,
        "id": "XX3cI504cMcV",
        "outputId": "53fb6ba0-0e3a-4cc4-bc54-4216251faa33",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 455
        }
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Age                 int64\n",
              "Sex                object\n",
              "ChestPainType      object\n",
              "RestingBP           int64\n",
              "Cholesterol         int64\n",
              "FastingBS           int64\n",
              "RestingECG         object\n",
              "MaxHR               int64\n",
              "ExerciseAngina     object\n",
              "Oldpeak           float64\n",
              "ST_Slope           object\n",
              "HeartDisease        int64\n",
              "dtype: object"
            ],
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>0</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>Age</th>\n",
              "      <td>int64</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Sex</th>\n",
              "      <td>object</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>ChestPainType</th>\n",
              "      <td>object</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>RestingBP</th>\n",
              "      <td>int64</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Cholesterol</th>\n",
              "      <td>int64</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>FastingBS</th>\n",
              "      <td>int64</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>RestingECG</th>\n",
              "      <td>object</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>MaxHR</th>\n",
              "      <td>int64</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>ExerciseAngina</th>\n",
              "      <td>object</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Oldpeak</th>\n",
              "      <td>float64</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>ST_Slope</th>\n",
              "      <td>object</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>HeartDisease</th>\n",
              "      <td>int64</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div><br><label><b>dtype:</b> object</label>"
            ]
          },
          "metadata": {},
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "As we can see the string data in the dataframe is in the form of object, we need to convert it back to string to work on it"
      ],
      "metadata": {
        "id": "bFo4kd0YcMcV"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "string_col = df.select_dtypes(include=\"object\").columns\n",
        "df[string_col]=df[string_col].astype(\"string\")"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:24:53.674934Z",
          "iopub.execute_input": "2021-12-13T06:24:53.675538Z",
          "iopub.status.idle": "2021-12-13T06:24:53.70147Z",
          "shell.execute_reply.started": "2021-12-13T06:24:53.675495Z",
          "shell.execute_reply": "2021-12-13T06:24:53.700455Z"
        },
        "trusted": true,
        "id": "4VBEaPTscMcW"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df.dtypes"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:24:53.702986Z",
          "iopub.execute_input": "2021-12-13T06:24:53.703732Z",
          "iopub.status.idle": "2021-12-13T06:24:53.714413Z",
          "shell.execute_reply.started": "2021-12-13T06:24:53.703691Z",
          "shell.execute_reply": "2021-12-13T06:24:53.713513Z"
        },
        "trusted": true,
        "id": "7nyTuZF7cMcW",
        "outputId": "a222345a-6ce9-46bf-d6a1-a51daf8ae313",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 455
        }
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Age                        int64\n",
              "Sex               string[python]\n",
              "ChestPainType     string[python]\n",
              "RestingBP                  int64\n",
              "Cholesterol                int64\n",
              "FastingBS                  int64\n",
              "RestingECG        string[python]\n",
              "MaxHR                      int64\n",
              "ExerciseAngina    string[python]\n",
              "Oldpeak                  float64\n",
              "ST_Slope          string[python]\n",
              "HeartDisease               int64\n",
              "dtype: object"
            ],
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>0</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>Age</th>\n",
              "      <td>int64</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Sex</th>\n",
              "      <td>string[python]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>ChestPainType</th>\n",
              "      <td>string[python]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>RestingBP</th>\n",
              "      <td>int64</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Cholesterol</th>\n",
              "      <td>int64</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>FastingBS</th>\n",
              "      <td>int64</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>RestingECG</th>\n",
              "      <td>string[python]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>MaxHR</th>\n",
              "      <td>int64</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>ExerciseAngina</th>\n",
              "      <td>string[python]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Oldpeak</th>\n",
              "      <td>float64</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>ST_Slope</th>\n",
              "      <td>string[python]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>HeartDisease</th>\n",
              "      <td>int64</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div><br><label><b>dtype:</b> object</label>"
            ]
          },
          "metadata": {},
          "execution_count": 6
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "So, as we can see here the object data has been converted to string"
      ],
      "metadata": {
        "id": "PK3DDGBCcMcW"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Getting the categorical columns"
      ],
      "metadata": {
        "id": "w2vKf8nicMcW"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "string_col=df.select_dtypes(\"string\").columns.to_list()"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:24:53.715969Z",
          "iopub.execute_input": "2021-12-13T06:24:53.716596Z",
          "iopub.status.idle": "2021-12-13T06:24:53.724925Z",
          "shell.execute_reply.started": "2021-12-13T06:24:53.716556Z",
          "shell.execute_reply": "2021-12-13T06:24:53.724357Z"
        },
        "trusted": true,
        "id": "u75bYiPDcMcW"
      },
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "num_col=df.columns.to_list()\n",
        "#print(num_col)\n",
        "for col in string_col:\n",
        "    num_col.remove(col)\n",
        "num_col.remove(\"HeartDisease\")"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:24:53.725785Z",
          "iopub.execute_input": "2021-12-13T06:24:53.72644Z",
          "iopub.status.idle": "2021-12-13T06:24:53.73486Z",
          "shell.execute_reply.started": "2021-12-13T06:24:53.726402Z",
          "shell.execute_reply": "2021-12-13T06:24:53.734241Z"
        },
        "trusted": true,
        "id": "3zAHx5agcMcW"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df.describe().T"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:24:53.736525Z",
          "iopub.execute_input": "2021-12-13T06:24:53.737118Z",
          "iopub.status.idle": "2021-12-13T06:24:53.781784Z",
          "shell.execute_reply.started": "2021-12-13T06:24:53.737067Z",
          "shell.execute_reply": "2021-12-13T06:24:53.781156Z"
        },
        "trusted": true,
        "id": "167rYH31cMcW",
        "outputId": "d0d4013a-c936-41c7-aad2-88f82454f546",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 266
        }
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "              count        mean         std   min     25%    50%    75%    max\n",
              "Age           918.0   53.510893    9.432617  28.0   47.00   54.0   60.0   77.0\n",
              "RestingBP     918.0  132.396514   18.514154   0.0  120.00  130.0  140.0  200.0\n",
              "Cholesterol   918.0  198.799564  109.384145   0.0  173.25  223.0  267.0  603.0\n",
              "FastingBS     918.0    0.233115    0.423046   0.0    0.00    0.0    0.0    1.0\n",
              "MaxHR         918.0  136.809368   25.460334  60.0  120.00  138.0  156.0  202.0\n",
              "Oldpeak       918.0    0.887364    1.066570  -2.6    0.00    0.6    1.5    6.2\n",
              "HeartDisease  918.0    0.553377    0.497414   0.0    0.00    1.0    1.0    1.0"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-20e59420-6a89-4eb6-aa76-5c50ac4a1cd8\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>count</th>\n",
              "      <th>mean</th>\n",
              "      <th>std</th>\n",
              "      <th>min</th>\n",
              "      <th>25%</th>\n",
              "      <th>50%</th>\n",
              "      <th>75%</th>\n",
              "      <th>max</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>Age</th>\n",
              "      <td>918.0</td>\n",
              "      <td>53.510893</td>\n",
              "      <td>9.432617</td>\n",
              "      <td>28.0</td>\n",
              "      <td>47.00</td>\n",
              "      <td>54.0</td>\n",
              "      <td>60.0</td>\n",
              "      <td>77.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>RestingBP</th>\n",
              "      <td>918.0</td>\n",
              "      <td>132.396514</td>\n",
              "      <td>18.514154</td>\n",
              "      <td>0.0</td>\n",
              "      <td>120.00</td>\n",
              "      <td>130.0</td>\n",
              "      <td>140.0</td>\n",
              "      <td>200.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Cholesterol</th>\n",
              "      <td>918.0</td>\n",
              "      <td>198.799564</td>\n",
              "      <td>109.384145</td>\n",
              "      <td>0.0</td>\n",
              "      <td>173.25</td>\n",
              "      <td>223.0</td>\n",
              "      <td>267.0</td>\n",
              "      <td>603.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>FastingBS</th>\n",
              "      <td>918.0</td>\n",
              "      <td>0.233115</td>\n",
              "      <td>0.423046</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.00</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>MaxHR</th>\n",
              "      <td>918.0</td>\n",
              "      <td>136.809368</td>\n",
              "      <td>25.460334</td>\n",
              "      <td>60.0</td>\n",
              "      <td>120.00</td>\n",
              "      <td>138.0</td>\n",
              "      <td>156.0</td>\n",
              "      <td>202.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Oldpeak</th>\n",
              "      <td>918.0</td>\n",
              "      <td>0.887364</td>\n",
              "      <td>1.066570</td>\n",
              "      <td>-2.6</td>\n",
              "      <td>0.00</td>\n",
              "      <td>0.6</td>\n",
              "      <td>1.5</td>\n",
              "      <td>6.2</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>HeartDisease</th>\n",
              "      <td>918.0</td>\n",
              "      <td>0.553377</td>\n",
              "      <td>0.497414</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.00</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1.0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-20e59420-6a89-4eb6-aa76-5c50ac4a1cd8')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-20e59420-6a89-4eb6-aa76-5c50ac4a1cd8 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-20e59420-6a89-4eb6-aa76-5c50ac4a1cd8');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-1c624af8-ba13-47c3-8ca8-553e416ef16c\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-1c624af8-ba13-47c3-8ca8-553e416ef16c')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-1c624af8-ba13-47c3-8ca8-553e416ef16c button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "summary": "{\n  \"name\": \"df\",\n  \"rows\": 7,\n  \"fields\": [\n    {\n      \"column\": \"count\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0.0,\n        \"min\": 918.0,\n        \"max\": 918.0,\n        \"num_unique_values\": 1,\n        \"samples\": [\n          918.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"mean\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 81.16595478381392,\n        \"min\": 0.23311546840958605,\n        \"max\": 198.7995642701525,\n        \"num_unique_values\": 7,\n        \"samples\": [\n          53.510893246187365\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"std\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 39.088770004754714,\n        \"min\": 0.423045624739302,\n        \"max\": 109.38414455220337,\n        \"num_unique_values\": 7,\n        \"samples\": [\n          9.432616506732007\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"min\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 23.625127865615177,\n        \"min\": -2.6,\n        \"max\": 60.0,\n        \"num_unique_values\": 4,\n        \"samples\": [\n          0.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"25%\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 71.6043818491578,\n        \"min\": 0.0,\n        \"max\": 173.25,\n        \"num_unique_values\": 4,\n        \"samples\": [\n          120.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"50%\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 87.47257529403961,\n        \"min\": 0.0,\n        \"max\": 223.0,\n        \"num_unique_values\": 7,\n        \"samples\": [\n          54.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"75%\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 102.4169233597465,\n        \"min\": 0.0,\n        \"max\": 267.0,\n        \"num_unique_values\": 7,\n        \"samples\": [\n          60.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"max\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 216.2527753744281,\n        \"min\": 1.0,\n        \"max\": 603.0,\n        \"num_unique_values\": 6,\n        \"samples\": [\n          77.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# The Attributess include:\n",
        "- Age: age of the patient [years]\n",
        "- Sex: sex of the patient [M: Male, F: Female]\n",
        "- ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]\n",
        "- RestingBP: resting blood pressure [mm Hg]\n",
        "- Cholesterol: serum cholesterol [mm/dl]\n",
        "- FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]\n",
        "- RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]\n",
        "- MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]\n",
        "- ExerciseAngina: exercise-induced angina [Y: Yes, N: No]\n",
        "- Oldpeak: oldpeak = ST [Numeric value measured in depression]\n",
        "- ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]\n",
        "- HeartDisease: output class [1: heart disease, 0: Normal]"
      ],
      "metadata": {
        "id": "ldZQZEWOcMcW"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Exploratory Data Analysis\n",
        "\n",
        "<img src=\"https://media.giphy.com/media/HUplkVCPY7jTW/giphy.gif\">\n",
        "\n",
        "## First Question should be why do we need this ??\n",
        "\n",
        "Out Come of this phase is as given below :\n",
        "\n",
        "- Understanding the given dataset and helps clean up the given dataset.\n",
        "- It gives you a clear picture of the features and the relationships between them.\n",
        "- Providing guidelines for essential variables and leaving behind/removing non-essential variables.\n",
        "- Handling Missing values or human error.\n",
        "- Identifying outliers.\n",
        "- EDA process would be maximizing insights of a dataset.\n",
        "- This process is time-consuming but very effective,"
      ],
      "metadata": {
        "id": "hAML197TcMcW"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Correlation Matrix\n",
        "### Its necessary to remove correlated variables to improve your model.One can find correlations using pandas “.corr()” function and can visualize the correlation matrix using plotly express.\n",
        "- Lighter shades represents positive correlation\n",
        "- Darker shades represents negative correlation"
      ],
      "metadata": {
        "id": "bU0tcdTHcMcX"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "px.imshow(df.corr(),title=\"Correlation Plot of the Heat Failure Prediction\")"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:24:53.782668Z",
          "iopub.execute_input": "2021-12-13T06:24:53.783262Z",
          "iopub.status.idle": "2021-12-13T06:24:54.91476Z",
          "shell.execute_reply.started": "2021-12-13T06:24:53.783224Z",
          "shell.execute_reply": "2021-12-13T06:24:54.913855Z"
        },
        "trusted": true,
        "id": "RBPM-a3HcMcX",
        "outputId": "15504606-d903-4e45-88ce-e3a5dbe8736a",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 282
        }
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "error",
          "ename": "ValueError",
          "evalue": "could not convert string to float: 'Up'",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mValueError\u001b[0m                                Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-10-4e19d90307f8>\u001b[0m in \u001b[0;36m<cell line: 0>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mpx\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mimshow\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdf\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcorr\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mtitle\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m\"Correlation Plot of the Heat Failure Prediction\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py\u001b[0m in \u001b[0;36mcorr\u001b[0;34m(self, method, min_periods, numeric_only)\u001b[0m\n\u001b[1;32m  11047\u001b[0m         \u001b[0mcols\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mdata\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m  11048\u001b[0m         \u001b[0midx\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcols\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcopy\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m> 11049\u001b[0;31m         \u001b[0mmat\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mdata\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mto_numpy\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdtype\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mfloat\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mna_value\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mnan\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcopy\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mFalse\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m  11050\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m  11051\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mmethod\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;34m\"pearson\"\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py\u001b[0m in \u001b[0;36mto_numpy\u001b[0;34m(self, dtype, copy, na_value)\u001b[0m\n\u001b[1;32m   1991\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mdtype\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1992\u001b[0m             \u001b[0mdtype\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdtype\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdtype\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1993\u001b[0;31m         \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_mgr\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mas_array\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdtype\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mdtype\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcopy\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mcopy\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mna_value\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mna_value\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1994\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mresult\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdtype\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mdtype\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1995\u001b[0m             \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0masarray\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mresult\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mdtype\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py\u001b[0m in \u001b[0;36mas_array\u001b[0;34m(self, dtype, copy, na_value)\u001b[0m\n\u001b[1;32m   1692\u001b[0m                 \u001b[0marr\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mflags\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mwriteable\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mFalse\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1693\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1694\u001b[0;31m             \u001b[0marr\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_interleave\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdtype\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mdtype\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mna_value\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mna_value\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1695\u001b[0m             \u001b[0;31m# The underlying data was copied within _interleave, so no need\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1696\u001b[0m             \u001b[0;31m# to further copy if copy=True or setting na_value\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py\u001b[0m in \u001b[0;36m_interleave\u001b[0;34m(self, dtype, na_value)\u001b[0m\n\u001b[1;32m   1745\u001b[0m                 \u001b[0;31m# error: Item \"ndarray\" of \"Union[ndarray, ExtensionArray]\" has no\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1746\u001b[0m                 \u001b[0;31m# attribute \"to_numpy\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1747\u001b[0;31m                 arr = blk.values.to_numpy(  # type: ignore[union-attr]\n\u001b[0m\u001b[1;32m   1748\u001b[0m                     \u001b[0mdtype\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mdtype\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1749\u001b[0m                     \u001b[0mna_value\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mna_value\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/numpy_.py\u001b[0m in \u001b[0;36mto_numpy\u001b[0;34m(self, dtype, copy, na_value)\u001b[0m\n\u001b[1;32m    503\u001b[0m             \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_ndarray\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    504\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 505\u001b[0;31m         \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0masarray\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mresult\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mdtype\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    506\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    507\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mcopy\u001b[0m \u001b[0;32mand\u001b[0m \u001b[0mresult\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_ndarray\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mValueError\u001b[0m: could not convert string to float: 'Up'"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Here we can see Heart Disease has a high negative correlation with \"MaxHR\" and somewhat negative correlation wiht \"Cholesterol\", where as here positive correatlation with \"Oldpeak\",\"FastingBS\" and \"RestingBP\""
      ],
      "metadata": {
        "id": "UcH962YqcMcX"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Shows the Distribution of Heat Diseases with respect to male and female\n",
        "fig=px.histogram(df,\n",
        "                 x=\"HeartDisease\",\n",
        "                 color=\"Sex\",\n",
        "                 hover_data=df.columns,\n",
        "                 title=\"Distribution of Heart Diseases\",\n",
        "                 barmode=\"group\")\n",
        "fig.show()"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:24:54.917588Z",
          "iopub.execute_input": "2021-12-13T06:24:54.917901Z",
          "iopub.status.idle": "2021-12-13T06:24:55.051245Z",
          "shell.execute_reply.started": "2021-12-13T06:24:54.917868Z",
          "shell.execute_reply": "2021-12-13T06:24:55.050353Z"
        },
        "trusted": true,
        "id": "u7RdPUo4cMcX",
        "outputId": "4167a656-ee61-4180-dc71-b2dd4adc09a5",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 542
        }
      },
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/html": [
              "<html>\n",
              "<head><meta charset=\"utf-8\" /></head>\n",
              "<body>\n",
              "    <div>            <script src=\"https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG\"></script><script type=\"text/javascript\">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: \"STIX-Web\"}});}</script>                <script type=\"text/javascript\">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>\n",
              "        <script charset=\"utf-8\" src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>                <div id=\"30cbf971-3723-48c4-8b0e-ebd18d4125ce\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"30cbf971-3723-48c4-8b0e-ebd18d4125ce\")) {                    Plotly.newPlot(                        \"30cbf971-3723-48c4-8b0e-ebd18d4125ce\",                        [{\"alignmentgroup\":\"True\",\"bingroup\":\"x\",\"hovertemplate\":\"Sex=M\\u003cbr\\u003eHeartDisease=%{x}\\u003cbr\\u003ecount=%{y}\\u003cextra\\u003e\\u003c\\u002fextra\\u003e\",\"legendgroup\":\"M\",\"marker\":{\"color\":\"#636efa\",\"pattern\":{\"shape\":\"\"}},\"name\":\"M\",\"offsetgroup\":\"M\",\"orientation\":\"v\",\"showlegend\":true,\"x\":[0,0,0,0,0,1,1,0,1,1,1,1,0,1,0,0,0,0,0,1,0,1,1,0,1,0,0,1,0,0,0,0,1,1,0,1,1,0,1,0,0,1,0,1,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,1,1,0,0,1,1,0,0,0,0,1,0,1,1,0,0,0,0,1,0,0,1,1,0,0,0,0,0,1,1,1,1,0,1,1,1,1,1,0,0,0,0,1,0,0,0,0,0,1,1,0,1,0,1,1,0,0,1,1,0,0,0,0,0,0,0,1,1,1,0,0,1,0,1,0,1,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,1,0,1,1,0,0,0,1,0,0,1,0,1,0,0,0,0,0,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,0,0,1,0,0,1,1,1,0,1,0,1,0,1,0,0,1,1,0,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,0,1,0,1,1,0,1,1,1,1,0,1,1,0,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,0,1,1,1,0,1,0,1,1,0,1,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,0,1,1,1,1,1,1,0,1,1,1,0,1,1,0,1,0,1,1,0,1,1,1,1,0,1,1,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,1,0,1,0,1,1,1,0,0,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1,0,0,1,1,1,1,1,0,1,1,0,1,1,1,0,1,1,1,1,0,1,0,1,1,1,0,0,1,1,1,0,0,0,1,1,1,0,0,1,0,0,0,1,1,0,1,1,1,1,1,0,0,1,0,0,1,0,1,1,1,1,0,1,1,0,0,0,1,0,1,1,0,1,0,0,1,1,1,0,0,0,0,0,1,0,1,1,1,1,0,1,1,1,1,1,0,1,0,0,1,1,1,1,1,0,1,0,1,1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,0,1,0,0,1,0,1,0,0,1,0,1,1,0,1,1,1,0,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,1,1,1,0,0,1,0,1,0,1,0,0,0,0,1,1,0,1,0,0,1,0,1,0,1,0,0,1,0,1,1,1,1,1,0,1,0,0,0,1,1,0,1,1,0,1,0,0,0,1,0,1,1,1,0,1,1,0,0,1,1,0,0,0,1,1,1,0,1,1,1,1,0],\"xaxis\":\"x\",\"yaxis\":\"y\",\"type\":\"histogram\"},{\"alignmentgroup\":\"True\",\"bingroup\":\"x\",\"hovertemplate\":\"Sex=F\\u003cbr\\u003eHeartDisease=%{x}\\u003cbr\\u003ecount=%{y}\\u003cextra\\u003e\\u003c\\u002fextra\\u003e\",\"legendgroup\":\"F\",\"marker\":{\"color\":\"#EF553B\",\"pattern\":{\"shape\":\"\"}},\"name\":\"F\",\"offsetgroup\":\"F\",\"orientation\":\"v\",\"showlegend\":true,\"x\":[1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,1,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1],\"xaxis\":\"x\",\"yaxis\":\"y\",\"type\":\"histogram\"}],                        {\"template\":{\"data\":{\"histogram2dcontour\":[{\"type\":\"histogram2dcontour\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"choropleth\":[{\"type\":\"choropleth\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}],\"histogram2d\":[{\"type\":\"histogram2d\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"heatmap\":[{\"type\":\"heatmap\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"heatmapgl\":[{\"type\":\"heatmapgl\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"contourcarpet\":[{\"type\":\"contourcarpet\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}],\"contour\":[{\"type\":\"contour\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"surface\":[{\"type\":\"surface\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"mesh3d\":[{\"type\":\"mesh3d\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}],\"scatter\":[{\"fillpattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2},\"type\":\"scatter\"}],\"parcoords\":[{\"type\":\"parcoords\",\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatterpolargl\":[{\"type\":\"scatterpolargl\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"scattergeo\":[{\"type\":\"scattergeo\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatterpolar\":[{\"type\":\"scatterpolar\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"scattergl\":[{\"type\":\"scattergl\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatter3d\":[{\"type\":\"scatter3d\",\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scattermapbox\":[{\"type\":\"scattermapbox\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatterternary\":[{\"type\":\"scatterternary\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scattercarpet\":[{\"type\":\"scattercarpet\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}],\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}]},\"layout\":{\"autotypenumbers\":\"strict\",\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"hovermode\":\"closest\",\"hoverlabel\":{\"align\":\"left\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"#E5ECF6\",\"polar\":{\"bgcolor\":\"#E5ECF6\",\"angularaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"radialaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"ternary\":{\"bgcolor\":\"#E5ECF6\",\"aaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"caxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]]},\"xaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"automargin\":true,\"zerolinewidth\":2},\"yaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"automargin\":true,\"zerolinewidth\":2},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\",\"gridwidth\":2},\"yaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\",\"gridwidth\":2},\"zaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\",\"gridwidth\":2}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"geo\":{\"bgcolor\":\"white\",\"landcolor\":\"#E5ECF6\",\"subunitcolor\":\"white\",\"showland\":true,\"showlakes\":true,\"lakecolor\":\"white\"},\"title\":{\"x\":0.05},\"mapbox\":{\"style\":\"light\"}}},\"xaxis\":{\"anchor\":\"y\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"HeartDisease\"}},\"yaxis\":{\"anchor\":\"x\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"count\"}},\"legend\":{\"title\":{\"text\":\"Sex\"},\"tracegroupgap\":0},\"title\":{\"text\":\"Distribution of Heart Diseases\"},\"barmode\":\"group\"},                        {\"responsive\": true}                    ).then(function(){\n",
              "                            \n",
              "var gd = document.getElementById('30cbf971-3723-48c4-8b0e-ebd18d4125ce');\n",
              "var x = new MutationObserver(function (mutations, observer) {{\n",
              "        var display = window.getComputedStyle(gd).display;\n",
              "        if (!display || display === 'none') {{\n",
              "            console.log([gd, 'removed!']);\n",
              "            Plotly.purge(gd);\n",
              "            observer.disconnect();\n",
              "        }}\n",
              "}});\n",
              "\n",
              "// Listen for the removal of the full notebook cells\n",
              "var notebookContainer = gd.closest('#notebook-container');\n",
              "if (notebookContainer) {{\n",
              "    x.observe(notebookContainer, {childList: true});\n",
              "}}\n",
              "\n",
              "// Listen for the clearing of the current output cell\n",
              "var outputEl = gd.closest('.output');\n",
              "if (outputEl) {{\n",
              "    x.observe(outputEl, {childList: true});\n",
              "}}\n",
              "\n",
              "                        })                };                            </script>        </div>\n",
              "</body>\n",
              "</html>"
            ]
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "fig=px.histogram(df,\n",
        "                 x=\"ChestPainType\",\n",
        "                 color=\"Sex\",\n",
        "                 hover_data=df.columns,\n",
        "                 title=\"Types of Chest Pain\"\n",
        "                )\n",
        "fig.show()"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:24:55.0524Z",
          "iopub.execute_input": "2021-12-13T06:24:55.052623Z",
          "iopub.status.idle": "2021-12-13T06:24:55.137753Z",
          "shell.execute_reply.started": "2021-12-13T06:24:55.052596Z",
          "shell.execute_reply": "2021-12-13T06:24:55.136824Z"
        },
        "trusted": true,
        "id": "NiMqYiYicMcX",
        "outputId": "82ffa29d-4924-4be9-82e1-729cdc115990",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 542
        }
      },
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/html": [
              "<html>\n",
              "<head><meta charset=\"utf-8\" /></head>\n",
              "<body>\n",
              "    <div>            <script src=\"https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG\"></script><script type=\"text/javascript\">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: \"STIX-Web\"}});}</script>                <script type=\"text/javascript\">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>\n",
              "        <script charset=\"utf-8\" src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>                <div id=\"b307f406-a052-4a28-8a5d-dda6b88528d0\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"b307f406-a052-4a28-8a5d-dda6b88528d0\")) {                    Plotly.newPlot(                        \"b307f406-a052-4a28-8a5d-dda6b88528d0\",                        [{\"alignmentgroup\":\"True\",\"bingroup\":\"x\",\"hovertemplate\":\"Sex=M\\u003cbr\\u003eChestPainType=%{x}\\u003cbr\\u003ecount=%{y}\\u003cextra\\u003e\\u003c\\u002fextra\\u003e\",\"legendgroup\":\"M\",\"marker\":{\"color\":\"#636efa\",\"pattern\":{\"shape\":\"\"}},\"name\":\"M\",\"offsetgroup\":\"M\",\"orientation\":\"v\",\"showlegend\":true,\"x\":[\"ATA\",\"ATA\",\"NAP\",\"NAP\",\"ATA\",\"ASY\",\"ATA\",\"ATA\",\"ASY\",\"ASY\",\"ASY\",\"ATA\",\"ATA\",\"ATA\",\"NAP\",\"NAP\",\"ASY\",\"ATA\",\"ATA\",\"NAP\",\"NAP\",\"ASY\",\"ASY\",\"ATA\",\"ASY\",\"ATA\",\"NAP\",\"ASY\",\"NAP\",\"ASY\",\"ATA\",\"NAP\",\"ASY\",\"ASY\",\"ATA\",\"ASY\",\"NAP\",\"ASY\",\"ASY\",\"ATA\",\"ASY\",\"ASY\",\"ATA\",\"ASY\",\"ASY\",\"ATA\",\"ATA\",\"ASY\",\"ASY\",\"NAP\",\"ASY\",\"ATA\",\"ASY\",\"NAP\",\"ATA\",\"ASY\",\"ATA\",\"ASY\",\"ASY\",\"ASY\",\"TA\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ATA\",\"NAP\",\"ASY\",\"ATA\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ATA\",\"ATA\",\"ASY\",\"ATA\",\"ASY\",\"ASY\",\"ATA\",\"ASY\",\"TA\",\"ASY\",\"ATA\",\"ATA\",\"NAP\",\"NAP\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"ATA\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ATA\",\"ATA\",\"ASY\",\"ATA\",\"ATA\",\"ATA\",\"ASY\",\"ATA\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ATA\",\"ASY\",\"ASY\",\"ATA\",\"ATA\",\"TA\",\"ASY\",\"ATA\",\"ASY\",\"NAP\",\"ATA\",\"NAP\",\"ATA\",\"NAP\",\"ASY\",\"ASY\",\"ASY\",\"ATA\",\"NAP\",\"ASY\",\"ATA\",\"ASY\",\"ASY\",\"NAP\",\"ATA\",\"ASY\",\"ASY\",\"ASY\",\"ATA\",\"ATA\",\"NAP\",\"ASY\",\"ATA\",\"ASY\",\"TA\",\"NAP\",\"NAP\",\"ATA\",\"ASY\",\"ATA\",\"ATA\",\"ATA\",\"ASY\",\"ASY\",\"NAP\",\"ASY\",\"ASY\",\"NAP\",\"ATA\",\"ATA\",\"ASY\",\"NAP\",\"ATA\",\"ASY\",\"NAP\",\"ASY\",\"ATA\",\"ASY\",\"NAP\",\"ASY\",\"ATA\",\"ASY\",\"TA\",\"ASY\",\"ASY\",\"ATA\",\"ASY\",\"ATA\",\"ASY\",\"TA\",\"NAP\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ATA\",\"ASY\",\"NAP\",\"ATA\",\"ASY\",\"ASY\",\"NAP\",\"ATA\",\"ASY\",\"ATA\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"NAP\",\"NAP\",\"ASY\",\"NAP\",\"ASY\",\"ASY\",\"ASY\",\"ATA\",\"ATA\",\"NAP\",\"ASY\",\"ASY\",\"TA\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ATA\",\"ASY\",\"ASY\",\"ASY\",\"ATA\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ATA\",\"NAP\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"ASY\",\"NAP\",\"ASY\",\"ASY\",\"TA\",\"NAP\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"TA\",\"ASY\",\"ATA\",\"NAP\",\"NAP\",\"NAP\",\"ASY\",\"NAP\",\"ASY\",\"NAP\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"NAP\",\"ASY\",\"NAP\",\"NAP\",\"ATA\",\"ATA\",\"ASY\",\"ASY\",\"NAP\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"NAP\",\"NAP\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"NAP\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"NAP\",\"NAP\",\"ASY\",\"ATA\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"ATA\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"ATA\",\"NAP\",\"TA\",\"ASY\",\"ASY\",\"ATA\",\"ATA\",\"ASY\",\"TA\",\"ASY\",\"NAP\",\"ASY\",\"TA\",\"NAP\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"ASY\",\"NAP\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"TA\",\"ASY\",\"NAP\",\"NAP\",\"ASY\",\"NAP\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"ASY\",\"ATA\",\"NAP\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"NAP\",\"ASY\",\"ASY\",\"NAP\",\"TA\",\"ASY\",\"TA\",\"ASY\",\"NAP\",\"NAP\",\"TA\",\"NAP\",\"NAP\",\"ASY\",\"NAP\",\"NAP\",\"ASY\",\"NAP\",\"ATA\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"ASY\",\"ATA\",\"ASY\",\"NAP\",\"ASY\",\"NAP\",\"ATA\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"NAP\",\"ASY\",\"NAP\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"TA\",\"ASY\",\"ASY\",\"ATA\",\"ASY\",\"ATA\",\"ASY\",\"ASY\",\"NAP\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"TA\",\"ASY\",\"ASY\",\"TA\",\"TA\",\"ASY\",\"ATA\",\"ASY\",\"ASY\",\"NAP\",\"NAP\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"TA\",\"NAP\",\"ASY\",\"ASY\",\"NAP\",\"ATA\",\"NAP\",\"NAP\",\"NAP\",\"ASY\",\"ATA\",\"ASY\",\"ATA\",\"ASY\",\"ATA\",\"NAP\",\"NAP\",\"TA\",\"ASY\",\"ASY\",\"NAP\",\"ASY\",\"ATA\",\"ASY\",\"ASY\",\"ATA\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"ASY\",\"TA\",\"ATA\",\"TA\",\"ASY\",\"ATA\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"ASY\",\"ATA\",\"ATA\",\"ASY\",\"ASY\",\"NAP\",\"ASY\",\"ASY\",\"NAP\",\"ASY\",\"ATA\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"TA\",\"ATA\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"NAP\",\"ASY\",\"ATA\",\"TA\",\"TA\",\"ATA\",\"ASY\",\"ASY\",\"NAP\",\"TA\",\"TA\",\"NAP\",\"ASY\",\"TA\",\"ASY\",\"NAP\",\"ASY\",\"NAP\",\"NAP\",\"ASY\",\"NAP\",\"NAP\",\"ASY\",\"ASY\",\"NAP\",\"ASY\",\"ASY\",\"ATA\",\"ASY\",\"NAP\",\"ASY\",\"NAP\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"TA\",\"NAP\",\"NAP\",\"NAP\",\"TA\",\"NAP\",\"ASY\",\"ATA\",\"NAP\",\"NAP\",\"ATA\",\"ASY\",\"ASY\",\"NAP\",\"ASY\",\"ATA\",\"TA\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"ASY\",\"NAP\",\"ATA\",\"NAP\",\"NAP\",\"ASY\",\"ASY\",\"NAP\",\"ASY\",\"ASY\",\"ATA\",\"ASY\",\"NAP\",\"NAP\",\"NAP\",\"ASY\",\"NAP\",\"ASY\",\"ASY\",\"ATA\",\"ATA\",\"ATA\",\"NAP\",\"ATA\",\"ASY\",\"ASY\",\"NAP\",\"ASY\",\"ASY\",\"TA\",\"TA\",\"ATA\",\"ASY\",\"NAP\",\"ATA\",\"ASY\",\"ASY\",\"ATA\",\"ATA\",\"ATA\",\"NAP\",\"ASY\",\"ASY\",\"ATA\",\"ASY\",\"TA\",\"ASY\",\"ASY\",\"NAP\"],\"xaxis\":\"x\",\"yaxis\":\"y\",\"type\":\"histogram\"},{\"alignmentgroup\":\"True\",\"bingroup\":\"x\",\"hovertemplate\":\"Sex=F\\u003cbr\\u003eChestPainType=%{x}\\u003cbr\\u003ecount=%{y}\\u003cextra\\u003e\\u003c\\u002fextra\\u003e\",\"legendgroup\":\"F\",\"marker\":{\"color\":\"#EF553B\",\"pattern\":{\"shape\":\"\"}},\"name\":\"F\",\"offsetgroup\":\"F\",\"orientation\":\"v\",\"showlegend\":true,\"x\":[\"NAP\",\"ASY\",\"ATA\",\"ATA\",\"NAP\",\"NAP\",\"ATA\",\"ATA\",\"TA\",\"ATA\",\"ATA\",\"ATA\",\"ATA\",\"ATA\",\"ASY\",\"ATA\",\"NAP\",\"ASY\",\"ATA\",\"ASY\",\"ATA\",\"NAP\",\"ATA\",\"ATA\",\"ASY\",\"ASY\",\"ASY\",\"ATA\",\"ATA\",\"ATA\",\"NAP\",\"ASY\",\"ASY\",\"ATA\",\"NAP\",\"ASY\",\"ASY\",\"TA\",\"NAP\",\"NAP\",\"ATA\",\"ATA\",\"ASY\",\"ATA\",\"ASY\",\"ATA\",\"ATA\",\"ATA\",\"ATA\",\"ASY\",\"ATA\",\"ASY\",\"ATA\",\"TA\",\"TA\",\"ATA\",\"NAP\",\"NAP\",\"TA\",\"ASY\",\"NAP\",\"ASY\",\"ATA\",\"ATA\",\"ATA\",\"NAP\",\"ATA\",\"NAP\",\"ATA\",\"ATA\",\"NAP\",\"ATA\",\"ATA\",\"ASY\",\"ATA\",\"NAP\",\"ATA\",\"NAP\",\"ATA\",\"ATA\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"ASY\",\"TA\",\"ASY\",\"ASY\",\"ATA\",\"ASY\",\"ASY\",\"ATA\",\"ASY\",\"ASY\",\"NAP\",\"ATA\",\"ASY\",\"ASY\",\"ASY\",\"ATA\",\"NAP\",\"ASY\",\"NAP\",\"NAP\",\"NAP\",\"ASY\",\"ASY\",\"NAP\",\"ATA\",\"ATA\",\"ATA\",\"NAP\",\"ASY\",\"ASY\",\"TA\",\"ATA\",\"NAP\",\"ASY\",\"NAP\",\"ASY\",\"NAP\",\"NAP\",\"ATA\",\"ASY\",\"NAP\",\"ATA\",\"NAP\",\"ASY\",\"ASY\",\"ATA\",\"ASY\",\"ASY\",\"NAP\",\"ATA\",\"ASY\",\"TA\",\"NAP\",\"NAP\",\"NAP\",\"ASY\",\"NAP\",\"NAP\",\"ATA\",\"NAP\",\"NAP\",\"NAP\",\"ASY\",\"NAP\",\"ATA\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ATA\",\"ASY\",\"NAP\",\"TA\",\"NAP\",\"NAP\",\"ATA\",\"NAP\",\"ASY\",\"NAP\",\"ASY\",\"TA\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"ASY\",\"NAP\",\"NAP\",\"ATA\",\"NAP\",\"ATA\",\"NAP\",\"NAP\",\"NAP\",\"ASY\",\"ASY\",\"ASY\",\"ATA\",\"ASY\",\"ASY\",\"ATA\"],\"xaxis\":\"x\",\"yaxis\":\"y\",\"type\":\"histogram\"}],                        {\"template\":{\"data\":{\"histogram2dcontour\":[{\"type\":\"histogram2dcontour\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"choropleth\":[{\"type\":\"choropleth\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}],\"histogram2d\":[{\"type\":\"histogram2d\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"heatmap\":[{\"type\":\"heatmap\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"heatmapgl\":[{\"type\":\"heatmapgl\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"contourcarpet\":[{\"type\":\"contourcarpet\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}],\"contour\":[{\"type\":\"contour\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"surface\":[{\"type\":\"surface\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"mesh3d\":[{\"type\":\"mesh3d\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}],\"scatter\":[{\"fillpattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2},\"type\":\"scatter\"}],\"parcoords\":[{\"type\":\"parcoords\",\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatterpolargl\":[{\"type\":\"scatterpolargl\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"scattergeo\":[{\"type\":\"scattergeo\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatterpolar\":[{\"type\":\"scatterpolar\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"scattergl\":[{\"type\":\"scattergl\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatter3d\":[{\"type\":\"scatter3d\",\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scattermapbox\":[{\"type\":\"scattermapbox\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatterternary\":[{\"type\":\"scatterternary\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scattercarpet\":[{\"type\":\"scattercarpet\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}],\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}]},\"layout\":{\"autotypenumbers\":\"strict\",\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"hovermode\":\"closest\",\"hoverlabel\":{\"align\":\"left\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"#E5ECF6\",\"polar\":{\"bgcolor\":\"#E5ECF6\",\"angularaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"radialaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"ternary\":{\"bgcolor\":\"#E5ECF6\",\"aaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"caxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]]},\"xaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"automargin\":true,\"zerolinewidth\":2},\"yaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"automargin\":true,\"zerolinewidth\":2},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\",\"gridwidth\":2},\"yaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\",\"gridwidth\":2},\"zaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\",\"gridwidth\":2}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"geo\":{\"bgcolor\":\"white\",\"landcolor\":\"#E5ECF6\",\"subunitcolor\":\"white\",\"showland\":true,\"showlakes\":true,\"lakecolor\":\"white\"},\"title\":{\"x\":0.05},\"mapbox\":{\"style\":\"light\"}}},\"xaxis\":{\"anchor\":\"y\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"ChestPainType\"}},\"yaxis\":{\"anchor\":\"x\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"count\"}},\"legend\":{\"title\":{\"text\":\"Sex\"},\"tracegroupgap\":0},\"title\":{\"text\":\"Types of Chest Pain\"},\"barmode\":\"relative\"},                        {\"responsive\": true}                    ).then(function(){\n",
              "                            \n",
              "var gd = document.getElementById('b307f406-a052-4a28-8a5d-dda6b88528d0');\n",
              "var x = new MutationObserver(function (mutations, observer) {{\n",
              "        var display = window.getComputedStyle(gd).display;\n",
              "        if (!display || display === 'none') {{\n",
              "            console.log([gd, 'removed!']);\n",
              "            Plotly.purge(gd);\n",
              "            observer.disconnect();\n",
              "        }}\n",
              "}});\n",
              "\n",
              "// Listen for the removal of the full notebook cells\n",
              "var notebookContainer = gd.closest('#notebook-container');\n",
              "if (notebookContainer) {{\n",
              "    x.observe(notebookContainer, {childList: true});\n",
              "}}\n",
              "\n",
              "// Listen for the clearing of the current output cell\n",
              "var outputEl = gd.closest('.output');\n",
              "if (outputEl) {{\n",
              "    x.observe(outputEl, {childList: true});\n",
              "}}\n",
              "\n",
              "                        })                };                            </script>        </div>\n",
              "</body>\n",
              "</html>"
            ]
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "fig=px.histogram(df,\n",
        "                 x=\"Sex\",\n",
        "                 hover_data=df.columns,\n",
        "                 title=\"Sex Ratio in the Data\")\n",
        "fig.show()"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:24:55.139095Z",
          "iopub.execute_input": "2021-12-13T06:24:55.139419Z",
          "iopub.status.idle": "2021-12-13T06:24:55.219925Z",
          "shell.execute_reply.started": "2021-12-13T06:24:55.13938Z",
          "shell.execute_reply": "2021-12-13T06:24:55.219019Z"
        },
        "trusted": true,
        "id": "bDiijbndcMcX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "fig=px.histogram(df,\n",
        "                 x=\"RestingECG\",\n",
        "                 hover_data=df.columns,\n",
        "                 title=\"Distribution of Resting ECG\")\n",
        "fig.show()"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:24:55.221403Z",
          "iopub.execute_input": "2021-12-13T06:24:55.22172Z",
          "iopub.status.idle": "2021-12-13T06:24:55.300794Z",
          "shell.execute_reply.started": "2021-12-13T06:24:55.22168Z",
          "shell.execute_reply": "2021-12-13T06:24:55.300117Z"
        },
        "trusted": true,
        "id": "Mkjr74MDcMcX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "To plot multiple pairwise bivariate distributions in a dataset, you can use the pairplot() function. This shows the relationship for (n, 2) combination of variable in a DataFrame as a matrix of plots and the diagonal plots are the univariate plots."
      ],
      "metadata": {
        "id": "-Hv0MlRhcMcX"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "plt.figure(figsize=(15,10))\n",
        "sns.pairplot(df,hue=\"HeartDisease\")\n",
        "plt.title(\"Looking for Insites in Data\")\n",
        "plt.legend(\"HeartDisease\")\n",
        "plt.tight_layout()\n",
        "plt.plot()"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:24:55.302219Z",
          "iopub.execute_input": "2021-12-13T06:24:55.302459Z",
          "iopub.status.idle": "2021-12-13T06:25:08.152054Z",
          "shell.execute_reply.started": "2021-12-13T06:24:55.302429Z",
          "shell.execute_reply": "2021-12-13T06:25:08.151284Z"
        },
        "trusted": true,
        "id": "5zZvTvoCcMcX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Now to check the linearity of the variables it is a good practice to plot distribution graph and look for skewness of features. Kernel density estimate (kde) is a quite useful tool for plotting the shape of a distribution."
      ],
      "metadata": {
        "id": "_VHwKQHScMcY"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "plt.figure(figsize=(15,10))\n",
        "for i,col in enumerate(df.columns,1):\n",
        "    plt.subplot(4,3,i)\n",
        "    plt.title(f\"Distribution of {col} Data\")\n",
        "    sns.histplot(df[col],kde=True)\n",
        "    plt.tight_layout()\n",
        "    plt.plot()\n",
        "\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:25:08.153162Z",
          "iopub.execute_input": "2021-12-13T06:25:08.153867Z",
          "iopub.status.idle": "2021-12-13T06:25:12.20015Z",
          "shell.execute_reply.started": "2021-12-13T06:25:08.153832Z",
          "shell.execute_reply": "2021-12-13T06:25:12.199516Z"
        },
        "trusted": true,
        "id": "gMbmBD15cMcY"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Outliers\n",
        "\n",
        "A box plot (or box-and-whisker plot) shows the distribution of quantitative data in a way that facilitates comparisons between variables.The box shows the quartiles of the dataset while the whiskers extend to show the rest of the distribution.The box plot (a.k.a. box and whisker diagram) is a standardized way of displaying the distribution of data based on the five number summary:\n",
        "- Minimum\n",
        "- First quartile\n",
        "- Median\n",
        "- Third quartile\n",
        "- Maximum.\n",
        "\n",
        "In the simplest box plot the central rectangle spans the first quartile to the third quartile (the interquartile range or IQR).A segment inside the rectangle shows the median and “whiskers” above and below the box show the locations of the minimum and maximum."
      ],
      "metadata": {
        "id": "X9B3hxa8cMcY"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "fig = px.box(df,y=\"Age\",x=\"HeartDisease\",title=f\"Distrubution of Age\")\n",
        "fig.show()"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:25:12.20129Z",
          "iopub.execute_input": "2021-12-13T06:25:12.201666Z",
          "iopub.status.idle": "2021-12-13T06:25:12.292689Z",
          "shell.execute_reply.started": "2021-12-13T06:25:12.201626Z",
          "shell.execute_reply": "2021-12-13T06:25:12.291973Z"
        },
        "trusted": true,
        "id": "dwJOzMkMcMcY"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "fig = px.box(df,y=\"RestingBP\",x=\"HeartDisease\",title=f\"Distrubution of RestingBP\",color=\"Sex\")\n",
        "fig.show()"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:25:12.293821Z",
          "iopub.execute_input": "2021-12-13T06:25:12.294346Z",
          "iopub.status.idle": "2021-12-13T06:25:12.368804Z",
          "shell.execute_reply.started": "2021-12-13T06:25:12.294315Z",
          "shell.execute_reply": "2021-12-13T06:25:12.367979Z"
        },
        "trusted": true,
        "id": "ELgy-0SJcMcY"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "fig = px.box(df,y=\"Cholesterol\",x=\"HeartDisease\",title=f\"Distrubution of Cholesterol\")\n",
        "fig.show()"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:25:12.36991Z",
          "iopub.execute_input": "2021-12-13T06:25:12.370119Z",
          "iopub.status.idle": "2021-12-13T06:25:12.435667Z",
          "shell.execute_reply.started": "2021-12-13T06:25:12.370094Z",
          "shell.execute_reply": "2021-12-13T06:25:12.434686Z"
        },
        "trusted": true,
        "id": "2T5q9K-4cMcY"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "fig = px.box(df,y=\"Oldpeak\",x=\"HeartDisease\",title=f\"Distrubution of Oldpeak\")\n",
        "fig.show()"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:25:12.437191Z",
          "iopub.execute_input": "2021-12-13T06:25:12.437438Z",
          "iopub.status.idle": "2021-12-13T06:25:12.502247Z",
          "shell.execute_reply.started": "2021-12-13T06:25:12.437409Z",
          "shell.execute_reply": "2021-12-13T06:25:12.501565Z"
        },
        "trusted": true,
        "id": "FD6EDnADcMcY"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "fig = px.box(df,y=\"MaxHR\",x=\"HeartDisease\",title=f\"Distrubution of MaxHR\")\n",
        "fig.show()"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:25:12.503547Z",
          "iopub.execute_input": "2021-12-13T06:25:12.503947Z",
          "iopub.status.idle": "2021-12-13T06:25:12.570452Z",
          "shell.execute_reply.started": "2021-12-13T06:25:12.503918Z",
          "shell.execute_reply": "2021-12-13T06:25:12.569666Z"
        },
        "trusted": true,
        "id": "FRHZRZvjcMcY"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Data Preprocessing\n",
        "Data preprocessing is an integral step in Machine Learning as the quality of data and the useful information that can be derived from it directly affects the ability of our model to learn; therefore, it is extremely important that we preprocess our data before feeding it into our model.\n",
        "\n",
        "The concepts that I will cover in this article are\n",
        "1. Handling Null Values\n",
        "2. Feature Scaling\n",
        "3. Handling Categorical Variables"
      ],
      "metadata": {
        "id": "Z5CgXspjcMcZ"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 1. Handling Null Values :\n",
        "In any real-world dataset, there are always few null values. It doesn’t really matter whether it is a regression, classification or any other kind of problem, no model can handle these NULL or NaN values on its own so we need to intervene.\n",
        "\n",
        "> In python NULL is reprsented with NaN. So don’t get confused between these two,they can be used interchangably.\n"
      ],
      "metadata": {
        "id": "RgHIOJxicMcZ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Checking for Type of data\n",
        "df.info()"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:25:12.571986Z",
          "iopub.execute_input": "2021-12-13T06:25:12.572468Z",
          "iopub.status.idle": "2021-12-13T06:25:12.589891Z",
          "shell.execute_reply.started": "2021-12-13T06:25:12.572424Z",
          "shell.execute_reply": "2021-12-13T06:25:12.588877Z"
        },
        "trusted": true,
        "id": "ouF14unGcMcZ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Checking for NULLs in the data\n",
        "df.isnull().sum()"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:25:12.591066Z",
          "iopub.execute_input": "2021-12-13T06:25:12.591566Z",
          "iopub.status.idle": "2021-12-13T06:25:12.601357Z",
          "shell.execute_reply.started": "2021-12-13T06:25:12.591529Z",
          "shell.execute_reply": "2021-12-13T06:25:12.600466Z"
        },
        "trusted": true,
        "id": "U-zrF5O1cMcZ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "So we can see our data does not have any null values\n",
        "but in case we have missing values, we can remove the data as well.\n",
        "\n",
        "However, it is not the best option to remove the rows and columns from our dataset as it can result in significant information loss. If you have 300K data points then removing 2–3 rows won’t affect your dataset much but if you only have 100 data points and out of which 20 have NaN values for a particular field then you can’t simply drop those rows. In real-world datasets, it can happen quite often that you have a large number of NaN values for a particular field.\n",
        "Ex — Suppose we are collecting the data from a survey, then it is possible that there could be an optional field which let’s say 20% of people left blank. So when we get the dataset then we need to understand that the remaining 80% of data is still useful, so rather than dropping these values we need to somehow substitute the missing 20% values. We can do this with the help of Imputation.\n",
        "\n",
        "### Imputation:\n",
        "\n",
        "Imputation is simply the process of substituting the missing values of our dataset. We can do this by defining our own customised function or we can simply perform imputation by using the SimpleImputer class provided by sklearn.\n",
        "\n",
        "For example :\n",
        "\n",
        "```python\n",
        "from sklearn.impute import SimpleImputer\n",
        "\n",
        "imputer = SimpleImputer(missing_values=np.nan, strategy='mean')\n",
        "imputer = imputer.fit(df[['Weight']])\n",
        "df['Weight'] = imputer.transform(df[['Weight']])\n",
        "\n",
        "```\n",
        "\n",
        "### As we do not have any missing data so we will not be using this approch"
      ],
      "metadata": {
        "id": "3DS5dv8mcMcZ"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 2. Feature Scaling\n"
      ],
      "metadata": {
        "id": "6_NREHvOcMca"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Why Should we Use Feature Scaling?\n",
        "\n",
        "The first question we need to address – why do we need to scale the variables in our dataset? Some machine learning algorithms are sensitive to feature scaling while others are virtually invariant to it. Let me explain that in more detail.\n",
        "\n",
        "## 1. Distance Based Algorithms :\n",
        "    \n",
        "Distance algorithms like **\"KNN\"**, **\"K-means\"** and **\"SVM\"** are most affected by the range of features. This is because behind the scenes they are using distances between data points to determine their similarity.\n",
        "Whem two features have different scales, there is a chance that higher weightage is given to features with higher magnitude. This will impact the performance of the machine learning algorithm and obviously, we do not want our algorithm to be biassed towards one feature.\n",
        "\n",
        "Therefore, we scale our data before employing a distance based algorithm so that all the features contribute equally to the result.\n",
        "\n",
        "<img src=\"https://miro.medium.com/max/1000/0*MZKG8sTIdSNv6TXB\" width=50%>\n",
        "\n",
        "\n",
        "## 2. Tree-Based Algorithms :\n",
        "\n",
        "Tree-based algorithms, on the other hand, are fairly insensitive to the scale of the features. Think about it, a decision tree is only splitting a node based on a single feature. The decision tree splits a node on a feature that increases the homogeneity of the node. This split on a feature is not influenced by other features.\n",
        "\n",
        "So, there is virtually no effect of the remaining features on the split. This is what makes them invariant to the scale of the features!\n",
        "\n",
        "<img src=\"https://miro.medium.com/max/925/0*U0rcW7XrdHpvI0hU.jpeg\" width=70%>"
      ],
      "metadata": {
        "id": "Eoe3BYAgcMca"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### What is Normalization?\n",
        "\n",
        "Normalization is a scaling technique in which values are shifted and rescaled so that they end up ranging between 0 and 1. It is also known as Min-Max scaling.\n",
        "\n",
        "Here's the fromula for normalization :\n",
        "\n",
        "<img src=\"https://i.stack.imgur.com/EuitP.png\" width=40%>\n",
        "\n",
        "Here, Xmax and Xmin are the maximum and the minimum values of the feature respectively.\n",
        "\n",
        "When the value of X is the minimum value in the column, the numerator will be 0, and hence X’ is 0\n",
        "On the other hand, when the value of X is the maximum value in the column, the numerator is equal to the denominator and thus the value of X’ is 1\n",
        "If the value of X is between the minimum and the maximum value, then the value of X’ is between 0 and 1"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-12T10:03:15.924288Z",
          "iopub.execute_input": "2021-12-12T10:03:15.924676Z",
          "iopub.status.idle": "2021-12-12T10:03:15.941799Z",
          "shell.execute_reply.started": "2021-12-12T10:03:15.924636Z",
          "shell.execute_reply": "2021-12-12T10:03:15.940934Z"
        },
        "id": "emgDuHLXcMca"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## What is Standardization?\n",
        "\n",
        "Standardization is another scaling technique where the values are centered around the mean with a unit standard deviation. This means that the mean of the attribute becomes zero and the resultant distribution has a unit standard deviation\n",
        "\n",
        "Here's the formula for Standarization:\n",
        "\n",
        "<img src=\"https://clavelresearch.files.wordpress.com/2019/03/z-score-population.png\" width=30%>\n",
        "\n",
        "## The Big Question – Normalize or Standardize?\n",
        "\n",
        "Normalization vs. standardization is an eternal question among machine learning newcomers. Let me elaborate on the answer in this section.\n",
        "\n",
        "- Normalization is good to use when you know that the distribution of your data does not follow a Gaussian distribution. This can be useful in algorithms that do not assume any distribution of the data like K-Nearest Neighbors and Neural Networks.\n",
        "\n",
        "- Standardization, on the other hand, can be helpful in cases where the data follows a Gaussian distribution. However, this does not have to be necessarily true. Also, unlike normalization, standardization does not have a bounding range. So, even if you have outliers in your data, they will not be affected by standardization.\n",
        "\n",
        "However, at the end of the day, the choice of using normalization or standardization will depend on your problem and the machine learning algorithm you are using. There is no hard and fast rule to tell you when to normalize or standardize your data."
      ],
      "metadata": {
        "id": "0T-AWWYVcMca"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Robust Scaler\n",
        "When working with outliers we can use Robust Scaling for scakling our data,\n",
        "It scales features using statistics that are robust to outliers. This method removes the median and scales the data in the range between 1st quartile and 3rd quartile. i.e., in between 25th quantile and 75th quantile range. This range is also called an Interquartile range.\n",
        "The median and the interquartile range are then stored so that it could be used upon future data using the transform method. If outliers are present in the dataset, then the median and the interquartile range provide better results and outperform the sample mean and variance.\n",
        "RobustScaler uses the interquartile range so that it is robust to outliers"
      ],
      "metadata": {
        "id": "lD0BfKjbcMca"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# data\n",
        "x = pd.DataFrame({\n",
        "    # Distribution with lower outliers\n",
        "    'x1': np.concatenate([np.random.normal(20, 2, 1000), np.random.normal(1, 2, 25)]),\n",
        "    # Distribution with higher outliers\n",
        "    'x2': np.concatenate([np.random.normal(30, 2, 1000), np.random.normal(50, 2, 25)]),\n",
        "})\n",
        "np.random.normal\n",
        "\n",
        "scaler = preprocessing.RobustScaler()\n",
        "robust_df = scaler.fit_transform(x)\n",
        "robust_df = pd.DataFrame(robust_df, columns =['x1', 'x2'])\n",
        "\n",
        "scaler = preprocessing.StandardScaler()\n",
        "standard_df = scaler.fit_transform(x)\n",
        "standard_df = pd.DataFrame(standard_df, columns =['x1', 'x2'])\n",
        "\n",
        "scaler = preprocessing.MinMaxScaler()\n",
        "minmax_df = scaler.fit_transform(x)\n",
        "minmax_df = pd.DataFrame(minmax_df, columns =['x1', 'x2'])\n",
        "\n",
        "fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols = 4, figsize =(20, 5))\n",
        "ax1.set_title('Before Scaling')\n",
        "\n",
        "sns.kdeplot(x['x1'], ax = ax1, color ='r')\n",
        "sns.kdeplot(x['x2'], ax = ax1, color ='b')\n",
        "ax2.set_title('After Robust Scaling')\n",
        "\n",
        "sns.kdeplot(robust_df['x1'], ax = ax2, color ='red')\n",
        "sns.kdeplot(robust_df['x2'], ax = ax2, color ='blue')\n",
        "ax3.set_title('After Standard Scaling')\n",
        "\n",
        "sns.kdeplot(standard_df['x1'], ax = ax3, color ='black')\n",
        "sns.kdeplot(standard_df['x2'], ax = ax3, color ='g')\n",
        "ax4.set_title('After Min-Max Scaling')\n",
        "\n",
        "sns.kdeplot(minmax_df['x1'], ax = ax4, color ='black')\n",
        "sns.kdeplot(minmax_df['x2'], ax = ax4, color ='g')\n",
        "plt.show()"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:25:12.60303Z",
          "iopub.execute_input": "2021-12-13T06:25:12.603284Z",
          "iopub.status.idle": "2021-12-13T06:25:13.364331Z",
          "shell.execute_reply.started": "2021-12-13T06:25:12.603257Z",
          "shell.execute_reply": "2021-12-13T06:25:13.363382Z"
        },
        "trusted": true,
        "id": "MKJjEDvXcMca"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 3. Handling Categorical Variables\n",
        "\n",
        "Categorical variables/features are any feature type can be classified into two major types:\n",
        "- Nominal\n",
        "- Ordinal\n",
        "\n",
        "Nominal variables are variables that have two or more categories which do not have any kind of order associated with them. For example, if gender is classified into two groups, i.e. male and female, it can be considered as a nominal variable.Ordinal variables, on the other hand, have “levels” or categories with a particular order associated with them. For example, an ordinal categorical variable can be a feature with three different levels: low, medium and high. Order is important.\n",
        "\n",
        "It is a binary classification problem:\n",
        "the target here is **not skewed** but we use the best metric for this binary classification problem which would be Area Under the ROC Curve (AUC). We can use precision and recall too, but AUC combines these two metrics. Thus, we will be using AUC to evaluate the model that we build on this dataset.\n",
        "\n",
        "We have to know that computers do not understand text data and thus, we need to convert these categories to numbers. A simple way of doing that can be to use :\n",
        "- Label Encoding\n",
        "```python\n",
        "from sklearn.preprocessing import LabelEncoder\n",
        "```\n",
        "- One Hot Encoding\n",
        "```python\n",
        "pd.get_dummies()\n",
        "```\n",
        "\n",
        "but we need to understand where to use which type of label encoding:\n",
        "\n",
        "**For not Tree based Machine Learning Algorithms the best way to go will be to use One-Hot Encoding**\n",
        "- One-Hot-Encoding has the advantage that the result is binary rather than ordinal and that everything sits in an orthogonal vector space.\n",
        "- The disadvantage is that for high cardinality, the feature space can really blow up quickly and you start fighting with the curse of dimensionality. In these cases, I typically employ one-hot-encoding followed by PCA for dimensionality reduction. I find that the judicious combination of one-hot plus PCA can seldom be beat by other encoding schemes. PCA finds the linear overlap, so will naturally tend to group similar features into the same feature\n",
        "\n",
        "**For Tree based Machine Learning Algorithms the best way to go is with Label Encoding**\n",
        "\n",
        "- LabelEncoder can turn [dog,cat,dog,mouse,cat] into [1,2,1,3,2], but then the imposed ordinality means that the average of dog and mouse is cat. Still there are algorithms like decision trees and random forests that can work with categorical variables just fine and LabelEncoder can be used to store values using less disk space."
      ],
      "metadata": {
        "id": "3ZIt03vBcMca"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df[string_col].head()\n",
        "for col in string_col:\n",
        "    print(f\"The distribution of categorical valeus in the {col} is : \")\n",
        "    print(df[col].value_counts())"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:25:13.365611Z",
          "iopub.execute_input": "2021-12-13T06:25:13.365873Z",
          "iopub.status.idle": "2021-12-13T06:25:13.383826Z",
          "shell.execute_reply.started": "2021-12-13T06:25:13.365842Z",
          "shell.execute_reply": "2021-12-13T06:25:13.382727Z"
        },
        "trusted": true,
        "id": "slhFI5YUcMca"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# As we will be using both types of approches for demonstration lets do First Label Ecoding\n",
        "# which will be used with Tree Based Algorthms\n",
        "df_tree = df.apply(LabelEncoder().fit_transform)\n",
        "df_tree.head()"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:25:13.385395Z",
          "iopub.execute_input": "2021-12-13T06:25:13.386051Z",
          "iopub.status.idle": "2021-12-13T06:25:13.405358Z",
          "shell.execute_reply.started": "2021-12-13T06:25:13.386005Z",
          "shell.execute_reply": "2021-12-13T06:25:13.404467Z"
        },
        "trusted": true,
        "id": "0yKwrl41cMca"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "We can use this directly in many tree-based models:\n",
        "- Decision trees\n",
        "- Random forest\n",
        "- Extra Trees\n",
        "- Or any kind of boosted trees model\n",
        "    - XGBoost\n",
        "    - GBM\n",
        "    - LightGBM\n",
        "    \n",
        "This type of encoding cannot be used in linear models, support vector machines or neural networks as they expect data to be normalized (or standardized). For these types of models, we can binarize the data.\n",
        "As shown bellow :"
      ],
      "metadata": {
        "id": "bu4UafoxcMcb"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "## Creaeting one hot encoded features for working with non tree based algorithms\n",
        "df_nontree=pd.get_dummies(df,columns=string_col,drop_first=False)\n",
        "df_nontree.head()"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:25:13.407862Z",
          "iopub.execute_input": "2021-12-13T06:25:13.408094Z",
          "iopub.status.idle": "2021-12-13T06:25:13.439913Z",
          "shell.execute_reply.started": "2021-12-13T06:25:13.408067Z",
          "shell.execute_reply": "2021-12-13T06:25:13.439054Z"
        },
        "trusted": true,
        "id": "6MgedQzhcMcb"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Getting the target column at the end\n",
        "target=\"HeartDisease\"\n",
        "y=df_nontree[target].values\n",
        "df_nontree.drop(\"HeartDisease\",axis=1,inplace=True)\n",
        "df_nontree=pd.concat([df_nontree,df[target]],axis=1)\n",
        "df_nontree.head()"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:25:13.442075Z",
          "iopub.execute_input": "2021-12-13T06:25:13.442837Z",
          "iopub.status.idle": "2021-12-13T06:25:13.468562Z",
          "shell.execute_reply.started": "2021-12-13T06:25:13.442793Z",
          "shell.execute_reply": "2021-12-13T06:25:13.467694Z"
        },
        "trusted": true,
        "id": "77h37R0PcMcb"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Chossing the right Cross-Validation"
      ],
      "metadata": {
        "id": "eo-5LJgDcMcb"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Choosing the right cross-validation depends on the dataset you are dealing with, and one’s choice of cross-validation on one dataset may or may not apply to other datasets. However, there are a few types of cross-validation techniques which are the most popular and widely used.\n",
        "These include:\n",
        "\n",
        "- k-fold cross-validation\n",
        "- stratified k-fold cross-validation\n",
        "Cross-validation is dividing training data into a few parts. We train the model on some of these parts and test on the remaining parts\n",
        "<br>\n",
        "\n",
        "<img src=\"https://i.stack.imgur.com/8uEci.png\" width=50%>\n",
        "<br>\n",
        "\n",
        "## 1. K-fold cross-validation :\n",
        "\n",
        "As you can see, we divide the samples and the targets associated with them. We can divide the data into k different sets which are exclusive of each other. This is known as k-fold cross-validation, We can split any data into k-equal parts using KFold from scikit-learn. Each sample is assigned a value from 0 to k-1 when using k-fold cross validation.\n",
        "\n",
        "## 2. Stratified k-fold cross-validation :\n",
        "\n",
        "If you have a skewed dataset for binary classification with 90% positive samples and only 10% negative samples, you don't want to use random k-fold cross-validation. Using simple k-fold cross-validation for a dataset like this can result in folds with all negative samples. In these cases, we prefer using stratified k-fold cross-validation. Stratified k-fold cross-validation keeps the ratio of labels in each fold constant. So, in each fold, you will have the same 90% positive and 10% negative samples. Thus, whatever metric you choose to evaluate, it will give similar results across all folds.\n",
        "\n"
      ],
      "metadata": {
        "id": "2txGY8r4cMcb"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Training our Machine Learning Model :\n"
      ],
      "metadata": {
        "id": "Zx7MqBPWcMcb"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# NON-TREE BASED ALGORITHMS\n",
        "\n",
        "<img src=\"https://media.giphy.com/media/zMukICnMEZmSf8zvXd/giphy.gif\">"
      ],
      "metadata": {
        "id": "-qse6JnOcMcb"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "So as have talked earlier we have to use differnt ways to works with categorical data, so we will be using different methods:\n",
        "\n",
        "## 1.Using Logistic Regression :\n",
        "\n",
        "Logistic regression is a calculation used to predict a binary outcome: either something happens, or does not. This can be exhibited as Yes/No, Pass/Fail, Alive/Dead, etc.\n",
        "\n",
        "Independent variables are analyzed to determine the binary outcome with the results falling into one of two categories. The independent variables can be categorical or numeric, but the dependent variable is always categorical. Written like this:\n",
        "\n",
        "P(Y=1|X) or P(Y=0|X)\n",
        "\n",
        "It calculates the probability of dependent variable Y, given independent variable X.\n",
        "\n",
        "This can be used to calculate the probability of a word having a positive or negative connotation (0, 1, or on a scale between). Or it can be used to determine the object contained in a photo (tree, flower, grass, etc.), with each object given a probability between 0 and 1.\n",
        "\n",
        "<img src=\"https://static.javatpoint.com/tutorial/machine-learning/images/logistic-regression-in-machine-learning.png\">\n"
      ],
      "metadata": {
        "id": "7vUW5emqcMcb"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "feature_col_nontree=df_nontree.columns.to_list()\n",
        "feature_col_nontree.remove(target)"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:25:13.473215Z",
          "iopub.execute_input": "2021-12-13T06:25:13.473647Z",
          "iopub.status.idle": "2021-12-13T06:25:13.47836Z",
          "shell.execute_reply.started": "2021-12-13T06:25:13.473599Z",
          "shell.execute_reply": "2021-12-13T06:25:13.477608Z"
        },
        "trusted": true,
        "id": "6yONE__0cMcb"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn import model_selection\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_auc_score\n",
        "from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler\n",
        "acc_log=[]\n",
        "\n",
        "kf=model_selection.StratifiedKFold(n_splits=5)\n",
        "for fold , (trn_,val_) in enumerate(kf.split(X=df_nontree,y=y)):\n",
        "\n",
        "    X_train=df_nontree.loc[trn_,feature_col_nontree]\n",
        "    y_train=df_nontree.loc[trn_,target]\n",
        "\n",
        "    X_valid=df_nontree.loc[val_,feature_col_nontree]\n",
        "    y_valid=df_nontree.loc[val_,target]\n",
        "\n",
        "    #print(pd.DataFrame(X_valid).head())\n",
        "    ro_scaler=MinMaxScaler()\n",
        "    X_train=ro_scaler.fit_transform(X_train)\n",
        "    X_valid=ro_scaler.transform(X_valid)\n",
        "\n",
        "\n",
        "    clf=LogisticRegression()\n",
        "    clf.fit(X_train,y_train)\n",
        "    y_pred=clf.predict(X_valid)\n",
        "    print(f\"The fold is : {fold} : \")\n",
        "    print(classification_report(y_valid,y_pred))\n",
        "    acc=roc_auc_score(y_valid,y_pred)\n",
        "    acc_log.append(acc)\n",
        "    print(f\"The accuracy for Fold {fold+1} : {acc}\")\n",
        "    pass"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:25:13.479799Z",
          "iopub.execute_input": "2021-12-13T06:25:13.480214Z",
          "iopub.status.idle": "2021-12-13T06:25:13.852581Z",
          "shell.execute_reply.started": "2021-12-13T06:25:13.480151Z",
          "shell.execute_reply": "2021-12-13T06:25:13.851737Z"
        },
        "trusted": true,
        "id": "vo-SctBkcMcb"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## 2.Using Naive Bayers\n",
        "\n",
        "<img src=\"https://insightimi.files.wordpress.com/2020/04/unnamed-1.png\">"
      ],
      "metadata": {
        "id": "CfE2Lv2LcMcb"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.naive_bayes import GaussianNB\n",
        "acc_Gauss=[]\n",
        "kf=model_selection.StratifiedKFold(n_splits=5)\n",
        "for fold , (trn_,val_) in enumerate(kf.split(X=df_nontree,y=y)):\n",
        "\n",
        "    X_train=df_nontree.loc[trn_,feature_col_nontree]\n",
        "    y_train=df_nontree.loc[trn_,target]\n",
        "\n",
        "    X_valid=df_nontree.loc[val_,feature_col_nontree]\n",
        "    y_valid=df_nontree.loc[val_,target]\n",
        "\n",
        "    ro_scaler=MinMaxScaler()\n",
        "    X_train=ro_scaler.fit_transform(X_train)\n",
        "    X_valid=ro_scaler.transform(X_valid)\n",
        "\n",
        "    clf=GaussianNB()\n",
        "    clf.fit(X_train,y_train)\n",
        "    y_pred=clf.predict(X_valid)\n",
        "    print(f\"The fold is : {fold} : \")\n",
        "    print(classification_report(y_valid,y_pred))\n",
        "    acc=roc_auc_score(y_valid,y_pred)\n",
        "    acc_Gauss.append(acc)\n",
        "    print(f\"The accuracy for {fold+1} : {acc}\")\n",
        "\n",
        "    pass"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:25:13.853907Z",
          "iopub.execute_input": "2021-12-13T06:25:13.854522Z",
          "iopub.status.idle": "2021-12-13T06:25:13.985275Z",
          "shell.execute_reply.started": "2021-12-13T06:25:13.854475Z",
          "shell.execute_reply": "2021-12-13T06:25:13.984555Z"
        },
        "trusted": true,
        "id": "HT-wM7P9cMcb"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## 3.Using SVM(Support Vector Machines):\n",
        "\n",
        "A support vector machine (SVM) uses algorithms to train and classify data within degrees of polarity, taking it to a degree beyond X/Y prediction.\n",
        "\n",
        "For a simple visual explanation, we’ll use two tags: red and blue, with two data features: X and Y, then train our classifier to output an X/Y coordinate as either red or blue.\n",
        "\n",
        "<img src=\"https://monkeylearn.com/static/93a102a9b7b96d9047212e15b627724b/d8712/image4-3.webp\" width=40%>\n",
        "\n",
        "The SVM then assigns a hyperplane that best separates the tags. In two dimensions this is simply a line. Anything on one side of the line is red and anything on the other side is blue. In sentiment analysis, for example, this would be positive and negative.\n",
        "\n",
        "In order to maximize machine learning, the best hyperplane is the one with the largest distance between each tag:\n",
        "\n",
        "<img src=\"https://monkeylearn.com/static/e662f65502ffd24d3ee23c07efe88d9e/d8712/image3-2.webp\" width=40%>\n",
        "\n",
        "However, as data sets become more complex, it may not be possible to draw a single line to classify the data into two camps:\n",
        "\n",
        "<img src=\"https://monkeylearn.com/static/5db2d9178789315ce9fa42f579c895a6/93a24/image2-3.webp\" width=40%>\n",
        "\n",
        "Using SVM, the more complex the data, the more accurate the predictor will become. Imagine the above in three dimensions, with a Z-axis added, so it becomes a circle.\n",
        "\n",
        "Mapped back to two dimensions with the best hyperplane, it looks like this\n",
        "\n",
        "<img src=\"https://monkeylearn.com/static/583405ebadf21c9691030ec4bb875e48/93a24/image6-2.webp\" width=40%>\n",
        "\n",
        "SVM allows for more accurate machine learning because it’s multidimensional.\n",
        "\n",
        "We need to choose the best Kernel according to our need.\n",
        "- The linear kernel is mostly preferred for text classification problems as it performs well for large datasets.\n",
        "- Gaussian kernels tend to give good results when there is no additional information regarding data that is not available.\n",
        "- Rbf kernel is also a kind of Gaussian kernel which projects the high dimensional data and then searches a linear separation for it.\n",
        "- Polynomial kernels give good results for problems where all the training data is normalized."
      ],
      "metadata": {
        "id": "nznkvcbIcMcb"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Using Linear Kernel\n",
        "from sklearn.svm import SVC\n",
        "acc_svm=[]\n",
        "kf=model_selection.StratifiedKFold(n_splits=5)\n",
        "for fold , (trn_,val_) in enumerate(kf.split(X=df_nontree,y=y)):\n",
        "\n",
        "    X_train=df_nontree.loc[trn_,feature_col_nontree]\n",
        "    y_train=df_nontree.loc[trn_,target]\n",
        "\n",
        "    X_valid=df_nontree.loc[val_,feature_col_nontree]\n",
        "    y_valid=df_nontree.loc[val_,target]\n",
        "\n",
        "    ro_scaler=MinMaxScaler()\n",
        "    X_train=ro_scaler.fit_transform(X_train)\n",
        "    X_valid=ro_scaler.transform(X_valid)\n",
        "\n",
        "    clf=SVC(kernel=\"linear\")\n",
        "    clf.fit(X_train,y_train)\n",
        "    y_pred=clf.predict(X_valid)\n",
        "    print(f\"The fold is : {fold} : \")\n",
        "    print(classification_report(y_valid,y_pred))\n",
        "    acc=roc_auc_score(y_valid,y_pred)\n",
        "    acc_svm.append(acc)\n",
        "    print(f\"The accuracy for {fold+1} : {acc}\")\n",
        "\n",
        "    pass"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:25:13.986392Z",
          "iopub.execute_input": "2021-12-13T06:25:13.986714Z",
          "iopub.status.idle": "2021-12-13T06:25:14.123747Z",
          "shell.execute_reply.started": "2021-12-13T06:25:13.986687Z",
          "shell.execute_reply": "2021-12-13T06:25:14.122901Z"
        },
        "trusted": true,
        "id": "a99FXJ4lcMcb"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "## Using Sigmoid Kernel\n",
        "from sklearn.svm import SVC\n",
        "acc_svm_sig=[]\n",
        "kf=model_selection.StratifiedKFold(n_splits=5)\n",
        "for fold , (trn_,val_) in enumerate(kf.split(X=df_nontree,y=y)):\n",
        "\n",
        "    X_train=df_nontree.loc[trn_,feature_col_nontree]\n",
        "    y_train=df_nontree.loc[trn_,target]\n",
        "\n",
        "    X_valid=df_nontree.loc[val_,feature_col_nontree]\n",
        "    y_valid=df_nontree.loc[val_,target]\n",
        "\n",
        "    ro_scaler=MinMaxScaler()\n",
        "    X_train=ro_scaler.fit_transform(X_train)\n",
        "    X_valid=ro_scaler.transform(X_valid)\n",
        "\n",
        "    clf=SVC(kernel=\"sigmoid\")\n",
        "    clf.fit(X_train,y_train)\n",
        "    y_pred=clf.predict(X_valid)\n",
        "    print(f\"The fold is : {fold} : \")\n",
        "    print(classification_report(y_valid,y_pred))\n",
        "    acc=roc_auc_score(y_valid,y_pred)\n",
        "    acc_svm_sig.append(acc)\n",
        "    print(f\"The accuracy for {fold+1} : {acc}\")\n",
        "\n",
        "    pass"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:25:14.125111Z",
          "iopub.execute_input": "2021-12-13T06:25:14.125442Z",
          "iopub.status.idle": "2021-12-13T06:25:14.318971Z",
          "shell.execute_reply.started": "2021-12-13T06:25:14.125399Z",
          "shell.execute_reply": "2021-12-13T06:25:14.318033Z"
        },
        "trusted": true,
        "id": "kfCylnM0cMcb"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "## Using RBF kernel\n",
        "from sklearn.svm import SVC\n",
        "acc_svm_rbf=[]\n",
        "kf=model_selection.StratifiedKFold(n_splits=5)\n",
        "for fold , (trn_,val_) in enumerate(kf.split(X=df_nontree,y=y)):\n",
        "\n",
        "    X_train=df_nontree.loc[trn_,feature_col_nontree]\n",
        "    y_train=df_nontree.loc[trn_,target]\n",
        "\n",
        "    X_valid=df_nontree.loc[val_,feature_col_nontree]\n",
        "    y_valid=df_nontree.loc[val_,target]\n",
        "\n",
        "    ro_scaler=MinMaxScaler()\n",
        "    X_train=ro_scaler.fit_transform(X_train)\n",
        "    X_valid=ro_scaler.transform(X_valid)\n",
        "\n",
        "    clf=SVC(kernel=\"rbf\")\n",
        "    clf.fit(X_train,y_train)\n",
        "    y_pred=clf.predict(X_valid)\n",
        "    print(f\"The fold is : {fold} : \")\n",
        "    print(classification_report(y_valid,y_pred))\n",
        "    acc=roc_auc_score(y_valid,y_pred)\n",
        "    acc_svm_rbf.append(acc)\n",
        "    print(f\"The accuracy for {fold+1} : {acc}\")\n",
        "\n",
        "    pass"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:25:14.320482Z",
          "iopub.execute_input": "2021-12-13T06:25:14.32099Z",
          "iopub.status.idle": "2021-12-13T06:25:14.476021Z",
          "shell.execute_reply.started": "2021-12-13T06:25:14.32095Z",
          "shell.execute_reply": "2021-12-13T06:25:14.475125Z"
        },
        "trusted": true,
        "id": "iyLlKC4icMcc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "## Using RBF kernel\n",
        "from sklearn.svm import SVC\n",
        "acc_svm_poly=[]\n",
        "kf=model_selection.StratifiedKFold(n_splits=5)\n",
        "for fold , (trn_,val_) in enumerate(kf.split(X=df_nontree,y=y)):\n",
        "\n",
        "    X_train=df_nontree.loc[trn_,feature_col_nontree]\n",
        "    y_train=df_nontree.loc[trn_,target]\n",
        "\n",
        "    X_valid=df_nontree.loc[val_,feature_col_nontree]\n",
        "    y_valid=df_nontree.loc[val_,target]\n",
        "\n",
        "    ro_scaler=MinMaxScaler()\n",
        "    X_train=ro_scaler.fit_transform(X_train)\n",
        "    X_valid=ro_scaler.transform(X_valid)\n",
        "\n",
        "    clf=SVC(kernel=\"poly\")\n",
        "    clf.fit(X_train,y_train)\n",
        "    y_pred=clf.predict(X_valid)\n",
        "    print(f\"The fold is : {fold} : \")\n",
        "    print(classification_report(y_valid,y_pred))\n",
        "    acc=roc_auc_score(y_valid,y_pred)\n",
        "    acc_svm_poly.append(acc)\n",
        "    print(f\"The accuracy for {fold+1} : {acc}\")\n",
        "\n",
        "    pass"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:25:14.477422Z",
          "iopub.execute_input": "2021-12-13T06:25:14.477662Z",
          "iopub.status.idle": "2021-12-13T06:25:14.63591Z",
          "shell.execute_reply.started": "2021-12-13T06:25:14.477634Z",
          "shell.execute_reply": "2021-12-13T06:25:14.635066Z"
        },
        "trusted": true,
        "id": "5ifbsw_5cMcc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Using K-nearest Neighbors\n",
        "The optimal K value usually found is the square root of N, where N is the total number of samples\n",
        "\n",
        "\n",
        "K-nearest neighbors (k-NN) is a pattern recognition algorithm that uses training datasets to find the k closest relatives in future examples.\n",
        "\n",
        "When k-NN is used in classification, you calculate to place data within the category of its nearest neighbor. If k = 1, then it would be placed in the class nearest 1. K is classified by a plurality poll of its neighbors.\n",
        "\n",
        "<img src=\"http://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1531424125/KNN_final1_ibdm8a.png\">\n"
      ],
      "metadata": {
        "id": "sn7aNst7cMcc"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "## Using RBF kernel\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "acc_KNN=[]\n",
        "kf=model_selection.StratifiedKFold(n_splits=5)\n",
        "for fold , (trn_,val_) in enumerate(kf.split(X=df_nontree,y=y)):\n",
        "\n",
        "    X_train=df_nontree.loc[trn_,feature_col_nontree]\n",
        "    y_train=df_nontree.loc[trn_,target]\n",
        "\n",
        "    X_valid=df_nontree.loc[val_,feature_col_nontree]\n",
        "    y_valid=df_nontree.loc[val_,target]\n",
        "\n",
        "    ro_scaler=MinMaxScaler()\n",
        "    X_train=ro_scaler.fit_transform(X_train)\n",
        "    X_valid=ro_scaler.transform(X_valid)\n",
        "\n",
        "    clf=KNeighborsClassifier(n_neighbors=32)\n",
        "    clf.fit(X_train,y_train)\n",
        "    y_pred=clf.predict(X_valid)\n",
        "    print(f\"The fold is : {fold} : \")\n",
        "    print(classification_report(y_valid,y_pred))\n",
        "    acc=roc_auc_score(y_valid,y_pred)\n",
        "    acc_KNN.append(acc)\n",
        "    print(f\"The accuracy for {fold+1} : {acc}\")\n",
        "\n",
        "    pass"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:25:14.63737Z",
          "iopub.execute_input": "2021-12-13T06:25:14.637601Z",
          "iopub.status.idle": "2021-12-13T06:25:14.86269Z",
          "shell.execute_reply.started": "2021-12-13T06:25:14.637572Z",
          "shell.execute_reply": "2021-12-13T06:25:14.861709Z"
        },
        "trusted": true,
        "id": "1COtmJBvcMcc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# TREE BASED ALGORITHM\n",
        "<img src=\"https://media.giphy.com/media/IyZM6HFe2zgrK/giphy.gif\">"
      ],
      "metadata": {
        "id": "IFu1-pKvcMcc"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Using Decission tree Classifier\n",
        "A decision tree is a supervised learning algorithm that is perfect for classification problems, as it’s able to order classes on a precise level. It works like a flow chart, separating data points into two similar categories at a time from the “tree trunk” to “branches,” to “leaves,” where the categories become more finitely similar. This creates categories within categories, allowing for organic classification with limited human supervision."
      ],
      "metadata": {
        "id": "LigWVqMLcMcc"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "feature_col_tree=df_tree.columns.to_list()\n",
        "feature_col_tree.remove(target)"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:25:14.864035Z",
          "iopub.execute_input": "2021-12-13T06:25:14.864354Z",
          "iopub.status.idle": "2021-12-13T06:25:14.868935Z",
          "shell.execute_reply.started": "2021-12-13T06:25:14.864312Z",
          "shell.execute_reply": "2021-12-13T06:25:14.867971Z"
        },
        "trusted": true,
        "id": "SfdZnOLAcMcc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.tree import DecisionTreeClassifier\n",
        "acc_Dtree=[]\n",
        "kf=model_selection.StratifiedKFold(n_splits=5)\n",
        "for fold , (trn_,val_) in enumerate(kf.split(X=df_tree,y=y)):\n",
        "\n",
        "    X_train=df_tree.loc[trn_,feature_col_tree]\n",
        "    y_train=df_tree.loc[trn_,target]\n",
        "\n",
        "    X_valid=df_tree.loc[val_,feature_col_tree]\n",
        "    y_valid=df_tree.loc[val_,target]\n",
        "\n",
        "    clf=DecisionTreeClassifier(criterion=\"entropy\")\n",
        "    clf.fit(X_train,y_train)\n",
        "    y_pred=clf.predict(X_valid)\n",
        "    print(f\"The fold is : {fold} : \")\n",
        "    print(classification_report(y_valid,y_pred))\n",
        "    acc=roc_auc_score(y_valid,y_pred)\n",
        "    acc_Dtree.append(acc)\n",
        "    print(f\"The accuracy for {fold+1} : {acc}\")"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:54:58.119708Z",
          "iopub.execute_input": "2021-12-13T06:54:58.121148Z",
          "iopub.status.idle": "2021-12-13T06:54:58.289541Z",
          "shell.execute_reply.started": "2021-12-13T06:54:58.12103Z",
          "shell.execute_reply": "2021-12-13T06:54:58.288256Z"
        },
        "trusted": true,
        "id": "D1xZbC7ecMcc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import graphviz\n",
        "from sklearn import tree\n",
        "# DOT data\n",
        "dot_data = tree.export_graphviz(clf, out_file=None,\n",
        "                                feature_names=feature_col_tree,\n",
        "                                class_names=target,\n",
        "                                filled=True)\n",
        "\n",
        "# Draw graph\n",
        "graph = graphviz.Source(dot_data, format=\"png\")\n",
        "graph"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:56:11.089816Z",
          "iopub.execute_input": "2021-12-13T06:56:11.090696Z",
          "iopub.status.idle": "2021-12-13T06:56:11.965711Z",
          "shell.execute_reply.started": "2021-12-13T06:56:11.090656Z",
          "shell.execute_reply": "2021-12-13T06:56:11.964696Z"
        },
        "trusted": true,
        "id": "cjJiWtklcMcc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Using Random Forest Classifier\n",
        "\n",
        "Random forest, like its name implies, consists of a large number of individual decision trees that operate as an ensemble. Each individual tree in the random forest spits out a class prediction and the class with the most votes becomes our model’s prediction (see figure below).\n",
        "\n",
        "<img src=\"https://miro.medium.com/max/5752/1*5dq_1hnqkboZTcKFfwbO9A.png\" width=70%>\n",
        "\n",
        "The fundamental concept behind random forest is a simple but powerful one — the wisdom of crowds. In data science speak, the reason that the random forest model works so well is:\n",
        "\n",
        ">A large number of relatively uncorrelated models (trees) operating as a committee will outperform any of the individual constituent models.\n",
        "\n",
        "The low correlation between models is the key. Just like how investments with low correlations (like stocks and bonds) come together to form a portfolio that is greater than the sum of its parts, uncorrelated models can produce ensemble predictions that are more accurate than any of the individual predictions. The reason for this wonderful effect is that the trees protect each other from their individual errors (as long as they don’t constantly all err in the same direction). While some trees may be wrong, many other trees will be right, so as a group the trees are able to move in the correct direction."
      ],
      "metadata": {
        "id": "Ba4nCxYkcMcc"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.ensemble import RandomForestClassifier\n",
        "acc_RandF=[]\n",
        "kf=model_selection.StratifiedKFold(n_splits=5)\n",
        "for fold , (trn_,val_) in enumerate(kf.split(X=df_tree,y=y)):\n",
        "\n",
        "    X_train=df_tree.loc[trn_,feature_col_tree]\n",
        "    y_train=df_tree.loc[trn_,target]\n",
        "\n",
        "    X_valid=df_tree.loc[val_,feature_col_tree]\n",
        "    y_valid=df_tree.loc[val_,target]\n",
        "\n",
        "    clf=RandomForestClassifier(n_estimators=200,criterion=\"entropy\")\n",
        "    clf.fit(X_train,y_train)\n",
        "    y_pred=clf.predict(X_valid)\n",
        "    print(f\"The fold is : {fold} : \")\n",
        "    print(classification_report(y_valid,y_pred))\n",
        "    acc=roc_auc_score(y_valid,y_pred)\n",
        "    acc_RandF.append(acc)\n",
        "    print(f\"The accuracy for {fold+1} : {acc}\")\n",
        "\n"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:58:48.838287Z",
          "iopub.execute_input": "2021-12-13T06:58:48.838734Z",
          "iopub.status.idle": "2021-12-13T06:58:51.449926Z",
          "shell.execute_reply.started": "2021-12-13T06:58:48.838704Z",
          "shell.execute_reply": "2021-12-13T06:58:51.449125Z"
        },
        "trusted": true,
        "id": "pGQRLPS5cMcc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "## Checking Feature importance\n",
        "\n",
        "plt.figure(figsize=(20,15))\n",
        "importance = clf.feature_importances_\n",
        "idxs = np.argsort(importance)\n",
        "plt.title(\"Feature Importance\")\n",
        "plt.barh(range(len(idxs)),importance[idxs],align=\"center\")\n",
        "plt.yticks(range(len(idxs)),[feature_col_tree[i] for i in idxs])\n",
        "plt.xlabel(\"Random Forest Feature Importance\")\n",
        "#plt.tight_layout()\n",
        "plt.show()"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T06:28:26.807052Z",
          "iopub.execute_input": "2021-12-13T06:28:26.807337Z",
          "iopub.status.idle": "2021-12-13T06:28:27.189819Z",
          "shell.execute_reply.started": "2021-12-13T06:28:26.807307Z",
          "shell.execute_reply": "2021-12-13T06:28:27.18921Z"
        },
        "trusted": true,
        "id": "ZIDo6CI_cMcc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Using XGBoost\n",
        "\n",
        "Unlike many other algorithms, XGBoost is an ensemble learning algorithm meaning that it combines the results of many models, called base learners to make a prediction.\n",
        "\n",
        "Just like in Random Forests, XGBoost uses Decision Trees as base learners:\n",
        "\n",
        "\n",
        "\n",
        "However, the trees used by XGBoost are a bit different than traditional decision trees. They are called CART trees (Classification and Regression trees) and instead of containing a single decision in each “leaf” node, they contain real-value scores of whether an instance belongs to a group. After the tree reaches max depth, the decision can be made by converting the scores into categories using a certain threshold."
      ],
      "metadata": {
        "id": "80fI_W60cMcd"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from xgboost import XGBClassifier\n",
        "acc_XGB=[]\n",
        "kf=model_selection.StratifiedKFold(n_splits=5)\n",
        "for fold , (trn_,val_) in enumerate(kf.split(X=df_tree,y=y)):\n",
        "\n",
        "    X_train=df_tree.loc[trn_,feature_col_tree]\n",
        "    y_train=df_tree.loc[trn_,target]\n",
        "\n",
        "    X_valid=df_tree.loc[val_,feature_col_tree]\n",
        "    y_valid=df_tree.loc[val_,target]\n",
        "\n",
        "    clf=XGBClassifier()\n",
        "    clf.fit(X_train,y_train)\n",
        "    y_pred=clf.predict(X_valid)\n",
        "    print(f\"The fold is : {fold} : \")\n",
        "    print(classification_report(y_valid,y_pred))\n",
        "    acc=roc_auc_score(y_valid,y_pred)\n",
        "    acc_XGB.append(acc)\n",
        "    print(f\"The accuracy for {fold+1} : {acc}\")"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T07:35:23.72433Z",
          "iopub.execute_input": "2021-12-13T07:35:23.724615Z",
          "iopub.status.idle": "2021-12-13T07:35:25.319586Z",
          "shell.execute_reply.started": "2021-12-13T07:35:23.724585Z",
          "shell.execute_reply": "2021-12-13T07:35:25.318707Z"
        },
        "trusted": true,
        "id": "q7LgU7LScMcd"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "fig, ax = plt.subplots(figsize=(30, 30))\n",
        "from xgboost import plot_tree\n",
        "plot_tree(clf,num_trees=0,rankdir=\"LR\",ax=ax)\n",
        "plt.show()"
      ],
      "metadata": {
        "execution": {
          "iopub.status.busy": "2021-12-13T07:37:50.314494Z",
          "iopub.execute_input": "2021-12-13T07:37:50.31508Z",
          "iopub.status.idle": "2021-12-13T07:37:51.959147Z",
          "shell.execute_reply.started": "2021-12-13T07:37:50.315047Z",
          "shell.execute_reply": "2021-12-13T07:37:51.958266Z"
        },
        "trusted": true,
        "id": "mNkXIa4EcMcd"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Choosing the best Evaluation Matrix:\n",
        "\n",
        "If we talk about classification problems, the most common metrics used are:\n",
        "- Accuracy\n",
        "- Precision (P)\n",
        "- Recall (R)\n",
        "- F1 score (F1)\n",
        "- Area under the ROC (Receiver Operating Characteristic) curve or simply AUC (AUC):\n",
        "    - If we calculate the area under the ROC curve, we are calculating another metric which is used very often when you have a dataset which has skewed binary targets. This metric is known as the Area Under ROC Curve or Area Under Curve or just simply AUC. There are many ways to calculate the area under the ROC curve\n",
        "    - AUC is a widely used metric for skewed binary classification tasks in the industry,and a metric everyone should know about\n",
        "\n",
        "- Log loss\n",
        "    > Log Loss = - 1.0 * ( target * log(prediction) + (1 - target) * log(1 - prediction) )\n",
        "\n",
        "Most of the metrics that we discussed until now can be converted to a multi-class version. The idea is quite simple. Let’s take precision and recall. We can calculate precision and recall for each class in a multi-class classification problem\n",
        "\n",
        "- **Mcro averaged precision**: calculate precision for all classes individually and then average them\n",
        "- **Micro averaged precision**: calculate class wise true positive and false positive and then use that to calculate overall precision\n",
        "- **Weighted precision**: same as macro but in this case, it is weighted average depending on the number of items in each class\n",
        "\n",
        "<img src=\"https://cdn-images-1.medium.com/max/800/1*1WPbfzztdv50V22TpA6njw.png\">\n"
      ],
      "metadata": {
        "id": "6hTS5agCcMcd"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Further Readings if interested!!!!\n",
        "\n",
        "I have created these notebooks as well for anyone who is intereseted in learning new things in the domain of machine learning application. Please refer to them if interested.😁\n",
        "\n",
        "- Notebook : Automating the Machine Learning workflow using [AutoXGB : XGBoost + Optuna](https://www.kaggle.com/durgancegaur/autoxgb-xgboost-optuna-score-0-95437-dec)\n",
        "- Notebook : [Working with CatBoost](https://www.kaggle.com/durgancegaur/tps-dec-hyperp-tuning-catboost-score-0-95451)\n",
        "- Notebook : Working with data imbalance and [upsampling of data SMOTE](https://www.kaggle.com/durgancegaur/working-with-data-imbalance-eda-99-auc)\n",
        "- Notebook : Working wtih [H2OAuoML](https://www.kaggle.com/durgancegaur/eda-and-working-with-h2oautoml)\n",
        "\n",
        "\n",
        "<img src=\"https://media.giphy.com/media/wsWaiO3gBYXyZeq9v2/giphy.gif\">"
      ],
      "metadata": {
        "id": "EAidlJdjcMcd"
      }
    },
    {
      "cell_type": "raw",
      "source": [
        "# Thanks for reading till the end ! If you liked the notebook pls do Upvote👍 and give some remarks/advice if you feels some things need to be added😁😁😁"
      ],
      "metadata": {
        "id": "GUkZEBeMcMcd"
      }
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "h2t3FK2OcMcd"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
