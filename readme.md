# 📌 CNN EMNIST Visualization Project

This project implements Convolutional Neural Networks (CNNs) to classify handwritten digits (MNIST) and letters (EMNIST). It provides a user-friendly interface via a **Streamlit web application** where users can draw characters and visualize the model's real-time predictions and internal layer activations.

## 📖 Table of Contents
- [Project Structure](#project-structure)
- [System Requirements](#system-requirements)
- [Getting Started](#getting-started)
    - [Step 1: Install Python 3.11.7](#step-1-install-python-3117)
    - [Step 2: Create Virtual Environment (Recommended)](#step-2-create-virtual-environment-recommended)
    - [Step 3: Install Required Libraries](#step-3-install-required-libraries)
    - [Step 4: Train the Models](#step-4-train-the-models)
    - [Step 5: Run the Streamlit App](#step-5-run-the-streamlit-app)
- [App Features](#app-features)
- [Notes](#notes)
- [Troubleshooting](#troubleshooting)

## 📂 Project Structure

The project should be organized with the following folder and file structure. **Do not rename the data files.**

```
CNN_EMNIST_VISUALIZATION/
│
├── data/
│   └── emnist_letters/
│       ├── emnist-letters-mapping.txt
│       ├── emnist-letters-train-images-idx3-ubyte
│       ├── emnist-letters-train-labels-idx1-ubyte
│       ├── emnist-letters-test-images-idx3-ubyte
│       └── emnist-letters-test-labels-idx1-ubyte
│
├── app.py
├── train_digit_model.py
├── train_letter_model.py
├── requirements.txt
└── README.md
```

## 🧰 System Requirements

| Requirement | Details |
| :--- | :--- |
| **Python Version** | **3.11.7** (Crucial for compatibility) |
| Operating System | Windows / Linux / macOS |
| Internet Access | Required only for initial library installation |

## 🚀 Getting Started

Follow these steps to set up the project, train the models, and run the visualization application.

### Step 1: Install Python 3.11.7

It is essential to use the specified Python version for library compatibility.

1.  Download Python 3.11.7 from the official website: [https://www.python.org/downloads/](https://www.python.org/downloads/)
2.  During the installation process, ensure you check the box for **`Add Python to PATH`**.
3.  Verify the installation by opening a terminal or command prompt and running:

    ```bash
    python --version
    ```

    The output should confirm the version: `Python 3.11.7`.

### Step 2: Create Virtual Environment (Recommended)

Using a virtual environment is highly recommended to manage project dependencies.

1.  Open your terminal or command prompt inside the project's root directory (`CNN_EMNIST_VISUALIZATION/`).
2.  Create the virtual environment:

    ```bash
    python -m venv venv
    ```

3.  Activate the environment:

    -   **Windows:**
        ```bash
        venv\Scripts\activate
        ```
    -   **Linux / macOS:**
        ```bash
        source venv/bin/activate
        ```

### Step 3: Install Required Libraries

Ensure the `requirements.txt` file is present in the root directory.

1.  Upgrade `pip` and install all required libraries:

    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    Wait for all packages to install successfully.

### Step 4: Train the Models

Both the digit and letter classification models must be trained before running the application. **Training may take a few minutes depending on your CPU.**

#### 🔢 Train Digit (MNIST) Model

Run the following command to train the CNN on MNIST digits:

```bash
python train_digit_model.py
```

This script will train the model and save the resulting weights (e.g., as `digit_cnn_model.h5`).

#### 🔤 Train Letter (EMNIST) Model

Run the following command to train the CNN on EMNIST letters:

```bash
python train_letter_model.py
```

This script will load the EMNIST letter dataset, train the classification model, and save the weights (e.g., as `letter_cnn_model.h5`).

### Step 5: Run the Streamlit App

Once both models are trained and saved, you can launch the interactive visualization application.

```bash
streamlit run app.py
```

Streamlit will automatically open the application in your default web browser at: `http://localhost:8501`.

## ✍️ Features of the App

The Streamlit application provides a rich, interactive experience for visualizing the CNN models:

*   **Draw Digits or Letters**: Interactive canvas for user input.
*   **Live CNN Prediction**: Real-time inference on the drawn character.
*   **Layer Visualization**: Visual representation of feature maps from the CNN layers.
*   **Probability Bars**: Graphical display of the model's confidence across all classes.
*   **Real-time Inference**: Immediate feedback as the user draws.

## 🧠 Notes

*   Ensure the **folder structure** is maintained exactly as specified above.
*   **Do not rename** the data files in the `data/emnist_letters/` directory.
*   If the Streamlit application shows an error, try restarting your terminal and running `streamlit run app.py` again.

## 🛠 Troubleshooting

If you encounter common dependency conflicts, the following fixes may resolve the issues:

| Error | Fix | Command |
| :--- | :--- | :--- |
| **TensorFlow Error** | Install a specific, compatible version of TensorFlow. | `pip install tensorflow==2.20.0` |
| **Protobuf Error** | Install a specific, compatible version of Protobuf. | `pip install protobuf==6.33.3` |
