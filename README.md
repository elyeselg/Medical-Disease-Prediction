Medical Chronic Disease Prediction


This project aims to predict chronic diseases based on medical datasets containing both structured and clinical data. Using a combination of exploratory data analysis (EDA), feature engineering, and machine learning (Random Forest Classifier), the model predicts disease classifications with high accuracy. The project also includes model interpretation and modular design for maintainability and scalability.


## Project Overview

Chronic diseases, such as diabetes and hypertension, have a growing impact worldwide. Early prediction of these diseases can save lives by enabling timely intervention. This project leverages machine learning to create a reliable prediction model trained on medical datasets.
Key Features:

    Exploratory Data Analysis (EDA): Understand the relationships between features and target variables using heatmaps, histograms, and more.
    Data Preprocessing: Handle missing values, normalize numeric features, and encode categorical data.
    Model Training and Optimization: Use Random Forest Classifier with hyperparameter tuning for optimal performance.
    Model Evaluation: Evaluate accuracy, feature importance, and visualize confusion matrices.
    Modular Codebase: Organized into separate files for preprocessing, analysis, and training.
    

## Dataset

The dataset contains 26 medical features collected from patients. The target variable (classification) indicates the diagnosis category.
Columns:

    id - Patient ID
    age - Age of the patient
    bp - Blood Pressure
    sg - Specific Gravity (urine test)
    al - Albumin (protein levels in urine)
    su - Sugar levels in urine
    rbc - Red Blood Cells
    pc - Pus Cells
    pcc - Pus Cell Clumps
    ba - Presence of Bacteria
    bgr - Blood Glucose Random
    bu - Blood Urea
    sc - Serum Creatinine
    sod - Sodium
    pot - Potassium
    hemo - Hemoglobin
    pcv - Packed Cell Volume
    wc - White Cell Count
    rc - Red Cell Count
    htn - Hypertension
    dm - Diabetes Mellitus
    cad - Coronary Artery Disease
    appet - Appetite
    pe - Pedal Edema
    ane - Anemia
    classification - Disease classification (Target variable)
    

## Installation
Prerequisites:

    Python 3.8+
    Virtual environment (recommended)

Steps:

    Clone the repository: git clone https://github.com/elyeselg/Medical-Disease-Prediction.git
    cd Medical-Disease-Prediction
    Run the main script: python main.py
    
    Outputs:

    Exploratory Data Analysis (EDA): Generates heatmaps and histograms for insights.
    Model Training and Evaluation: Trains the Random Forest model, evaluates accuracy, and visualizes the confusion matrix.
    Feature Importance: Shows the most relevant medical features influencing the prediction.
    Model Export: Saves the trained model as a .pkl file.


## Future Improvements

    Dataset Expansion:
        Incorporate additional patient data to generalize the model.
    Model Optimization:
        Experiment with Gradient Boosting or Neural Networks.
    Deployment:
        Create an API for real-time disease prediction.
    Explainability:
        Use SHAP or LIME for better interpretability of predictions.

        

## Contributing

Contributions are welcome! If you have ideas for improving the model or adding new features, please fork the repository and submit a pull request.


## Acknowledgements

Special thanks to:

    Kaggle for providing access to the dataset.
    Python community for excellent libraries like pandas, scikit-learn, and seaborn.
    

## Contact

For questions or suggestions, feel free to reach out:

    Email: elgamouneelyes@gmail.com


## License

This project is licensed under the MIT License.
