# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- **Model type:** Random Forest Classifier
- **Library used:** scikit-learn
- **Number of estimators:** 100
- **Random state:** 42
- **Input features:**
  - Categorical: workclass, education, marital-status, occupation, relationship, race, sex, native-country
  - Numerical: age, fnlgt, education-num, capital-gain, capital-loss, hours-per-week
- **Output:** Binary label predicting whether salary is `>50K` or `<=50K`

## Intended Use
- **Purpose:** Predict income category for census data
- **Users:** Researchers, analysts, or developers interested in income prediction models
- **Intended settings:** Batch predictions or real-time predictions via FastAPI REST API
- **Example API usage:**
  - **GET root endpoint:**
    ```bash
    curl http://127.0.0.1:8000/
    ```
    Response:
    ```json
    {"message": "Welcome to the Census Income Prediction API"}

  - **POST data for inference:**
    ```bash
    curl -X POST "http://127.0.0.1:8000/data/" \
    -H "Content-Type: application/json" \
    -d '{
      "age": 37,
      "workclass": "Private",
      "fnlgt": 178356,
      "education": "HS-grad",
      "education-num": 10,
      "marital-status": "Married-civ-spouse",
      "occupation": "Prof-specialty",
      "relationship": "Husband",
      "race": "White",
      "sex": "Male",
      "capital-gain": 0,
      "capital-loss": 0,
      "hours-per-week": 40,
      "native-country": "United-States"
    }'
    ```
    Response:
    ```json
    {"result": "<=50K"}
    ```
## Training Data
- **Source:** census.csv
- **Size:** 32,561 rows
- **Preprocessing:**
  - One-hot encoding for categorical features
  - Label binarization for the target variable
- **Training split:** 80% training, 20% test

## Evaluation Data
- **Split:** 20% of original data reserved for testing
- **Evaluation method:** Model tested on unseen samples, including categorical slices
- **Slice analysis:** Metrics computed for each unique value of all categorical features

## Metrics
**Overall performance:**
- Precision: 0.7419
- Recall: 0.6384
- F1 Score: 0.6863 
*Note: All slices of categorical features are included in `slice_output.txt` for reference.*

## Ethical Considerations
- Model trained on historical census data, which may contain biases.
- Some demographic slices (e.g., rare categories) have very few samples → metrics may be unstable.
- Predictions should **not** be used for high-stakes decisions affecting individuals.
- Users should monitor for fairness and bias in deployment settings.

## Caveats and Recommendations
- Small sample sizes in certain categorical slices can lead to unreliable predictions.
- Continuous retraining is recommended if the model is deployed in real-world environments.
- Consider additional preprocessing for missing or unknown values.
- Evaluate fairness across sensitive features before deployment.