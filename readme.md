# About this project
This project extracts feature from [durian leaf disease dataset](https://www.kaggle.com/datasets/cthng123/durian-leaf-disease-dataset) such as color, texture, edge features and converts those images to numeric feature vectors. Then we use data we extracted to train machine learning models, find the best model, and use the model in our test application.

# Set up guide
## Installing dependencies
`pip install fastapi uvicorn python-multipart opencv-python numpy scikit-image scikit-learn joblib pillow matplotlib seaborn xgboost`

### Start FastAPI server (For the tester UI):
`uvicorn app:app --reload`
