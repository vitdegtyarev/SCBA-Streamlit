# SCBA-Streamlit

This Streamlit application predicts the elastic buckling and ultimate loads of steel cellular beams using optimized ML models, including decision tree (DT), random forest (RF), k-nearest neighbor (KNN), gradient boosting regressor (GBR), extreme gradient boosting (XGBoost), light gradient boosting machine (LightGBM), and gradient boosting with categorical features support (CatBoost).

The application is based on the models described in the following paper: Degtyarev, V. V., & Tsavdaridis, K. D. (2022). Buckling and ultimate load prediction models for perforated steel beams using machine learning algorithms. Journal of Building Engineering, 104316, doi:10.1016/j.jobe.2022.104316

## Instructions on how to use the application

1. Download the content of the repository to the local machine.
2. With Python installed, install the required packages listed in the 'requirements.txt' file.
3. Open command-line interface (cmd.exe).
4. Change to the directory on the local machine where the application was saved.
5. Type the following: streamlit run SCBA.py
6. The application opens in a web browser tab.
7. Use sliders and radio buttons to change beam parameters.

![This is an image](GUI_App.png)
