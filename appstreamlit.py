import numpy as np
import joblib
import streamlit

iris_model = joblib.load(r"IRIS_dataset_model.pki")

def iris_prediction(var_1, var_2,var_3,var_4):
    pred_array=np.array([var_1,var_2,var_3, var_4])
    preds = pred_array.reshape(1,-1)
    preds=preds.astype(float)
    predictions = iris_model.predict(preds)
    return predictions

def run():
    streamlit.title("IRIS Model")

    var_1=streamlit.text_input("Variable 1")
    var_2=streamlit.text_input("Variable 2")
    var_3=streamlit.text_input("Variable 3")
    var_4=streamlit.text_input("Variable 4")
    
    prediction = ""
    
    if streamlit.button("Predict"):
        prediction=iris_prediction(var_1, var_2, var_3, var_4)
        
    streamlit.success("The prediction by model : {}".format(prediction))
    
if __name__=='__main__':
    run()


