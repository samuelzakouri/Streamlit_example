import streamlit as st
import pickle
import numpy as np

with open('iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('iris_target_names.pkl', 'rb') as f:
    iris_names = pickle.load(f)


def predict_species(sepal_length, sepal_width,  petal_length, petal_width):
    input = np.array([[sepal_length, sepal_width,  petal_length, petal_width]]).astype(np.float64)
    prediction = model.predict(input)

    return int(prediction)

def main():
    st.title("Iris Species Prediction")
    html_temp = """
       <div style="background:#025246 ;padding:10px">
       <h2 style="color:white;text-align:center;"> Iris Species Prediction ML App </h2>
       </div>
       """
    st.markdown(html_temp, unsafe_allow_html=True)

    sepal_length = st.text_input("Sepal length", "Type Here")
    sepal_width = st.text_input("Sepal width", "Type Here")
    petal_length = st.text_input("Petal length", "Type Here")
    petal_width = st.text_input("Petal width", "Type Here")


    setosa_html = """  
          <div style="background-color:#80ff80; padding:10px >
          <h2 style="color:white;text-align:center;"> The species is setosa</h2>
          </div>
        """
    versi_html = """  
          <div style="background-color:#F4D03F; padding:10px >
          <h2 style="color:white;text-align:center;"> The species is versicolor</h2>
          </div>
        """
    virgi_html = """  
          <div style="background-color:#F08080; padding:10px >
           <h2 style="color:black ;text-align:center;"> The species is virginica</h2>
           </div>
        """

    if st.button("Predict the species"):
        output = predict_species(sepal_length, sepal_width,  petal_length, petal_width)
        st.success('The species is {}'.format(iris_names[output]))

        if output == 0:
            st.markdown(setosa_html, unsafe_allow_html=True)
        elif output == 1:
            st.markdown(versi_html, unsafe_allow_html=True)
        elif output == 2:
            st.markdown(virgi_html, unsafe_allow_html=True)

if __name__=='__main__':
    main()