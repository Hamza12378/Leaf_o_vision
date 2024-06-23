import pandas as pd
import streamlit as st
import tensorflow as tf
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as MobileNetV2_preprocess_input

# App title
st.title('Leaf Diseases Identifier Web App ')

## APP info    
st.write('''
## About

Plant diseases pose a significant threat to global food security, but quickly identifying them remains challenging and time-consuming.
Agro-vision aims to offer a more efficient and rapid method to assist non-experts in identifying unhealthy rice plants from digital images
and detect diseases that attack the corn leaves. 


Now, Agrovision is revolutionizing disease detection in corn leaves by replacing traditional techniques such as visual inspection. 
Instead of relying on subjective assessments, farmers can now accurately identify diseases in corn plants.

 
This web app employs a Deep Learning model to identify diseases (such as Northern Leaf Blight, Common Rust, and Gray Leaf Spot) that
impact corn leaves, using digital images

Made by Asfan and Hamza.

''')
st.markdown(
         f"""
         <style>
         .stApp {{
            
            background: linear-gradient(135deg, rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0.75)),
                url("https://images.unsplash.com/photo-1524486361537-8ad15938e1a3?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1469&q=80");
            background-size: cover;
            box-shadow: 0px 0px 10px 5px rgba(0, 0, 0, 0.75);
         }}
         </style>
         """,
         unsafe_allow_html=True
)

## load file
st.sidebar.write("# File Required")
uploaded_image = st.sidebar.file_uploader('', type=['jpg','png','jpeg'])

################### Class Dict and Dataframe of Probabilites #############################
# Map class
map_class = {
        0:'Northern Leaf Blight',
        1:'Common Rust',
        2:'Gray Leaf Spot',
        3:'Healthy'
        }
        
#Dataframe 
dict_class = {
        'Corn Leaf Condition': ['Northern Leaf Blight', 'Common Rust','Gray Leaf Spot','Healthy'],
        'Confiance': [0,0,0,0]
        }
        
df_results = pd.DataFrame(dict_class, columns = ['Corn Leaf Condition', 'Confiance'])
    
def predictions(preds):
    df_results.loc[df_results['Corn Leaf Condition'].index[0], 'Confiance'] = preds[0][0]
    df_results.loc[df_results['Corn Leaf Condition'].index[1], 'Confiance'] = preds[0][1]
    df_results.loc[df_results['Corn Leaf Condition'].index[2], 'Confiance'] = preds[0][2]
    df_results.loc[df_results['Corn Leaf Condition'].index[3], 'Confiance'] = preds[0][3]

    return (df_results)          

########################################### Load the model #########################
#@st.cache
def get_model():

    model = tf.keras.models.load_model("model_mobnetv2")
    return model

if __name__=='__main__':
    
    # Model
    model = get_model()

    # Image preprocessing
    if not uploaded_image:
        st.sidebar.write('Please upload an image before preceeding!')
        st.stop()
    else:
        # Decode the image and Predict the class
        img_as_bytes = uploaded_image.read() # Encoding image
        st.write("## Corn Leaf Image")
        st.image(img_as_bytes, use_column_width= True) # Display the image
        img = tf.io.decode_image(img_as_bytes, channels = 3) # Convert image to tensor
        img = tf.image.resize(img,(224,224)) # Resize the image
        img_arr = tf.keras.preprocessing.image.img_to_array(img) # Convert image to array
        img_arr = tf.expand_dims(img_arr, 0) # Create a bacth

    img = MobileNetV2_preprocess_input(img_arr)

    Genrate_pred = st.button("Detect Result") 
 
    if Genrate_pred:
        st.subheader('Probabilities by Class') 
        preds = model.predict(img)
        preds_class = model.predict(img).argmax()

        st.dataframe(predictions(preds))

        if (map_class[preds_class]=="Northern Leaf Blight") or (map_class[preds_class]=="Common Rust") or (map_class[preds_class]=="Gray Leaf Spot"): 
            st.subheader("The Corn Leaf is infected by {} disease".format(map_class[preds_class]))

        else:
            st.subheader("The Corn Leaf is {}".format(map_class[preds_class]))
