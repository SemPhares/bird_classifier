import io
import torch
from torchvision import transforms
from PIL import Image, ImageOps
import streamlit as st
import streamlit.components.v1 as components


#load model
device = "cuda" if torch.cuda.is_available() else "cpu"

st.session_state.model = torch.load('models/mobilenet_bird_classifer.pkl', map_location=torch.device(device))
st.session_state.classes =  torch.load('models/labels_classes.pkl')

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 224

st.session_state.test_transforms = transforms.Compose(
    [
        transforms.Resize(size=(IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
)

#set up streamlit app header

with st.columns(3)[1]:
    st.header("Bird Classifier")
    # st.markdown("<h1 style='text-align: center; color: black;'>Bird Classifier</h1>", unsafe_allow_html=True)
    st.image("images/bird-app.png", width=200)


#initiate lists
images_names = []
ls_images = []
predictions = []
probabilities = []

#load images
uploaded_files = st.file_uploader("Veuillez charger une image",\
    type=['jpg','jpeg','png'],\
    help="Charger une image au format jpg,jpeg,png", \
        accept_multiple_files=True)

#compute prediction
for uploaded_file in uploaded_files:
    try:
        bytes_data = uploaded_file.read()
        image = Image.open(io.BytesIO(bytes_data))
        # Corriger les orientations EXIF et forcer un mode compatible
        image = ImageOps.exif_transpose(image)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        elif image.mode == "L":
            # Le modèle attend 3 canaux; convertir si nécessaire
            image = image.convert("RGB")

        images_names.append(uploaded_file.name)
        trans_img = st.session_state.test_transforms(image)
        pred_vector = st.session_state.model(trans_img.unsqueeze(0).to(device))
        pred = torch.argmax(pred_vector, dim=1).item()
        label = st.session_state.classes[pred]
        prob = torch.max(torch.softmax(pred_vector, dim=1)).item() * 100
        predictions.append(label)
        ls_images.append(image)
        probabilities.append(prob)
    except Exception as e:
        st.error(f"Échec du traitement de l'image {uploaded_file.name}: {e}")


submit = st.button("Submit")

if len(ls_images)>0:
    tabs = st.tabs(['image_'+str(i) for i in range(1, len(ls_images)+1)])

#display prediction
if submit:
    for i in range(0, len(ls_images)):
        
        with tabs[i]:
            st.image(ls_images[i], width= 300)
            st.write('Predicted :', predictions[i])
            st.write('Porbability :', probabilities[i])
            st.write('True label :',images_names[i])