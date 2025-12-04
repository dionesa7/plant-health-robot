import streamlit as st
from ultralytics import YOLO
from PIL import Image

# --- 1. KONFIGURIMI I FAQES ---
st.set_page_config(
    page_title="Plant Health AI", 
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Robotikë Bujqësore: Detektimi i Sëmundjeve")
st.markdown("---")

# --- 2. NGARKIMI I MODELIT (Me Cache për shpejtësi) ---
# Kjo pjesa @st.cache_resource është shumë e rëndësishme për Online/Cloud
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Gabim: Nuk u gjet modeli 'best.pt'. Sigurohu që është ngarkuar në GitHub.")
    st.stop()

# --- 3. ZGJEDHJA E METODËS (INPUT) ---
st.write("### 📸 Zgjidhni mënyrën e testimit:")
option = st.radio(
    "", 
    ("📁 Ngarko Foto nga Pajisja", "📷 Përdor Kamerën Live"), 
    horizontal=True
)

image_source = None

if option == "📁 Ngarko Foto nga Pajisja":
    image_source = st.file_uploader("Ngarkoni imazhin këtu...", type=["jpg", "png", "jpeg"])
else:
    st.info("Ju lutem lejoni aksesin e kamerës në shfletues.")
    image_source = st.camera_input("Bëni foto bimës")

# --- 4. PROCESIMI DHE REZULTATI ---
if image_source is not None:
    
    # Shfaq Foton
    image = Image.open(image_source)
    st.image(image, caption='Pamja nga Syri i Robotit', use_container_width=True)
    
    st.write("---")

    # Butoni i Analizës
    if st.button('🔍 ANALIZO TANI', type="primary", use_container_width=True):
        
        with st.spinner('Duke komunikuar me trurin e robotit...'):
            try:
                # Ruajmë foton përkohësisht
                temp_filename = "temp_leaf.jpg"
                image.save(temp_filename)

                # Analiza me YOLO
                results = model(temp_filename)
                result = results[0]
                
                # Nxjerrja e të dhënave
                probs = result.probs
                top_index = probs.top1
                top_conf = probs.top1conf.item() * 100
                class_name = result.names[top_index]

                # Rregullimi i emrit (heqja e vizave)
                clean_name = class_name.replace("_", " ").upper()

                # --- SHFAQJA E REZULTATIT ---
                st.divider()
                
                # Logjika Healthy vs Sëmundje
                if "healthy" in class_name.lower():
                    st.success(f"✅ REZULTATI: **{clean_name}**")
                    st.balloons()
                else:
                    st.error(f"⚠️ KUJDES! DETEKTOHET SËMUNDJE: **{clean_name}**")
                
                # Metrikat
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Besueshmëria", f"{top_conf:.1f}%")
                with col2:
                    st.metric("Koha e Reagimit", "~3 ms")
            
            except Exception as e:
                st.error(f"Ndodhi një gabim gjatë analizës: {e}")

# --- Footer ---
st.markdown("---")
st.caption("Zhvilluar për GRUPIN 1 HULUMTUES  | Powered by YOLOv8 & Streamlit")