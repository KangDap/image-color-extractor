import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import io
import hashlib

# Fungsi konversi RGB ke HEX
def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

# Fungsi hash gambar untuk deteksi perubahan
def get_image_hash(image_bytes):
    return hashlib.md5(image_bytes).hexdigest()

# Cache proses ekstraksi warna dominan agar tidak berubah saat UI interaktif disentuh
@st.cache_resource
def get_dominant_colors(image_bytes, n_colors=5):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((200, 200))  # resize agar proses cepat
    img_np = np.array(image).reshape((-1, 3))  # 3 channel (R, G, B)

    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    kmeans.fit(img_np)

    colors = kmeans.cluster_centers_.astype(int)
    return colors

# UI Streamlit
st.title("✨Dominant Color Picker from Image✨")

# Inisialisasi hash gambar sebelumnya
if 'last_image_hash' not in st.session_state:
    st.session_state.last_image_hash = None

st.write("Upload an image to extract its dominant colors.")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image_bytes = uploaded_file.read()
    image_hash = get_image_hash(image_bytes)

    # Cek apakah gambar berbeda dari sebelumnya
    is_new_image = image_hash != st.session_state.last_image_hash
    if is_new_image:
        st.session_state.last_image_hash = image_hash
        st.balloons()  # tampilkan balon hanya saat gambar baru

    image = Image.open(io.BytesIO(image_bytes))
    st.image(image, caption='Uploaded Image', use_container_width=True)

    with st.spinner("Extracting dominant colors..."):
        colors = get_dominant_colors(image_bytes, n_colors=5)
        hex_colors = [rgb_to_hex(color) for color in colors]

    st.subheader("Dominant Color Palette")
    cols = st.columns(5)
    for i, col in enumerate(cols):
        with col:
            st.markdown(
                f'<p style="font-size: 10px; color: gray;">Color {i+1}</p>',
                unsafe_allow_html=True
            )
            st.color_picker(f"Color {i+1}", hex_colors[i], label_visibility="collapsed")
            st.write(hex_colors[i])
            

    palette = np.zeros((100, 500, 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        palette[:, i * 100:(i + 1) * 100] = color

    fig, ax = plt.subplots(figsize=(5, 1.5))  # Tambah tinggi agar muat teks
    ax.imshow(palette)
    ax.axis('off')

    # Tambahkan kode HEX di bawah setiap blok warna
    for i, hex_color in enumerate(hex_colors):
        ax.text(i * 100 + 50, 110, hex_color, fontsize=8, ha='center', va='top', color='black')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    
    st.pyplot(fig)
    st.download_button(
        label="Download Color Palette",
        data=buf,
        file_name="color_palette.png",
        mime="image/png"
    )

st.markdown(
    """
    <div style='text-align: center; padding-top: 50px; font-size: 12px; color: gray;'>
        Made by Dafa Ghani Abdul Rabbani - 140810230022
    </div>
    """,
    unsafe_allow_html=True
)
