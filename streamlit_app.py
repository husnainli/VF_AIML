import streamlit as st
from PIL import Image
import io
import threading
from concurrent.futures import ThreadPoolExecutor

# Title
st.title("🌀 Image Rotator - Parallel Edition")

# Sidebar options
angle = st.sidebar.slider("Rotation Angle (degrees)", -360, 360, 90)
max_threads = st.sidebar.slider("Number of Threads", 1, 8, 4)

st.markdown("### Upload Images to Rotate")

uploaded_files = st.file_uploader("Choose image files", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

rotated_images = []

# Function to rotate image
def rotate_image(image_file, angle):
    try:
        img = Image.open(image_file)
        rotated = img.rotate(angle)
        rotated_images.append((image_file.name, rotated))
    except Exception as e:
        st.error(f"Error processing {image_file.name}: {str(e)}")

# Multithreaded rotation
if uploaded_files and st.button("Rotate Images"):
    rotated_images.clear()
    with st.spinner("Rotating images..."):

        # Using ThreadPoolExecutor for parallelism
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            for image_file in uploaded_files:
                executor.submit(rotate_image, image_file, angle)

    st.success("Rotation complete!")

    # Display rotated images
    st.markdown("### 🔄 Rotated Images")
    for name, rotated in rotated_images:
        st.markdown(f"**{name}**")
        st.image(rotated, use_column_width=True)

        # Optionally provide download
        buf = io.BytesIO()
        rotated.save(buf, format="PNG")
        st.download_button("Download", data=buf.getvalue(), file_name=f"rotated_{name}", mime="image/png")
