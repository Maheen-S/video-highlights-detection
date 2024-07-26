import streamlit as st
import external_script
import tempfile
import os

# Title of the app
st.title('Cricket Highlight Detection')

# File uploader for the video
uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    st.video(uploaded_video)
    st.success("Video uploaded and playable!")

    # Add a button to detect highlights
    if st.button('Detect Highlights'):
        st.write("Highlight detection in progress...")
        
        # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(uploaded_video.read())
            temp_video_path = temp_file.name
        
        # Run the external script to process the video and get the output path
        result = external_script.main_running_code(temp_video_path)
        
        st.success(result)
        st.success("Highlights detected and Video processed!")

        processed_video_path = "final_video1.mp4"  
        
        # Check if the processed video exists
        if os.path.exists(processed_video_path):
            st.video(processed_video_path)

            # Provide the download button for the processed video
            with open(processed_video_path, "rb") as file:
                btn = st.download_button(
                    label="Download Video",
                    data=file,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )

        else:
            st.error("Processed video not found.")
else:
    st.write("Please upload a video file to proceed.")
