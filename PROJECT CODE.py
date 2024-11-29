import os
import base64
import io
from dotenv import load_dotenv
from PIL import Image
import pyttsx3
import pytesseract
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("API key not found! Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

# Initialize Chat Model
chat_model = ChatGoogleGenerativeAI(api_key=api_key, model="gemini-1.5-flash")

# App Title
st.title("Vision for the Future: AI Solutions Empowering the Visually Impaired 👁️🤖💡")

# Helper: Text-to-Speech Conversion

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed
    engine.setProperty('volume', 1)  # Volume
    audio_file = "text-to-speech-local.mp3"
    
    try:
        engine.save_to_file(text, audio_file)
        engine.runAndWait()
        st.audio(audio_file, format="audio/mp3")
    except Exception as e:
        st.error(f"Audio generation failed: {e}")


# Feature 1: Real-Time Scene Understanding
def real_time_scene_understanding(image_base64):
    hmessage = HumanMessage(
        content=[
            {"type": "text", "text": """You are a real-time scene interpreter for visually impaired users. Your task is to analyze and describe images vividly, empathetically, and without technical jargon. Focus on delivering concise, actionable information that enhances understanding and safety.

                Description Guidelines:

                Scene Overview: Summarize the setting (e.g., indoor/outdoor, type of location).
                Key Objects & Layout: Describe objects with details like position, color, size, shape, and texture.
                Activities & Interactions: Highlight actions or interactions (e.g., "A person jogging with a dog").
                Mood & Atmosphere: Describe the tone (e.g., "Lively with bright sunlight").
                Text & Symbols: Transcribe visible text or signs.
                Sensory Details: Mention implied sounds, smells, or sensations.
                Accessibility & Safety: Identify potential hazards or challenges (e.g., "A step down near the doorway").
                Formatting Tips:

                Use short, clear sentences.
                Structure from general to specific details.
                Maintain a neutral, empathetic tone."""},
            {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"}
        ]
    )
    try:
        response = chat_model.invoke([hmessage])
        response_text = response.content
        st.write(response_text)
        text_to_speech(response_text)
    except Exception as e:
        st.error(f"Scene understanding failed: {e}")

# Feature 2: Text Extraction
def text_extraction(uploaded_image):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    try:
        extracted_text = pytesseract.image_to_string(uploaded_image)
        #st.session_state.text = extracted_text
        st.write(extracted_text)
        text_to_speech(extracted_text)
    except Exception as e:
        st.error(f"Text extraction failed: {e}")

# Feature 3: Object Detection
def object_detection(image_base64):
    hmessage = HumanMessage(
        content=[
            {"type": "text", "text": """"You are a visual accessibility specialist analyzing images to help visually impaired individuals navigate safely. Your goal is to provide detailed yet concise descriptions of visible objects and obstacles, prioritizing safety and situational awareness.

                Guidelines:

                Object and Obstacle Identification:
                List all visible objects, obstacles, or features.
                Highlight items critical for safety, such as steps, curbs, sharp edges, or spills.
                Detailed Descriptions:
                Describe each object/obstacle in terms of location (e.g., "top left corner"), size, and appearance (e.g., color, shape, texture).
                Include distinguishing features for better visualization.
                Safety Insights:
                Explain the significance or purpose of each item.
                For safety hazards, provide clear, actionable guidance (e.g., "A low-hanging branch is at head height in the center of the path—duck to avoid it").
                Communication Style:

                Use simple, accessible language. Avoid technical terms.
                Incorporate directional terms and approximate measurements.
                Maintain a calm, supportive tone to promote confidence and independence.
                Use bullet points or numbered lists for clarity."""},
            {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"}
        ]
    )
    try:
        response = chat_model.invoke([hmessage])
        response_text = response.content
        st.write(response_text)
        text_to_speech(response_text)
    except Exception as e:
        st.error(f"Object detection failed: {e}")

# Feature 4: Personalized Assistance
def personal_assistance(image_base64):
    hmessage = HumanMessage(
        content=[
            {"type": "text", "text": """As an assistive technology specialist, your role is to provide personalized, context-specific support for visually impaired users. Deliver clear, actionable descriptions to empower users with confidence and enhance accessibility.

                Tasks:

                Identify and Describe Items:
                List all objects, landmarks, or features visible in the image.
                Provide concise descriptions of size, color, shape, position, and distinguishing attributes.
                Interpret Text and Labels:
                Transcribe visible text (e.g., signs, labels) accurately.
                Explain the purpose or context (e.g., directions, warnings) for clarity.
                Context-Specific Guidance:
                Suggest practical actions based on the image (e.g., identifying products on a shelf).
                Explain interactions between objects (e.g., “The red button is below the green switch”).
                Formatting and Tone:

                Use clear, supportive language, avoiding jargon.
                Organize information logically with bullet points or lists.
                Acknowledge ambiguities transparently, offering the best interpretation where needed."""},
            {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"}
        ]
    )
    try:
        response = chat_model.invoke([hmessage])
        response_text = response.content
        st.write(response_text)
        text_to_speech(response_text)
    except Exception as e:
        st.error(f"Personalized assistance failed: {e}")


# Upload and process the image
uploaded_image = st.file_uploader("📤Upload an image", type=["jpg", "jpeg", "png"])
    
# Feature selection
feature = st.sidebar.radio(
    "Select a functionality🔧:",
    [
        "Real-Time Scene Understanding",
        "Text-to-Speech Conversion",
         "Object Detection",
        "Personalized Assistance",
    ],
    index=0,
)

if uploaded_image:
    # Display uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode()


    

    # Trigger corresponding feature
    if feature == "Real-Time Scene Understanding":
        if st.button("🔍 Run Scene Understanding"):
            with st.spinner(" Analyzing the scene and converting the response to audio, Be Patience...!"):
                real_time_scene_understanding(image_base64)
                
    elif feature == "Text-to-Speech Conversion":
        if st.button("📝 Convert Text-to-Speech"):
            with st.spinner("Extracting text from the image and generating audio, Be Patience...!"):
                text_extraction(image)
                
    elif feature == "Object Detection":
        if st.button("🕵️‍♂️ Run Object Detection"):
            with st.spinner("Detecting the objects and generating audio, Be Patience...!"):
                object_detection(image_base64)
                
    elif feature == "Personalized Assistance":
        if st.button("💡Run Personalized Assistance"):
            with st.spinner("Providing personalized guidance and generating audio, Be Patience...!"):
                personal_assistance(image_base64)

else:
    st.info("Please Upload Image to Proceed...!")        
                


