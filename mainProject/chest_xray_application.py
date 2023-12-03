import streamlit as st
import sqlite3
import hashlib
import os
import requests
from PIL import Image
import io
from textblob import TextBlob
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

nltk.download('punkt')

def analyze_sentiment(caption):
    analysis = TextBlob(caption)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

def plot_sentiment_distribution(analyzed_captions):
    sentiments = [sentiment for _, sentiment in analyzed_captions]
    plt.figure(figsize=(3, 3))
    sns.countplot(sentiments)

    plt.xlabel('Sentiment')
    plt.ylabel('Number of Captions')
    plt.title('Sentiment Distribution of Captions')
    st.pyplot(plt)

def plot_pie_chart(sentiments):
    sentiment_counts = pd.Series(sentiments).value_counts()
    plt.figure(figsize=(3, 3))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Sentiment Distribution in Pie Chart')
    st.pyplot(plt)

# Function to create a connection to the SQLite database
def create_connection():
    conn = sqlite3.connect('xray_app.db', check_same_thread=False)
    return conn

# Function to create tables in the database
def create_tables():
    conn = create_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
                        username TEXT PRIMARY KEY,
                        password TEXT,
                        role TEXT
                    );''')
    conn.execute('''CREATE TABLE IF NOT EXISTS patient_data (
                        username TEXT PRIMARY KEY,
                        front_image_path TEXT,
                        lateral_image_path TEXT,
                        captions TEXT,
                        doctor_response TEXT
                    );''')
    conn.close()

# Function to hash passwords for storage
def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

# Function for user signup
def signup_user(username, password, role):
    conn = create_connection()
    hashed_pw = hash_password(password)
    try:
        conn.execute('''INSERT INTO users (username, password, role)
                        VALUES (?, ?, ?);''', (username, hashed_pw, role))
        conn.commit()
        st.success("User created successfully.")
    except sqlite3.IntegrityError:
        st.error("Username already exists.")
    conn.close()

# Function for user login
def login_user(username, password):
    conn = create_connection()
    hashed_pw = hash_password(password)
    cursor = conn.execute('''SELECT role FROM users WHERE
                             username=? AND password=?;''', (username, hashed_pw))
    data = cursor.fetchone()
    conn.close()
    if data:
        # Set session state
        st.session_state['username'] = username
        st.session_state['role'] = data[0]
    return data

# Function to save uploaded file
def save_uploaded_file(directory, img, img_name):
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(directory, img_name), "wb") as f:
        f.write(img.getbuffer())
    return os.path.join(directory, img_name)

# Function to generate captions
def generate_caption(image1_path, image2_path, num_captions):
    url = 'http://432b-34-132-228-160.ngrok.io/predict'
    files = {
        'img1': open(image1_path, 'rb'),
        'img2': open(image2_path, 'rb')
    }
    data = {
        'num_captions': num_captions
    }
    response = requests.post(url, files=files, data=data)
    files['img1'].close()
    files['img2'].close()
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error: {response.status_code}")
# Function to get doctor's response for a specific patient
def get_doctors_response(patient_username):
    conn = create_connection()
    cursor = conn.execute("SELECT doctor_response FROM patient_data WHERE username=?", (patient_username,))
    response = cursor.fetchone()
    conn.close()
    return response[0] if response else None

# Patient Dashboard
def patient_dashboard(username):
    st.title("Patient")
    st.markdown(f"Welcome, {username}, to your personalized patient dashboard.")
    st.write("Here you can view your medical records, upcoming appointments, and contact your doctor.")

    # List registered doctors
    registered_doctors = list_registered_doctors()
    for doctor in registered_doctors:
        st.write(doctor)

    # Image upload and caption generation
    st.markdown("## X-ray Analysis")
    st.warning("Upload your X-ray images for analysis:")

    # File uploaders
    image1 = st.file_uploader("🩺 Upload the front image", type=["png", "jpg", "jpeg"], key="1")
    image2 = st.file_uploader("🔍 Upload the lateral image", type=["png", "jpg", "jpeg"], key="2")

    # Display uploaded images in expanders
    if image1 or image2:
        col1, col2 = st.columns(2)
        if image1:
            with col1:
                with st.expander("🖼️ Front Image"):
                    st.image(Image.open(io.BytesIO(image1.getvalue())), use_column_width=True)
        if image2:
            with col2:
                with st.expander("🖼️ Lateral Image"):
                    st.image(Image.open(io.BytesIO(image2.getvalue())), use_column_width=True)

    # Select number of captions
    num_captions = st.slider("🔢 Select how many captions to generate", 1, 5, 1)
    # Inside the patient_dashboard function or wherever you generate the captions
    if st.button("🚀 Generate Captions"):
        try:
            if image1 is not None and image2 is not None:
                with st.spinner('⏳ Processing images...'):
                    path1 = save_uploaded_file("uploaded_images", image1, image1.name)
                    path2 = save_uploaded_file("uploaded_images", image2, image2.name)
                    captions = generate_caption(path1, path2, num_captions)
                    st.markdown('# 📝 Generated Captions')
                    for caption in captions:
                        st.markdown(f"* {caption}")
                    # Save images and captions to the database
                    save_patient_data(username, path1, path2, captions)
                    st.success("✅ Image Captions Successfully Generated and Saved!")

                    analyzed_captions = [(caption, analyze_sentiment(caption)) for caption in captions]
                    st.markdown('# 📝 Generated Captions with Sentiments')
                    for caption, sentiment in analyzed_captions:
                        st.markdown(f"* {caption} - {sentiment}")

                    # Plotting
                    sentiments = [sentiment for _, sentiment in analyzed_captions]
                    st.markdown("## Sentiment Analysis Visualization")
                    with st.expander("🖼️ Bar Plot"):
                         plot_sentiment_distribution(analyzed_captions)
                    with st.expander("🖼️ Pie Chart"):
                         plot_pie_chart(sentiments)

        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")

    # Send images and captions to a selected doctor
    st.markdown("## Send Images and Captions to Doctor")
    selected_doctor = st.selectbox("Select Doctor:", registered_doctors, key=f"doctor_selector_{username}")
    if st.button("📤 Send Images and Captions"):
        send_images_and_captions(username, selected_doctor)
        st.success(f"Images and captions sent to {selected_doctor}.")

    if st.button("Back to Home 🏠"):
        st.session_state['role'] = None
        st.rerun()

    # Show doctor's response
    doctor_response = get_doctors_response(username)
    if doctor_response:
        st.markdown("## Doctor's Response 🩺")
        st.info(doctor_response)
    else:
        st.markdown("## Doctor's Response 🩺")
        st.warning("No response from the doctor yet.")

# Function to list registered doctors
def list_registered_doctors():
    conn = create_connection()
    cursor = conn.execute("SELECT username FROM users WHERE role='Doctor';")
    doctors = [doctor[0] for doctor in cursor.fetchall()]
    conn.close()
    return doctors

# Function to save patient data to the database
def save_patient_data(username, front_image_path, lateral_image_path, captions):
    conn = create_connection()
    try:
        conn.execute('''INSERT INTO patient_data (username, front_image_path, lateral_image_path, captions)
                        VALUES (?, ?, ?, ?);''', (username, front_image_path, lateral_image_path, ",".join(captions)))
        conn.commit()
    except sqlite3.IntegrityError:
        # Handle integrity error if needed
        pass
    conn.close()

# Function to clear all data in the tables
def clear_all_data():
    conn = create_connection()
    conn.execute("DELETE FROM users;")
    conn.execute("DELETE FROM patient_data;")
    conn.commit()
    conn.close()
# Function to send images and captions to a selected doctor
def send_images_and_captions(patient_username, doctor_username):
    conn = create_connection()
    cursor = conn.execute("SELECT front_image_path, lateral_image_path, captions FROM patient_data WHERE username=?", (patient_username,))
    data = cursor.fetchone()
    if data:
        front_image_path, lateral_image_path, captions = data
        # Add code to send images and captions to the selected doctor
        # For demonstration purposes, print the data
        print(f"Sending data to {doctor_username}:\nFront Image Path: {front_image_path}\nLateral Image Path: {lateral_image_path}\nCaptions: {captions}")
    conn.close()

# Doctor Dashboard
def doctor_dashboard():
    st.title("Doctor")
    st.markdown("Welcome to your professional, Doctor.")
    st.write("Here you can access patient records, manage appointments, and review medical images.")

    # Doctor login system
    if 'username' not in st.session_state or st.session_state['username'] is None:
        st.warning("Please log in as a doctor.")
        return

    # Display images and captions from specific patients
    st.markdown("## Patient Records")
    patient_records = get_patient_records(st.session_state['username'])
    for index, record in enumerate(patient_records):
        with st.container():  # Use containers to isolate widgets and avoid key clashes
            st.subheader(f"Patient: {record['username']}")
            st.write(f"Front Image Path: {record['front_image_path']}")
            st.write(f"Lateral Image Path: {record['lateral_image_path']}")

            # Display the front image
            front_image = Image.open(record['front_image_path'])
            st.image(front_image, caption="Front Image", use_column_width=True)

            # Display the lateral image
            lateral_image = Image.open(record['lateral_image_path'])
            st.image(lateral_image, caption="Lateral Image", use_column_width=True)

            # Display the captions
            st.markdown("### Captions:")
            captions = record['captions'].split(',')
            for caption in captions:
                st.write(f"* {caption.strip()}")

            # Provide a response mechanism for doctors
            response = st.text_area("Your Response:", key=f"response_{record['username']}_{index}")
            if st.button(f"Send Response for {record['username']}", key=f"send_response_{record['username']}_{index}"):
                send_response(record['username'], response)
                st.success("Response sent successfully.")

    # Unique key for back button outside of the loop to avoid duplication
    if st.button("Back to Home 🏠", key="back_home_key"):
        st.session_state['role'] = None
        st.experimental_rerun()


# Function to retrieve patient records for a specific doctor
def get_patient_records(doctor_username):
    conn = create_connection()
    cursor = conn.execute("SELECT username, front_image_path, lateral_image_path, captions FROM patient_data;")
    records = [{'username': record[0], 'front_image_path': record[1], 'lateral_image_path': record[2], 'captions': record[3]} for record in cursor.fetchall()]
    conn.close()
    return records

# Function to send a response to a patient
def send_response(patient_username, response):
    conn = create_connection()
    try:
        conn.execute("UPDATE patient_data SET doctor_response=? WHERE username=?", (response, patient_username))
        conn.commit()
    except sqlite3.IntegrityError:
        # Handle integrity error if needed
        pass
    finally:
        conn.close()


# Enhanced Home page layout with session state check
def home_page():
    if 'role' not in st.session_state or st.session_state['role'] is None:
        st.title("Smart AI XRay Analyzer")
        st.image("AI-7.jpg", width=600)  # Placeholder image
        st.subheader("Welcome to Smart AI XRay Analyzer, your smart assistant for X-ray image analysis 🩺💻")
        st.markdown("""
                ### About Smart AI XRay Analyzer
                XRay CaptionPro is an innovative solution designed to assist medical professionals and patients in the analysis of X-ray images.
                - 🧑‍⚕️ For doctors, it offers detailed insights and automated captioning of X-ray images.
                - 👩‍🔬 For patients, it provides easy access to their X-ray analyses and doctor's notes.
                Enjoy a streamlined, efficient, and user-friendly experience with XRay CaptionPro!
        """)

        # Show the login/signup options if no user is logged in
        menu = ["Home", "Login", "Signup"]
        choice = st.sidebar.selectbox("Menu", menu)

        if choice == "Login":
            username = st.sidebar.text_input("Username")
            password = st.sidebar.text_input("Password", type='password')
            if st.sidebar.button('Login'):
                role = login_user(username, password)
                if role:
                    st.session_state['role'] = role[0]
                    st.session_state['username'] = username
                    st.experimental_rerun()
                else:
                    st.error("Incorrect username or password.")
        elif choice == "Signup":
            new_user = st.text_input("Username")
            new_password = st.text_input("New Password", type='password')
            role = st.selectbox("Role", ["Patient", "Doctor"])
            if st.button('Signup'):
                signup_user(new_user, new_password, role)

        # Add an option to clear all data
        if st.sidebar.button('Clear All Data'):
            clear_all_data()
            st.success("All data cleared successfully.")
    else:
        # Redirect to appropriate dashboard based on session state
        if st.session_state['role'] == 'Patient':
            patient_dashboard(st.session_state['username'])
        elif st.session_state['role'] == 'Doctor':
            doctor_dashboard()

# Main app logic
def main():
    create_tables()
    if 'role' not in st.session_state:
        st.session_state['role'] = None
    home_page()

if __name__ == "__main__":
    main()