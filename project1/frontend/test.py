import streamlit as st
import requests
import streamlit.components.v1 as components

# Initialize the screen state using session state
if "screenstate" not in st.session_state:
    st.session_state.screenstate = {
        "login_page": True,
        "generate_reports": False,
    }

# Function to authenticate user and get token
def authenticate(username, password):
    try:
        response = requests.post(
            "http://127.0.0.1:8000/token/",
            data={"username": username, "password": password}
        )
        if response.status_code == 200:
            return response.json().get("access_token")
        else:
            st.error("Incorrect username or password.")
            return None
    except Exception as e:
        st.error(f"An error occurred during authentication: {str(e)}")
        return None

# JavaScript to interact with localStorage
save_token_js = """
<script>
    const saveToken = (token) => {
        localStorage.setItem('authToken', token);
    };
</script>
"""
get_token_js = """
<script>
    const getToken = () => {
        return localStorage.getItem('authToken');
    };
</script>
"""

# Streamlit App
st.title("Radiology Report Generator")

# Check if a token is saved in localStorage on page load
token = components.html(get_token_js, height=0)

if not token:
    if st.session_state.screenstate["login_page"]:
        # Authentication form
        with st.form("login_form"):
            st.write("Please log in to continue.")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login")

        # Authenticate user
        if login_button:
            token = authenticate(username, password)
            if token:
                st.session_state.token = token  # Store token in session state
                st.session_state.screenstate["login_page"] = False  # Hide login page
                st.session_state.screenstate["generate_reports"] = True  # Show report generator

                # Save token in localStorage
                components.html(f"<script>saveToken('{token}');</script>", height=0)

                st.success("Logged in successfully!")
                # st.rerun()
else:
    print("Token:", token)
    st.session_state.token = token
    st.session_state.screenstate["login_page"] = False
    st.session_state.screenstate["generate_reports"] = True
    # st.rerun()

# Show the report generation page if logged in
if st.session_state.screenstate["generate_reports"]:
    st.write("You are now logged in and can generate reports.")
    
    # File uploader for the chest X-ray image
    uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

    # Text input for the clinical indication
    indication = st.text_input("Clinical Indication", "Patient presenting with persistent cough, fever, and difficulty breathing. Evaluate for pneumonia.")

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Chest X-ray.', use_column_width=True)
        if st.button("Generate Report"):
            with st.spinner('Generating report...'):
                try:
                    img_str = uploaded_file.read()
                    response = requests.post(
                        "http://127.0.0.1:8000/generate_report/",
                        headers={"Authorization": f"Bearer {st.session_state.token}"},
                        files={"file": ("image.jpg", img_str, uploaded_file.type)},
                        data={"indication": indication}
                    )
                    if response.status_code == 200:
                        data = response.json()
                        report = data.get("report", "No report generated.")
                        st.success("Report generated successfully!")
                        st.write(f"**Report:** {report}")
                    else:
                        st.error(f"Error generating the report. Status code: {response.status_code}")
                        st.write(response.text)  # Print the error details
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
