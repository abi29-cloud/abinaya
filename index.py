import streamlit as st
import sqlite3

# Function to connect to SQLite database
def get_db_connection():
    conn = sqlite3.connect('user_credentials.db')
    conn.row_factory = sqlite3.Row  # to return rows as dictionaries
    return conn

# Create the table for storing user credentials (if it doesn't exist)
def create_user_table():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        email TEXT NOT NULL,
        password TEXT NOT NULL
    )
    ''')
    conn.commit()
    conn.close()

# Registration function to insert user into the database
def register_user(username, email, password):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
              (username, email, password))
    conn.commit()
    conn.close()

# Login validation function
def validate_user(username, password):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    user = c.fetchone()
    conn.close()
    return user

# Create the table if it doesn't exist
create_user_table()

# Streamlit Sidebar for page selection
page = st.sidebar.radio("Select a page", ("Registration", "Login"))

if page == "Registration":
    # Streamlit Registration Page
    st.title("User Registration")

    # Create input fields for registration
    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    # Placeholder for displaying registration status
    status = st.empty()

    if st.button("Register"):
        if password != confirm_password:
            status.error("Passwords do not match!")
        else:
            # Save user credentials in the database
            register_user(username, email, password)
            status.success("Registration successful!")

    # Display user input for debugging purposes (can be removed later)
    st.write(f"Username: {username}")
    st.write(f"Email: {email}")

elif page == "Login":
    # Streamlit Login Page
    st.title("Login Page")

    # Create a form for user input
    with st.form("login_form"):
        login_username = st.text_input("Username")
        login_password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")

    if submit_button:
        user = validate_user(login_username, login_password)
        if user:
            st.success("Login successful!")
            st.write("Welcome to the application!")
            import subprocess
            subprocess.run(['streamlit','run','app.py'])
        else:
            st.error("Invalid username or password. Please try again.")