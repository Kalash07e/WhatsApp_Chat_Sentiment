import streamlit as st
import json
import hashlib
import os
from datetime import datetime
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import secrets
import time

# User database file
USER_DB_FILE = "user_database.json"

# Admin configuration - Add your email here to become an admin
ADMIN_EMAILS = [
    # "your_email@example.com",  # Replace with your actual email address
    # "admin@example.com"        # Add more admin emails if needed
]

def is_admin_user(email):
    """Check if the given email has admin privileges"""
    return email.lower() in [admin.lower() for admin in ADMIN_EMAILS]

def configure_admin_access():
    """Configure admin access - to be called once by the website owner"""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #FF6B35, #F7931E);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    ">
        <h2 style="color: white; margin-bottom: 15px;">⚙️ Admin Configuration Required</h2>
        <p style="color: white; font-size: 1.1rem; margin: 0;">
            No admin users configured yet. Please set up admin access.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("🔧 **Set Up Admin Access** (Website Owner Only)", expanded=True):
        st.markdown("""
        **Instructions for Website Owner:**
        1. Enter your email address below
        2. This email will have admin privileges to view all users
        3. You can add more admin emails later by editing the `auth.py` file
        
        **Security Note:** Only trusted users should be given admin access.
        """)
        
        admin_email = st.text_input(
            "Enter Admin Email Address:",
            placeholder="your-email@example.com",
            help="This email will have admin privileges to view user data"
        )
        
        if st.button("🚀 **Configure Admin Access**", type="primary"):
            if admin_email and is_valid_email(admin_email)[0]:
                # Update the auth.py file with the admin email
                try:
                    # Read current auth.py content
                    with open('auth.py', 'r') as f:
                        content = f.read()
                    
                    # Replace the ADMIN_EMAILS configuration
                    new_config = f'''ADMIN_EMAILS = [
    "{admin_email}",  # Website owner - configured via admin setup
    # "additional@admin.com"  # Add more admin emails as needed
]'''
                    
                    # Update the content
                    import re
                    pattern = r'ADMIN_EMAILS = \[[^\]]*\]'
                    updated_content = re.sub(pattern, new_config, content, flags=re.DOTALL)
                    
                    # Write back to file
                    with open('auth.py', 'w') as f:
                        f.write(updated_content)
                    
                    st.success(f"✅ **Admin access configured!** {admin_email} is now an admin.")
                    st.info("🔄 **Please restart the application** for changes to take effect.")
                    st.balloons()
                    
                    # Clear cache to force reload
                    st.cache_data.clear()
                    
                except Exception as e:
                    st.error(f"❌ **Configuration failed:** {str(e)}")
                    st.info("💡 **Manual setup:** Edit `auth.py` and add your email to the `ADMIN_EMAILS` list.")
            else:
                st.error("❌ **Invalid email address.** Please enter a valid email.")
    
    return False

def load_user_database():
    """Load user database from JSON file"""
    if os.path.exists(USER_DB_FILE):
        try:
            with open(USER_DB_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_user_database(db):
    """Save user database to JSON file"""
    with open(USER_DB_FILE, 'w') as f:
        json.dump(db, f, indent=2)

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def is_valid_email(email):
    """Enhanced email validation with domain checking"""
    # Basic format validation
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        return False, "Invalid email format!"
    
    # Check for common email providers and valid domains
    valid_domains = [
        'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'icloud.com',
        'aol.com', 'protonmail.com', 'mail.com', 'zoho.com', 'yandex.com',
        'live.com', 'msn.com', 'rediffmail.com', 'fastmail.com'
    ]
    
    domain = email.split('@')[1].lower()
    
    # Check if it's a valid domain or educational domain
    if domain in valid_domains or domain.endswith('.edu') or domain.endswith('.org'):
        return True, "Valid email"
    
    # Additional validation for business domains
    if len(domain.split('.')) >= 2 and not any(suspicious in domain for suspicious in ['temp', 'fake', 'test', 'disposable']):
        return True, "Valid email"
    
    return False, f"Email domain '{domain}' may not be valid or is suspicious"

def verify_email_domain(email):
    """Advanced email domain verification"""
    domain = email.split('@')[1].lower()
    
    # Check for fake/disposable email services
    fake_domains = [
        '10minutemail', 'tempmail', 'guerrillamail', 'mailinator',
        'yopmail', 'dispostable', 'throwaway', 'temp-mail'
    ]
    
    if any(fake_domain in domain for fake_domain in fake_domains):
        return False, "Disposable email addresses are not allowed"
    
    return True, "Domain verified"

def register_user(full_name, email, password):
    """Register a new user with enhanced validation"""
    users_db = load_user_database()
    
    # Check if user already exists
    if email in users_db:
        return False, "User already exists with this email!"
    
    # Enhanced email validation
    email_valid, email_msg = is_valid_email(email)
    if not email_valid:
        return False, email_msg
    
    # Domain verification
    domain_valid, domain_msg = verify_email_domain(email)
    if not domain_valid:
        return False, domain_msg
    
    # Password strength check
    if len(password) < 6:
        return False, "Password must be at least 6 characters long!"
    
    # Create user record
    users_db[email] = {
        'full_name': full_name,
        'password_hash': hash_password(password),
        'registration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'login_count': 0,
        'last_login': None
    }
    
    save_user_database(users_db)
    return True, "Registration successful!"

def authenticate_user(email, password):
    """Authenticate user login"""
    users_db = load_user_database()
    
    if email not in users_db:
        return False, "User not found!"
    
    stored_hash = users_db[email]['password_hash']
    if hash_password(password) != stored_hash:
        return False, "Incorrect password!"
    
    # Update login information
    users_db[email]['login_count'] = users_db[email].get('login_count', 0) + 1
    users_db[email]['last_login'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    save_user_database(users_db)
    
    return True, "Login successful!"

def is_user_logged_in():
    """Check if user is currently logged in"""
    return st.session_state.get('logged_in', False) and st.session_state.get('user_email', '') != ''

def create_login_form():
    """Simple, clean, and minimalistic login form"""
    
    # Clean and minimal styling without white box
    st.markdown("""
    <style>
        /* Remove default Streamlit padding */
        .main > div {
            padding-top: 2rem;
            max-width: 500px;
            margin: 0 auto;
        }
        
        /* Simple background */
        .stApp {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        }
        
        /* Title styling */
        .login-title {
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .login-title h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, #58a6ff, #3fb950);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        .login-title p {
            color: #6c757d;
            font-size: 1rem;
            margin: 0;
        }
        
        /* Input styling */
        .stTextInput input {
            border-radius: 8px !important;
            border: 1px solid #dee2e6 !important;
            padding: 12px !important;
            background: rgba(255, 255, 255, 0.9) !important;
        }
        
        .stTextInput input:focus {
            border-color: #58a6ff !important;
            box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.1) !important;
            background: white !important;
        }
        
        /* Button styling */
        .stButton button {
            background: #58a6ff !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 12px 24px !important;
            font-weight: 500 !important;
            width: 100% !important;
        }
        
        .stButton button:hover {
            background: #4c9eff !important;
        }
        
        /* Form section styling */
        .login-section {
            padding: 1rem;
            margin: 1rem 0;
        }
        
        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
    
    # Simple title
    st.markdown("""
    <div class="login-title">
        <h1>WhatsApp Chat Analysis</h1>
        <p>🔐 Secure Access Portal</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'auth_tab' not in st.session_state:
        st.session_state.auth_tab = 'login'
    
    # Simple tab selection
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔑 Sign In", key="tab_login", use_container_width=True):
            st.session_state.auth_tab = 'login'
    with col2:
        if st.button("📝 Register", key="tab_register", use_container_width=True, type="secondary"):
            st.session_state.auth_tab = 'register'
    
    st.markdown("---")
    
    # Login Form
    if st.session_state.auth_tab == 'login':
        st.markdown("#### Welcome Back")
        
        with st.form("login", clear_on_submit=False):
            email = st.text_input("Email", placeholder="your@email.com")
            password = st.text_input("Password", type="password", placeholder="Your password")
            
            if st.form_submit_button("Sign In", type="primary"):
                if email and password:
                    success, message = authenticate_user(email, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user_email = email
                        st.success("Welcome back!")
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("Please fill all fields")
    
    # Register Form  
    else:
        st.markdown("#### Create Account")
        
        with st.form("register", clear_on_submit=False):
            name = st.text_input("Full Name", placeholder="John Doe")
            email = st.text_input("Email", placeholder="john@email.com") 
            password = st.text_input("Password", type="password", placeholder="Create password")
            confirm = st.text_input("Confirm Password", type="password", placeholder="Confirm password")
            
            agree = st.checkbox("I agree to Terms & Privacy Policy")
            
            if st.form_submit_button("Create Account", type="primary"):
                if name and email and password and confirm and agree:
                    if password != confirm:
                        st.error("Passwords don't match")
                    else:
                        success, message = register_user(name, email, password)
                        if success:
                            st.success("Account created! Please sign in.")
                            st.session_state.auth_tab = 'login'
                            st.rerun()
                        else:
                            st.error(message)
                else:
                    st.warning("Please fill all fields and accept terms")
    


def create_user_dashboard():
    """Create user dashboard after login"""
    user_email = st.session_state.get('user_email', 'Unknown')
    
    # Welcome message with professional styling
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    ">
        <h3 style="color: white; margin: 0;">👋 Welcome back!</h3>
        <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0 0;">Logged in as: <strong>{user_email}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # User controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("👤 **View Profile**", use_container_width=True):
            st.session_state.show_profile = True
    
    with col2:
        if st.button("🔄 **Refresh Session**", use_container_width=True):
            st.cache_data.clear()
            st.success("✅ Session refreshed!")
    
    with col3:
        if st.button("🚪 **Sign Out**", use_container_width=True, type="secondary"):
            st.session_state.logged_in = False
            st.session_state.user_email = ''
            st.session_state.show_profile = False
            st.success("✅ Successfully signed out!")
            time.sleep(1)
            st.rerun()

def show_user_profile():
    """Show user profile information"""
    st.markdown("### 👤 **User Profile**")
    
    user_email = st.session_state.get('user_email', '')
    users_db = load_user_database()
    
    if user_email in users_db:
        user_info = users_db[user_email]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Full Name:** {user_info.get('full_name', 'N/A')}")
            st.info(f"**Email:** {user_email}")
            st.info(f"**Registration Date:** {user_info.get('registration_date', 'N/A')}")
        
        with col2:
            st.info(f"**Total Logins:** {user_info.get('login_count', 0)}")
            st.info(f"**Last Login:** {user_info.get('last_login', 'Never')}")
            st.info(f"**Account Status:** Active ✅")
    
    if st.button("⬅️ **Back to Dashboard**"):
        st.session_state.show_profile = False
        st.rerun()

def get_all_users():
    """Get all registered users (admin only)"""
    return load_user_database()
