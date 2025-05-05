import streamlit as st
import pandas as pd
import psycopg2
from psycopg2 import sql
import hashlib
import plotly.express as px
import time
import os
from sklearn.ensemble import GradientBoostingClassifier

# Page configuration
st.set_page_config(
    page_title="AI Vehicle Insurability",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
/* [Previous CSS content remains exactly the same] */
</style>
""", unsafe_allow_html=True)

# Database functions (PostgreSQL)
def get_db_connection():
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

def init_db():
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        username TEXT PRIMARY KEY,
                        password TEXT,
                        email TEXT,
                        full_name TEXT
                    )
                """)
            conn.commit()
        except Exception as e:
            st.error(f"Database initialization failed: {e}")
        finally:
            conn.close()

def add_user(username, password, email, full_name):
    conn = get_db_connection()
    if not conn:
        return False
        
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("INSERT INTO users VALUES (%s, %s, %s, %s)"),
                (username, hashlib.sha256(password.encode()).hexdigest(), email, full_name)
            )
        conn.commit()
        return True
    except psycopg2.IntegrityError:
        return False
    except Exception as e:
        st.error(f"Error adding user: {e}")
        return False
    finally:
        conn.close()

def verify_user(username, password):
    conn = get_db_connection()
    if not conn:
        return False
        
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT 1 FROM users WHERE username = %s AND password = %s"),
                (username, hashlib.sha256(password.encode()).hexdigest())
            )
            return cur.fetchone() is not None
    except Exception as e:
        st.error(f"Verification error: {e}")
        return False
    finally:
        conn.close()

def user_exists(username):
    conn = get_db_connection()
    if not conn:
        return False
        
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT 1 FROM users WHERE username = %s"),
                (username,)
            )
            return cur.fetchone() is not None
    except Exception as e:
        st.error(f"User check error: {e}")
        return False
    finally:
        conn.close()

# Fallback session-based auth if DB fails
if 'users' not in st.session_state:
    st.session_state.users = {}

# Load sample data
@st.cache_data
def load_data():
    try:
        # Use raw GitHub URL or absolute path for deployment
        return pd.read_csv("zimbabwe_vehicle_insurance_dataset.csv")
    except FileNotFoundError:
        st.error("Dataset file not found")
        return pd.DataFrame()


# Train model
@st.cache_resource
def train_model(data):
    # Create a categorical risk class from risk_score
    data = data.copy()
    data['risk_class'] = pd.cut(
        data['risk_score'],
        bins=[-float('inf'), 10, 15, float('inf')],
        labels=[0, 1, 2]  # 0 = Low, 1 = Moderate, 2 = High
    ).astype(int)

    X = data[['year', 'mileage', 'engine_size', 'driver_age', 'license_years',
              'credit_score', 'accidents', 'claims', 'traffic_violations']]
    y = data['risk_class']

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X, y)
    return model

# Modified auth_page() with fallback
def auth_page():
    st.markdown("""
    <div style="text-align: center;">
        <h1 style="color: #4b6cb7; font-size: 2.5rem; margin-bottom: 0.5rem;">🚘 AI Vehicle Insurability Assessment</h1>
        <p style="color: #666; font-size: 1.1rem;">Smart risk assessment for modern insurers</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            <div class="custom-card">
                <h2 style="color: #4b6cb7; margin-bottom: 1.5rem;">Login</h2>
            """, unsafe_allow_html=True)
            
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            
            if st.button("Login", key="login_btn"):
                # Try database first, then fallback to session
                db_success = verify_user(username, password)
                session_auth = (username in st.session_state.users and 
                               st.session_state.users[username]['password'] == hashlib.sha256(password.encode()).hexdigest())
                
                if db_success or session_auth:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid username or password")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="custom-card">
                <h2 style="color: #4b6cb7; margin-bottom: 1.5rem;">Sign Up</h2>
            """, unsafe_allow_html=True)
            
            new_username = st.text_input("Choose a username", key="signup_user")
            new_password = st.text_input("Choose a password", type="password", key="signup_pass")
            email = st.text_input("Email address", key="signup_email")
            full_name = st.text_input("Full name", key="signup_name")
            
            if st.button("Create Account", key="signup_btn"):
                if user_exists(new_username) or new_username in st.session_state.users:
                    st.error("Username already exists")
                else:
                    db_success = add_user(new_username, new_password, email, full_name)
                    if db_success:
                        st.success("Account created in database! Please login.")
                    else:
                        # Fallback to session storage
                        st.session_state.users[new_username] = {
                            'password': hashlib.sha256(new_password.encode()).hexdigest(),
                            'email': email,
                            'full_name': full_name
                        }
                        st.success("Account created in session! Please login.")
            
            st.markdown("</div>", unsafe_allow_html=True)

# Main app
def main_app():
    # Sidebar
    with st.sidebar:
        st.markdown(f"""
        <div style="margin-bottom: 2rem;">
            <h2 style="color: white;">Welcome, {st.session_state.username}</h2>
            <p style="color: #d1d1d1;">Last login: Today</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Sign Out", key="signout_btn"):
            st.session_state.logged_in = False
            st.rerun()
        
        st.markdown("""
        <div style="margin-top: 2rem;">
            <h3 style="color: white;">Navigation</h3>
        </div>
        """, unsafe_allow_html=True)
        
        selected_tab = st.radio(
            "Go to",
            ["Dashboard", "Risk Assessment", "Data Explorer"],
            label_visibility="collapsed"
        )
        
        st.markdown("""
        <div style="margin-top: 2rem;">
            <h3 style="color: white;">About</h3>
            <p style="color: #d1d1d1;">
                AI Vehicle Insurability Assessment uses machine learning to evaluate vehicle risk
                and calculate appropriate premium rates.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    if selected_tab == "Dashboard":
        dashboard_tab()
    elif selected_tab == "Risk Assessment":
        assessment_tab()
    elif selected_tab == "Data Explorer":
        explorer_tab()

# Dashboard tab
def dashboard_tab():
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="color: #4b6cb7;">Insurability Dashboard</h1>
        <p style="color: #666;">Overview of risk assessment metrics and trends</p>
    </div>
    """, unsafe_allow_html=True)
    
    data = load_data()
    
    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Total Data Entries</h3>
            <h1>{:,}</h1>
        </div>
        """.format(len(data)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Avg Risk Score</h3>
            <h1>{:.1f}</h1>
            
        </div>
        """.format(data['risk_score'].mean()), unsafe_allow_html=True)
    
    with col3:
        approval_rate = 65
        st.markdown("""
        <div class="metric-card">
            <h3>Approval Rate</h3>
            <h1>{:.1f}%</h1>
        </div>
        """.format(approval_rate), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>Avg Premium</h3>
            <h1>${:,.1f}</h1>
            
        </div>
        """.format(data['annual_premium'].mean()), unsafe_allow_html=True)
    
    # Charts
    st.markdown("""
    <div class="custom-card">
        <h2 style="color: #4b6cb7;">Risk Score Distribution</h2>
    """, unsafe_allow_html=True)
    
    fig = px.histogram(data, x='risk_score', nbins=20, color_discrete_sequence=['#4b6cb7'])
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="custom-card">
        <h2 style="color: #4b6cb7;">Top Vehicle Makes by Risk</h2>
    """, unsafe_allow_html=True)
    
    make_risk = data.groupby('make')['risk_score'].mean().sort_values(ascending=False).head(10)
    fig = px.bar(make_risk, color=make_risk.values, color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Risk Assessment tab
def assessment_tab():
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="color: #4b6cb7;">Vehicle Risk Assessment</h1>
        <p style="color: #666;">Get insurability risk class and dynamic premium pricing</p>
    </div>
    """, unsafe_allow_html=True)

    data = load_data()
    model = train_model(data)

    with st.form("assessment_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="custom-card"><h2 style="color: #4b6cb7;">Vehicle Details</h2>
            """, unsafe_allow_html=True)
            make = st.selectbox("Make", data['make'].unique())
            model_name = st.selectbox("Model", data[data['make'] == make]['model'].unique())
            year = st.slider("Year", 1980, 2025, 2020)
            mileage = st.slider("Mileage", 0, 300000, 50000, step=1000)
            engine_size = st.slider("Engine Size (L)", 1.0, 5.0, 2.0, step=0.1)
            fuel_type = st.selectbox("Fuel Type", data['fuel_type'].unique())
            transmission = st.selectbox("Transmission", data['transmission'].unique())
            safety_features = st.slider("Safety Features Count", 0, 10, 5)
            vehicle_value = st.number_input("Vehicle Value ($)", min_value=1000, value=15000)

            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="custom-card"><h2 style="color: #4b6cb7;">Driver Details</h2>
            """, unsafe_allow_html=True)
            driver_age = st.slider("Driver Age", 18, 100, 30)
            license_years = st.slider("Years with License", 0, 50, 5)
            credit_score = st.slider("Credit Score", 300, 850, 650)
            province = st.selectbox("Province", data['province'].unique())
            urban = st.radio("Urban/Rural", ["Urban", "Rural"])
            urban_bool = urban == "Urban"
            accidents = st.slider("Past Accidents", 0, 10, 0)
            claims = st.slider("Past Claims", 0, 10, 0)
            traffic_violations = st.slider("Traffic Violations", 0, 10, 0)

            st.markdown("</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("Assess Risk", use_container_width=True)

    if submitted:
        with st.spinner("Analyzing vehicle risk..."):
            time.sleep(1)

            input_data = pd.DataFrame({
                'year': [year],
                'mileage': [mileage],
                'engine_size': [engine_size],
                'driver_age': [driver_age],
                'license_years': [license_years],
                'credit_score': [credit_score],
                'accidents': [accidents],
                'claims': [claims],
                'traffic_violations': [traffic_violations]
            })

            risk_class = model.predict(input_data)[0]
            risk_proba = model.predict_proba(input_data)[0]

            risk_tiers = {0: 'Low Risk', 1: 'Moderate Risk', 2: 'High Risk'}
            tier_label = risk_tiers.get(risk_class, 'Unknown')

            # Premium base and modifiers
            base_premium = vehicle_value * 0.05
            tier_multiplier = {0: 0.85, 1: 1.0, 2: 1.35}
            risk_multiplier = tier_multiplier.get(risk_class, 1.0)
            safety_discount = 1.0 - (safety_features * 0.015)
            location_factor = 1.1 if urban_bool else 1.0
            fuel_factor = 1.05 if fuel_type == 'Diesel' else 1.0
            transmission_discount = 0.95 if transmission == 'Automatic' else 1.0

            # Optional: Province-based risk factor
            higher_risk_provinces = ['Harare', 'Bulawayo']  # Example assumption
            province_factor = 1.05 if province in higher_risk_provinces else 1.0

            # Final premium calculation
            annual_premium = (
                base_premium *
                risk_multiplier *
                safety_discount *
                location_factor *
                fuel_factor *
                transmission_discount *
                province_factor
            )

            # Display results
            st.markdown("""
            <div class="custom-card"><h2 style="color: #4b6cb7;">Assessment Results</h2>
            """, unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Risk Tier</h3>
                    <h1>{tier_label}</h1>
                    <p>Confidence: {max(risk_proba)*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Approval Status</h3>
                    <h1>{"Approved" if risk_class < 2 else "Review Needed"}</h1>
                    <p>{make} {model_name}</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Annual Premium</h3>
                    <h1>${annual_premium:,.2f}</h1>
                    <p>Base: ${base_premium:,.2f}</p>
                </div>
                """, unsafe_allow_html=True)

            # Recommendation
            st.markdown("<h3 style='color:#4b6cb7;'>Recommendation</h3>", unsafe_allow_html=True)
            if risk_class == 0:
                st.success("Excellent profile. Eligible for premium coverage with additional benefits.")
            elif risk_class == 1:
                st.warning("Moderate risk. Eligible for standard coverage. Consider enhancing safety features.")
            else:
                st.error("High risk profile. Manual underwriting required. Additional documentation may be needed.")

            st.markdown("</div>", unsafe_allow_html=True)


# Data Explorer tab
def explorer_tab():
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="color: #4b6cb7;">Data Explorer</h1>
        <p style="color: #666;">Explore and analyze insurance data</p>
    </div>
    """, unsafe_allow_html=True)
    
    data = load_data()
    
    st.markdown("""
    <div class="custom-card">
        <h2 style="color: #4b6cb7;">Interactive Data Table</h2>
    """, unsafe_allow_html=True)
    
    st.dataframe(data, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    

# Initialize database
init_db()

# Check authentication
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    main_app()
else:
    auth_page()