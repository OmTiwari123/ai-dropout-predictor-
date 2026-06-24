import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import io

# ============== PAGE CONFIGURATION ==============
st.set_page_config(
    page_title="AI Dropout Predictor Pro",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== CUSTOM CSS (Makes it look premium) ==============
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 0.5rem 0;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    /* Risk Badge Styling */
    .risk-high {
        background-color: #dc3545;
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        font-size: 1.2rem;
    }
    .risk-moderate {
        background-color: #ffc107;
        color: #333;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        font-size: 1.2rem;
    }
    .risk-low {
        background-color: #28a745;
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        font-size: 1.2rem;
    }
    /* Metric Cards */
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 12px;
        border-left: 5px solid #2a5298;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e3c72;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #999;
        font-size: 0.85rem;
        border-top: 1px solid #eee;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============== CONFIGURATION ==============
# ⚠️ UPDATE THIS URL with your actual Render API URL!
API_URL = "https://ai-dropout-predictor-j3iv.onrender.com/predict"

# ============== HEADER ==============
st.markdown('<div class="main-title">🎓 AI Based Dropout Prediction & Counselling System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Intelligent Student Retention Platform powered by XGBoost & Explainable AI</div>', unsafe_allow_html=True)
st.markdown("---")

# ============== SIDEBAR ==============
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/student-center.png", width=80)
    st.markdown("### 📂 Data Management")
    
    uploaded_file = st.file_uploader("Upload Student Data (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df
        st.success(f"✅ Loaded {len(df)} students")
        st.caption(f"📊 Columns: {', '.join(df.columns.tolist())}")
    
    st.markdown("---")
    st.markdown("### 🛠️ Quick Actions")
    if st.button("🔄 Reset All Data"):
        st.session_state.clear()
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📋 About")
    st.caption("Version 2.0 | Built with Streamlit + XGBoost")
    st.caption("🔗 [GitHub Repo](https://github.com/OmTiwari123/ai-dropout-predictor-)")

# ============== MAIN CONTENT ==============
if 'df' in st.session_state:
    df = st.session_state['df']
    
    # ---- SECTION 1: SUMMARY METRICS ----
    st.markdown("### 📊 Dashboard Overview")
    
    # Calculate derived metrics
    total_students = len(df)
    at_risk_count = len(df[df['Target'] == 1]) if 'Target' in df.columns else 0
    retention_rate = round((1 - at_risk_count / total_students) * 100, 1) if total_students > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_students}</div>
            <div class="metric-label">📚 Total Students</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #dc3545;">
            <div class="metric-value" style="color: #dc3545;">{at_risk_count}</div>
            <div class="metric-label">⚠️ At-Risk Students</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #28a745;">
            <div class="metric-value" style="color: #28a745;">{retention_rate}%</div>
            <div class="metric-label">✅ Retention Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #ffc107;">
            <div class="metric-value">{len(df.columns)}</div>
            <div class="metric-label">📋 Data Features</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ---- SECTION 2: PREDICTION ENGINE ----
    st.markdown("### 🎯 Student Risk Assessment")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["🔍 Single Student", "📊 Bulk Prediction", "📈 Analytics"])
    
    # ======= TAB 1: SINGLE STUDENT PREDICTION =======
    with tab1:
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            student_ids = df['StudentID'].astype(str).tolist()
            selected_id = st.selectbox("👤 Select Student ID", student_ids, key="single_select")
            
            # Show student details
            student_row = df[df['StudentID'].astype(str) == selected_id].iloc[0]
            
            st.markdown("**📋 Student Profile**")
            profile_cols = st.columns(2)
            with profile_cols[0]:
                st.write(f"**ID:** {student_row['StudentID']}")
                st.write(f"**GPA:** {student_row.get('GPA', 'N/A')}")
                st.write(f"**Attendance:** {student_row.get('Attendance', 'N/A')}%")
            with profile_cols[1]:
                st.write(f"**Semester:** {student_row.get('Semester', 'N/A')}")
                st.write(f"**Fee Status:** {'⚠️ Pending' if student_row.get('Fee_Status') == 1 else '✅ Paid'}")
                st.write(f"**Scholarship:** {'✅ Yes' if student_row.get('Scholarship') == 1 else '❌ No'}")
        
        with col_right:
            if st.button("🚀 Predict Risk", type="primary", use_container_width=True):
                # Prepare features for API
                features = {k: v for k, v in student_row.items() if k not in ['StudentID', 'Target']}
                
                try:
                    with st.spinner("🧠 AI is analyzing..."):
                        response = requests.post(API_URL, json=features, timeout=10)
                        
                        if response.status_code == 200:
                            result = response.json()
                            prob = result['probability']
                            risk = result['risk_level']
                            
                            # Display risk meter
                            st.markdown("**📉 Dropout Probability**")
                            
                            # Progress bar with color
                            if risk == "High":
                                bar_color = "#dc3545"
                                badge = '<span class="risk-high">🔴 HIGH RISK</span>'
                            elif risk == "Moderate":
                                bar_color = "#ffc107"
                                badge = '<span class="risk-moderate">🟠 MODERATE RISK</span>'
                            else:
                                bar_color = "#28a745"
                                badge = '<span class="risk-low">🟢 LOW RISK</span>'
                            
                            # Custom progress bar
                            st.markdown(f"""
                            <div style="background: #e9ecef; border-radius: 10px; height: 30px; position: relative; margin: 10px 0;">
                                <div style="background: {bar_color}; width: {prob * 100}%; height: 100%; border-radius: 10px; transition: width 0.5s ease;"></div>
                                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: {'#333' if prob < 0.5 else 'white'}; font-weight: 700; font-size: 1rem;">
                                    {prob * 100:.1f}%
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"**Risk Level:** {badge}", unsafe_allow_html=True)
                            
                            # Counselling Recommendation
                            st.markdown("---")
                            st.markdown("### 💡 Counselling Recommendation")
                            if risk == "High":
                                st.error("""
                                🚨 **IMMEDIATE INTERVENTION REQUIRED**
                                
                                - 📞 Schedule an emergency meeting with the student
                                - 📚 Assign academic tutoring (GPA is critically low)
                                - 💰 Connect with financial aid office if fee pending
                                - 📊 Create a weekly progress tracking plan
                                """)
                            elif risk == "Moderate":
                                st.warning("""
                                ⚠️ **MONITOR CLOSELY**
                                
                                - 📧 Send a check-in email to the student
                                - 📈 Track attendance and GPA for next 2 weeks
                                - 🗓️ Schedule a mid-semester progress review
                                """)
                            else:
                                st.success("""
                                ✅ **STUDENT IS ON TRACK**
                                
                                - 🎯 Continue regular monitoring
                                - 📊 Encourage student to maintain current performance
                                - 🏆 Consider for academic recognition programs
                                """)
                            
                            # Store in session for history
                            if 'history' not in st.session_state:
                                st.session_state.history = []
                            st.session_state.history.append({
                                'student_id': selected_id,
                                'probability': prob,
                                'risk_level': risk,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
                            })
                            
                        else:
                            st.error(f"❌ API Error: {response.text}")
                            
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to the AI API. Please check if the server is running.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    # ======= TAB 2: BULK PREDICTION =======
    with tab2:
        st.markdown("**📊 Process all students at once**")
        st.caption("This will predict risk for every student in the uploaded dataset.")
        
        if st.button("⚡ Run Bulk Prediction", type="primary"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_rows = len(df)
            
            for idx, (_, row) in enumerate(df.iterrows()):
                status_text.text(f"Processing student {idx+1}/{total_rows}...")
                progress_bar.progress((idx+1) / total_rows)
                
                features = {k: v for k, v in row.items() if k not in ['StudentID', 'Target']}
                
                try:
                    response = requests.post(API_URL, json=features, timeout=5)
                    if response.status_code == 200:
                        result = response.json()
                        results.append({
                            'StudentID': row['StudentID'],
                            'Probability': round(result['probability'] * 100, 1),
                            'Risk_Level': result['risk_level']
                        })
                except:
                    results.append({
                        'StudentID': row['StudentID'],
                        'Probability': 'Error',
                        'Risk_Level': 'Error'
                    })
            
            progress_bar.empty()
            status_text.text("✅ Bulk Prediction Complete!")
            
            # Show results
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)
            
            # Summary
            st.markdown("---")
            st.markdown("### 📊 Risk Distribution Summary")
            
            risk_counts = results_df['Risk_Level'].value_counts().reset_index()
            risk_counts.columns = ['Risk Level', 'Count']
            
            # Pie Chart
            fig = px.pie(risk_counts, values='Count', names='Risk Level',
                         color='Risk Level',
                         color_discrete_map={'High': '#dc3545', 'Moderate': '#ffc107', 'Low': '#28a745'},
                         title='Risk Distribution Among Students')
            st.plotly_chart(fig, use_container_width=True)
            
            # Export button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"dropout_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    # ======= TAB 3: ANALYTICS =======
    with tab3:
        st.markdown("### 📈 Data Analytics & Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # GPA Distribution
            if 'GPA' in df.columns:
                fig = px.histogram(df, x='GPA', nbins=20, title='GPA Distribution',
                                   color_discrete_sequence=['#2a5298'])
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Attendance vs GPA scatter
            if 'Attendance' in df.columns and 'GPA' in df.columns:
                fig = px.scatter(df, x='Attendance', y='GPA',
                                 color='Target' if 'Target' in df.columns else None,
                                 title='Attendance vs GPA',
                                 color_continuous_scale='RdYlGn',
                                 labels={'Target': 'Dropout Risk'})
                st.plotly_chart(fig, use_container_width=True)
        
        # Risk Factors Breakdown
        st.markdown("### 🔑 Key Risk Factors")
        risk_factors = ['GPA', 'Attendance', 'Fee_Status', 'Semester']
        
        if all(col in df.columns for col in risk_factors):
            avg_by_risk = df.groupby('Target')[risk_factors].mean().reset_index()
            avg_by_risk['Target'] = avg_by_risk['Target'].map({0: 'Graduated', 1: 'Dropped Out'})
            
            fig = px.bar(avg_by_risk.melt(id_vars='Target', var_name='Factor', value_name='Average'),
                         x='Factor', y='Average', color='Target', barmode='group',
                         title='Average Feature Values by Student Outcome')
            st.plotly_chart(fig, use_container_width=True)

# ============== PREDICTION HISTORY ==============
if 'history' in st.session_state and len(st.session_state.history) > 0:
    with st.expander("📜 Prediction History", expanded=False):
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)

# ============== FOOTER ==============
st.markdown('<div class="footer">Built with ❤️ using XGBoost, Flask, and Streamlit | AI Dropout Prediction System v2.0</div>', unsafe_allow_html=True)
