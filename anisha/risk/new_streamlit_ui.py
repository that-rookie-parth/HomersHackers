import streamlit as st
import os
from rfp_agent import RFPAnalysisAgent
from utils_2 import analyze_rfp_document, analyze_clause_bias, suggest_balanced_clause

# Set wide mode and custom title
st.set_page_config(page_title="ConsultAdd RFP Analyzer", page_icon="📄", layout="wide")

st.markdown("""
    <style>
        .reportview-container {
            background-color: #F7F9FC;
        }
        .sidebar .sidebar-content {
            background-color: #EFF2F5;
        }
        h1, h2, h3 {
            color: #0D3B66;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📄 ConsultAdd RFP Risk Analyzer")
st.markdown("##### *AI-powered tool to ensure compliance, analyze risks, and assist in RFP submissions.*")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    risk_threshold = st.slider("Risk Sensitivity", 0.0, 1.0, 0.7, help="Lower values = more risk items detected")
    show_suggestions = st.checkbox("💡 Show Balanced Alternatives", True)
    show_entities = st.checkbox("📍 Highlight Named Entities", True)
    st.markdown("---")
    uploaded_file = st.file_uploader("📤 Upload RFP Document (PDF)", type=["pdf"])

# Init agent
agent = RFPAnalysisAgent()

# If file uploaded
if uploaded_file:
    temp_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("🔍 Analyzing your RFP..."):
        try:
            analysis = analyze_rfp_document(temp_path)

            if analysis:
                # Main UI with tabs
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📋 Eligibility Check",
                    "✅ Mandatory Criteria",
                    "🧾 Submission Checklist",
                    "⚖️ Contract Risk Analysis"
                ])

                with tab1:
                    st.subheader("📋 Standard Compliance Check")
                    st.success("✅ Legally eligible to bid!")  # Placeholder
                    st.warning("⚠️ Missing Past Performance Details")  # Placeholder

                with tab2:
                    st.subheader("✅ Must-Have Criteria")
                    st.markdown("Here’s what you need to qualify:")
                    for req_type, count in analysis['statistics']['by_type'].items():
                        st.info(f"🔹 {req_type.title()}: {count}")

                with tab3:
                    st.subheader("🧾 Submission Requirements Checklist")
                    st.markdown("Auto-extracted checklist from the RFP:")
                    checklist_items = [
                        "Max 10 pages (Arial, size 11)",
                        "Include TOC and section headers",
                        "Attach Company Registration & Tax Forms"
                    ]
                    for item in checklist_items:
                        st.checkbox(item, value=False)

                with tab4:
                    st.subheader("⚖️ Risky Clauses & Suggestions")

                    risk_levels = {'High': 0, 'Medium': 0, 'Low': 0}
                    for req in analysis['requirements']:
                        findings = analyze_clause_bias(req['text'])
                        for finding in findings:
                            risk_levels[finding['risk_level']] += 1

                    st.markdown("### 🔍 Risk Summary")
                    st.metric("🔴 High Risk Items", risk_levels["High"])
                    st.metric("🟡 Medium Risk Items", risk_levels["Medium"])
                    st.metric("🟢 Low Risk Items", risk_levels["Low"])

                    st.markdown("---")

                    for req in analysis['requirements']:
                        biased_findings = analyze_clause_bias(req['text'])
                        if biased_findings:
                            with st.expander(f"🚨 Risk in Requirement {req['id']}"):
                                st.markdown("**🔹 Original Clause:**")
                                st.write(req['text'])

                                for finding in biased_findings:
                                    color = {
                                        'High': '🔴',
                                        'Medium': '🟡',
                                        'Low': '🟢'
                                    }.get(finding['risk_level'], '⚪')
                                    
                                    st.markdown(f"**Risk Level:** {color} {finding['risk_level']}")
                                    st.markdown(f"**Issue Type:** {finding['type'].replace('_', ' ').title()}")

                                    if show_suggestions:
                                        st.markdown("**💡 Suggested Balanced Alternative:**")
                                        st.code(suggest_balanced_clause(finding), language='markdown')

                                        st.button("📋 Copy", key=f"copy_{hash(req['text'])}")
                                    
                                    st.radio(
                                        "Was this helpful?",
                                        ["Yes", "No", "Partially"],
                                        key=f"feedback_{hash(req['text'])}"
                                    )

        except Exception as e:
            st.error(f"🚫 Error: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

# Footer
st.markdown("---")
footer1, footer2 = st.columns([1, 2])
with footer1:
    st.markdown("🚀 Powered by **ConsultAdd AI**")
with footer2:
    st.markdown("Crafted with ❤️ by **Homer’s Hackers**")
