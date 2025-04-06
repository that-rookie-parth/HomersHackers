import os

import streamlit as st
from rfp_agent import RFPAnalysisAgent
from utils_2 import analyze_clause_bias, analyze_rfp_document, suggest_balanced_clause

# Page configuration
st.set_page_config(
    page_title="ConsultAdd RFP Analyzer",
    page_icon="📄",
    layout="wide"
)

# Initialize agent
agent = RFPAnalysisAgent()

# Title and description
st.title("📄 ConsultAdd RFP Risk Analyzer")
st.markdown("*AI-powered RFP analysis to identify and mitigate contractual risks*")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Analysis Settings")
    risk_threshold = st.slider(
        "Risk Sensitivity",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        help="Lower values will identify more potential risks"
    )
    
    show_suggestions = st.checkbox("Show Balanced Alternatives", value=True)
    show_entities = st.checkbox("Show Named Entities", value=True)

# Main interface
uploaded_file = st.file_uploader("Upload RFP Document (PDF)", type=['pdf'])

if uploaded_file:
    # Save uploaded file temporarily
    temp_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Analyzing RFP..."):
        try:
            # Analyze document
            analysis = analyze_rfp_document(temp_path)

            if analysis:
                # Create two columns for layout
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.header("📊 Document Analysis")

                    # Document statistics
                    stats_cols = st.columns(3)
                    with stats_cols[0]:
                        st.metric(
                            "Total Requirements",
                            analysis['statistics']['total_requirements']
                        )
                    with stats_cols[1]:
                        st.metric(
                            "Mandatory Items",
                            analysis['statistics']['mandatory_requirements']
                        )
                    with stats_cols[2]:
                        st.metric(
                            "Risk Items",
                            len([r for r in analysis['requirements'] 
                                if analyze_clause_bias(r['text'])])
                        )

                    # Risk Analysis Section
                    st.header("⚠️ Risk Analysis")
                    for req in analysis['requirements']:
                        biased_findings = analyze_clause_bias(req['text'])
                        if biased_findings:
                            with st.expander(f"🚨 Risk Found in Requirement {req['id']}"):
                                # Original text
                                st.markdown("**Original Text:**")
                                st.write(req['text'])

                                # Risk details
                                for finding in biased_findings:
                                    risk_color = {
                                        'High': '🔴',
                                        'Medium': '🟡',
                                        'Low': '🟢'
                                    }.get(finding['risk_level'], '⚪')

                                    st.markdown(f"**Risk Level:** {risk_color} {finding['risk_level']}")
                                    st.markdown("**Issue Type:**")
                                    st.write(finding['type'].replace('_', ' ').title())

                                    if show_suggestions:
                                        st.markdown("**Suggested Balanced Alternative:**")
                                        balanced = suggest_balanced_clause(finding)
                                        st.write(balanced)

                                        if st.button("📋 Copy Suggestion", 
                                                   key=f"copy_{hash(req['text'])}"):
                                            st.toast("✅ Copied to clipboard!")

                                    # Feedback mechanism
                                    feedback = st.radio(
                                        "Was this analysis helpful?",
                                        ["Yes", "No", "Partially"],
                                        key=f"feedback_{hash(req['text'])}"
                                    )

                                    if feedback != "Yes":
                                        user_feedback = st.text_area(
                                            "How can we improve this analysis?",
                                            key=f"feedback_text_{hash(req['text'])}"
                                        )

                                        if st.button("Submit Feedback", 
                                                   key=f"submit_{hash(req['text'])}"):
                                            agent._learn_from_feedback(
                                                req['text'],
                                                feedback,
                                                user_feedback
                                            )
                                            st.success("Thank you! Your feedback helps improve future analyses.")

                with col2:
                    st.header("📈 Risk Summary")

                    # Risk distribution
                    risk_levels = {'High': 0, 'Medium': 0, 'Low': 0}
                    for req in analysis['requirements']:
                        findings = analyze_clause_bias(req['text'])
                        for finding in findings:
                            risk_levels[finding['risk_level']] += 1

                    # Display metrics
                    st.metric("High Risk Items", risk_levels['High'], 
                            delta=None, delta_color="inverse")
                    st.metric("Medium Risk Items", risk_levels['Medium'], 
                            delta=None, delta_color="inverse")
                    st.metric("Low Risk Items", risk_levels['Low'], 
                            delta=None, delta_color="off")

                    # Requirements breakdown
                    st.subheader("Requirements by Type")
                    for req_type, count in analysis['statistics']['by_type'].items():
                        st.write(f"- {req_type.title()}: {count}")

                    # Download report button
                    if st.button("📥 Download Full Report"):
                        st.markdown("Generating report...")
                        # TODO: Implement report generation
                        st.success("Report downloaded!")

        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)

# Footer
st.markdown("---")
col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("*Powered by ConsultAdd AI*")
with col2:
    st.markdown("*Built with ❤️ by Homer's Hackers*")
