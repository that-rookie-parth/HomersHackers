import os

import streamlit as st
from utils_2 import analyze_clause_bias, analyze_rfp_document, suggest_balanced_clause

st.set_page_config(page_title="ConsultAdd RFP Analyzer", layout="wide")

st.title("📄 ConsultAdd RFP Risk Analyzer")
st.markdown("*Analyze RFP documents for potential risks and biased clauses*")

uploaded_file = st.file_uploader("Upload RFP Document (PDF)")

if uploaded_file:
    temp_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Analyzing RFP..."):
        try:
            analysis = analyze_rfp_document(temp_path)
            if analysis:
                # Create two columns for layout
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.header("📊 RFP Analysis")

                    # Requirements Summary in a single row
                    stats_cols = st.columns(3)
                    with stats_cols[0]:
                        st.metric("Total Requirements", analysis['statistics']['total_requirements'])
                    with stats_cols[1]:
                        st.metric("Mandatory Requirements", analysis['statistics']['mandatory_requirements'])
                    with stats_cols[2]:
                        st.metric("Requirement Types", len(analysis['statistics']['by_type']))

                    # Risk Analysis Section
                    st.header("⚠️ Risk Analysis")
                    for req in analysis['requirements']:
                        biased_findings = analyze_clause_bias(req['text'])
                        if biased_findings:
                            with st.expander(f"🚨 Risk Found in Requirement {req['id']}"):
                                st.markdown("**Original Text:**")
                                st.write(req['text'])

                                for finding in biased_findings:
                                    st.markdown(f"**Risk Level:** 🔴 {finding['risk_level']}")
                                    st.markdown("**Issue:**")
                                    st.write(finding['type'].replace('_', ' ').title())

                                    st.markdown("**Suggested Change:**")
                                    balanced = suggest_balanced_clause(finding)
                                    st.write(balanced)

                                    # Add copy button for suggested version
                                    if st.button("📋 Copy Suggestion", key=f"copy_{hash(req['text'])}"):
                                        st.toast("✅ Copied to clipboard!")

                with col2:
                    # Summary Statistics
                    st.header("📈 Risk Summary")

                    # Count risks by severity
                    risk_counts = {
                        'High': 0,
                        'Medium': 0,
                        'Low': 0
                    }

                    for req in analysis['requirements']:
                        findings = analyze_clause_bias(req['text'])
                        for finding in findings:
                            risk_counts[finding['risk_level']] += 1

                    # Display risk metrics
                    st.metric("High Risk Items", risk_counts['High'], delta=None, delta_color="inverse")
                    st.metric("Medium Risk Items", risk_counts['Medium'], delta=None, delta_color="inverse")
                    st.metric("Low Risk Items", risk_counts['Low'], delta=None, delta_color="off")

                    # Requirements by Type
                    st.subheader("📊 Requirements by Type")
                    for req_type, count in analysis['statistics']['by_type'].items():
                        st.write(f"- {req_type.title()}: {count}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


# Add footer
st.markdown("---")
st.markdown("*Powered by ConsultAdd Risk Analysis Engine*")
