import streamlit as st
from agents.risky_clause import extract_text_from_pdf
from agents.risky_clause import extract_format_requirements, extract_attachments_and_forms
from agents.risky_clause import analyze_risk_levels, rewrite_high_risk_clauses
from agents.risky_clause import compute_vendor_friendly_score
from agents.risky_clause import generate_pdf_report

st.set_page_config(page_title="🧠 RFP Risk Analyzer")

st.title("📑 AI-Powered RFP Risk Analyzer")

uploaded_file = st.file_uploader("Upload your RFP PDF", type="pdf")

if uploaded_file:
    with st.spinner("🔍 Extracting text..."):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        rfp_text = extract_text_from_pdf("temp.pdf")

    st.subheader("✅ Extracted Information")
    if st.button("Run Full Analysis"):
        formats = extract_format_requirements.invoke(rfp_text)
        forms = extract_attachments_and_forms.invoke(rfp_text)
        risks = analyze_risk_levels.invoke(rfp_text)
        rewrites = rewrite_high_risk_clauses.invoke(rfp_text)
        score = compute_vendor_friendly_score.invoke(rfp_text)

        st.write("📄 Format Requirements:", formats)
        st.write("📎 Attachments & Forms:", forms)

        st.subheader("🔥 Risk Heatmap")
        for para, level in risks:
            color = {"High": "red", "Medium": "orange", "Low": "green"}.get(level, "gray")
            st.markdown(f"<div style='background-color:{color};padding:10px;margin-bottom:10px;color:white'>{para}</div>", unsafe_allow_html=True)

        st.subheader("🧠 Vendor Score")
        st.metric("Score", score["score"])
        st.caption(score["reason"])

        st.subheader("✍️ Clause Suggestions")
        for item in rewrites:
            st.markdown(f"**🔴 Original:** {item['original']}")
            st.markdown(f"**✅ Suggested:** {item['suggested']}")
            st.markdown("---")

        st.subheader("📤 Export Report")
        report_data = {
            "Format Requirements": formats,
            "Forms": forms,
            "Vendor Score": score,
            "Suggestions": rewrites
        }
        generate_pdf_report("report.pdf", report_data)
        with open("report.pdf", "rb") as f:
            st.download_button("Download PDF Report", f, file_name="rfp_analysis_report.pdf")

