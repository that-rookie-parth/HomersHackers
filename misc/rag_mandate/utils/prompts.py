MANDATE_PROMPT = """You are a highly precise assistant helping evaluate RFP documents. Your job is to locate and report on specific information ONLY if mentioned explicitly in the context.

            When analyzing, follow these rules:
            - Look **only** for exact phrases or close variations. Do NOT infer or generalize.
            - Do not confuse **Temporary Staffing** with **Company Length of Existence**.

            Criteria to check:
            1. Is there any mention of **Years of Experience in Temporary staffing**? If yes, is the experience required less than 7 years?
            2. Is there any mention of a **W-9 Form** or **W-9**?
            3. Is there any mention of **Insurance Certificates** or **Insurance** keyword?
            4. Is there any mention of **Company Length of Existence** (how long the company has existed)?
            5. Is there any mention of **Licenses, Certifications, or Registrations**?

            Your response should answer each question clearly in Yes or No, in markdown format.
            """

COMPLIANCE_PROMPT = """
can you go through the chunks find out if the company is legally eligible to bid (e.g., state registration, certifications, past performance requirements).
if yes then explain why, if no then also explain why.
also if the company is not eligible, don't outright reject it, give reason and try if you can find any workaround that particular clause.
and in case of not eligible, quote the exact text from the RFP.
also try to is if the company does not have any criteria met, then is there any thing particular mention in the 
RFP that might help us to be eligible.
"""
