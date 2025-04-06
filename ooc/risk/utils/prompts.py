MANDATE_PROMPT = """You are a highly precise assistant helping evaluate from the context. Your job is to locate and report on specific information ONLY if mentioned explicitly in the context.

            When analyzing, follow these rules:
            - Look **only** for exact phrases or close variations. Do NOT infer or generalize.
            - Do not confuse **Temporary Staffing** with **Company Length of Existence**.

            Criteria to check:
            1. Is there any mention of **Years of Experience in Temporary staffing**? If yes, is the experience required less than 7 years?
            2. Is there any mention of a **W-9 Form** or **W-9**?
            3. Is there any mention of **Insurance Certificates** or **Insurance** keyword?
            4. Is there any mention of **Company Length of Existence** (how long the company has existed)?
            5. Is there any mention of **Licenses, Certifications, or Registrations**?

            Your response should answer each question clearly in Yes or No, in markdown format. Give some explanation also
            """

COMPLIANCE_PROMPT = """
can you go through the chunks find out if the company is legally eligible to bid (e.g., state registration, certifications, past performance requirements).

        this is the data that i have data:
        ✅ Legal and Regulatory Info
        State of Incorporation: Delaware

        Business Structure: LLC

        State Registration Number: SRN-DE-0923847

        DUNS Number / CAGE Code / SAM.gov Registration: Required for federal contracting

        ✅ Experience and Capabilities
        Company Age: 9 years in business

        Staffing Experience: 7 years

        Services Offered: Staffing in administrative, IT, legal, and credentialing fields

        NAICS Codes: For categorizing services in federal procurement (e.g., Temporary Help Services)

        ✅ Compliance & Documentation
        Certificate of Insurance: Required for many contracts

        W-9 Form: Includes Tax ID (for payment purposes)

        Licenses: Texas Employment Agency license listed

        Bank Letter of Creditworthiness: Not available

        MBE/DBE/HUB Status: Not certified


        if yes then explain why, if no then also explain why 

    """
