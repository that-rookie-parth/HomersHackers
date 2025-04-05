import pandas as pd

company_data = [
    {"Field": "Company Legal Name", "Data": "FirstStaff Workforce Solutions, LLC"},
    {
        "Field": "Principal Business Address",
        "Data": "3105 Maple Avenue, Suite 1200, Dallas, TX 75201",
    },
    {"Field": "Phone Number", "Data": "(214) 832-4455"},
    {"Field": "Fax Number", "Data": "(214) 832-4460"},
    {"Field": "Email Address", "Data": "proposals@firststaffsolutions.com"},
    {
        "Field": "Authorized Representative",
        "Data": "Meredith Chan, Director of Contracts",
    },
    {"Field": "Authorized Representative Phone", "Data": "(212) 555-0199"},
    {"Field": "Signature", "Data": "Meredith Chan (signed manually)"},
    {"Field": "Company Length of Existence", "Data": "9 years"},
    {"Field": "Years of Experience in Temp Staffing", "Data": "7 years"},
    {"Field": "DUNS Number", "Data": "07-842-1490"},
    {"Field": "CAGE Code", "Data": "8J4T7"},
    {"Field": "SAM.gov Registration Date", "Data": "03/01/2022"},
    {
        "Field": "NAICS Codes",
        "Data": "561320 – Temporary Help Services; 541611 – Admin Management",
    },
    {"Field": "State of Incorporation", "Data": "Delaware"},
    {"Field": "Bank Letter of Creditworthiness", "Data": "Not Available"},
    {"Field": "State Registration Number", "Data": "SRN-DE-0923847"},
    {
        "Field": "Services Provided",
        "Data": "Administrative, IT, Legal & Credentialing Staffing",
    },
    {"Field": "Business Structure", "Data": "Limited Liability Company (LLC)"},
    {"Field": "W-9 Form", "Data": "Attached (TIN: 47-6392011)"},
    {
        "Field": "Certificate of Insurance",
        "Data": "Travelers Insurance, Policy #TX-884529-A; includes Workers' Comp, Liability, and Auto",
    },
    {"Field": "Licenses", "Data": "Texas Employment Agency License #TXEA-34892"},
    {
        "Field": "Historically Underutilized Business / DBE Status",
        "Data": "Not certified",
    },
    {"Field": "MBE Certification", "Data": "Not certified"},
    {"Field": "Key Personnel – Project Manager", "Data": "Ramesh Iyer"},
    {"Field": "Key Personnel – Technical Lead", "Data": "Sarah Collins"},
    {"Field": "Key Personnel – Security Auditor", "Data": "James Wu"},
]

df = pd.DataFrame(company_data)

csv_path = "./data/company_data.csv"
df.to_csv(csv_path, index=False)

csv_path
