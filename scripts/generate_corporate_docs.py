"""
Generates rich corporate policy PDFs for CorpLens and builds their FAISS indices.
"""

import os
import sys

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import fitz  # PyMuPDF

# Ensure project root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.indexing.build_index import build_index_for_paper


def create_pdf(filename, title, department, sections):
    """
    Creates a multi-page PDF document with header and structured sections.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    doc = fitz.open()

    for page_idx, (section_title, section_body) in enumerate(sections):
        page = doc.new_page(width=595, height=842)  # A4 size

        # Header bar
        header_text = f"ACME GLOBAL ENTERPRISE  |  POLICY DIRECTIVE: {department.upper()}"
        page.insert_text(fitz.Point(50, 45), header_text, fontsize=9, fontname="helv", color=(0.4, 0.4, 0.4))

        # Divider line
        page.draw_line(fitz.Point(50, 55), fitz.Point(545, 55), color=(0.8, 0.8, 0.8), width=1)

        # Page Title / Subtitle on first page
        y_offset = 80
        if page_idx == 0:
            page.insert_text(fitz.Point(50, y_offset), title, fontsize=18, fontname="helv", color=(0.1, 0.2, 0.4))
            y_offset += 30
            meta = f"Classification: Internal Corporate Directive  •  Version: 4.0  •  Department: {department.title()}"
            page.insert_text(fitz.Point(50, y_offset), meta, fontsize=10, fontname="helv", color=(0.3, 0.3, 0.3))
            y_offset += 25
            page.draw_line(fitz.Point(50, y_offset), fitz.Point(545, y_offset), color=(0.2, 0.4, 0.7), width=2)
            y_offset += 25

        # Section Header
        page.insert_text(fitz.Point(50, y_offset), section_title, fontsize=13, fontname="helv", color=(0.15, 0.15, 0.15))
        y_offset += 20

        # Section Body
        body_rect = fitz.Rect(50, y_offset, 545, 780)
        page.insert_textbox(body_rect, section_body.strip(), fontsize=9.5, fontname="helv", color=(0.2, 0.2, 0.2), lineheight=1.35)

        # Footer
        footer_text = f"Page {page_idx + 1}  •  Acme Global Confidential Corporate Document"
        page.insert_text(fitz.Point(180, 810), footer_text, fontsize=8, fontname="helv", color=(0.5, 0.5, 0.5))

    doc.save(filename)
    doc.close()
    print(f"Generated: {filename}")


# -------------------- COMPREHENSIVE DOCUMENT DEFINITIONS --------------------

DOCS = {
    "employee_handbook": {
        "title": "Global Employee Workplace Handbook & Code of Conduct",
        "dept": "General HR",
        "path": "data/corporate_docs/employee_handbook/document.pdf",
        "save_dir": "data/corporate_docs/employee_handbook",
        "sections": [
            (
                "1. Organizational Mission, Hybrid & Remote Work Policy",
                """Acme Global Enterprise is dedicated to fostering an ethical, inclusive, and high-performance workplace. All team members must uphold integrity, transparency, and legal compliance across all global operations.

1.1 Core Collaboration Hours: Acme operates on a hybrid schedule. Mandatory core collaboration hours are 10:00 AM to 4:00 PM Eastern Standard Time (EST), Monday through Thursday. During these hours, all personnel must be reachable via Slack, Zoom, and corporate email.

1.2 On-Site Attendance: Office-based employees must be on-site at least two (2) days per week, specifically Tuesday and Thursday. 

1.3 Remote Work Setup Stipend: Fully remote employees receive a one-time tax-free home-office technology stipend of $500 USD upon hire for ergonomic chairs, monitors, or desks. Remote employees are also eligible for an ongoing monthly home broadband internet stipend of $50 USD."""
            ),
            (
                "2. Paid Time Off (PTO), Leaves of Absence & Parental Leave",
                """2.1 Vacation Entitlement: Permanent full-time employees accrue 20 paid vacation days per calendar year. A maximum of five (5) unused vacation days may be rolled over into the next year, which must be redeemed before March 31.

2.2 Sick and Mental Health Leave: All employees receive 10 paid sick days annually. Formal medical documentation is required only for continuous absences exceeding three (3) consecutive working days.

2.3 Paid Parental Leave: Acme offers sixteen (16) weeks of 100% fully paid parental leave for all eligible primary and secondary caregivers following the birth, adoption, or foster placement of a child. Leave must be taken within twelve (12) months of the qualifying event.

2.4 Corporate Holidays: Acme recognizes twelve (12) official paid holidays per year, including Memorial Day, Juneteenth, Independence Day, Labor Day, Thanksgiving (Thursday & Friday), and Winter Break (Dec 24-25)."""
            ),
            (
                "3. Healthcare, Retirement Benefits & Learning Stipend",
                """3.1 Health Insurance: Acme provides comprehensive medical, dental, and vision insurance beginning on the employee's very first day of employment. Acme pays 85% of employee coverage premiums and 70% of dependent coverage.

3.2 401(k) Matching: Acme matches 100% of employee 401(k) contributions up to the first 4% of base pay, plus 50% on the next 2%, delivering a maximum company match of 5%. All employer matching contributions vest immediately on day one.

3.3 Professional Development Budget: Every permanent employee is allocated an annual learning budget of $1,500 USD for professional certifications, industry conferences, and approved academic coursework."""
            ),
            (
                "4. Performance Reviews, Promotions & Improvement Plans",
                """4.1 Review Cycles: Performance evaluations occur bi-annually in June (mid-year check-in) and December (annual focal review). Reviews incorporate 360-degree peer feedback, self-evaluations, and measurable Objectives and Key Results (OKRs).

4.2 Promotion Criteria: Promotions require sustained performance at the next career band for at least six (6) consecutive months, manager recommendation, and calibration approval by the Department Compensation Committee.

4.3 Performance Improvement Plan (PIP): If an employee's performance fails to meet role expectations, a formal 60-day Performance Improvement Plan (PIP) is established with clear weekly milestones. If satisfactory performance is not achieved after 60 days, employment may be terminated with standard severance."""
            ),
            (
                "5. Moonlighting, Secondary Employment & Conflicts of Interest",
                """5.1 Outside Employment: Employees may engage in outside freelancing or teaching provided it does not interfere with Acme obligations. Secondary employment must be declared and pre-cleared via the Legal Compliance Portal.

5.2 Strict Prohibitions: Employees may never perform work for an Acme competitor, consult for an active vendor, or utilize Acme hardware, software licenses, or intellectual property for outside commercial ventures. Outside work is capped at a maximum of 15 hours per calendar week."""
            ),
            (
                "6. Resignation, Severance & Hardware Offboarding",
                """6.1 Notice Period: Individual contributors are requested to provide two (2) weeks written notice of resignation. Managers and Directors must provide four (4) weeks advance written notice.

6.2 Severance Entitlement: In involuntary non-disciplinary reductions in force (RIF), severance is provided at two (2) weeks of base pay per completed year of tenure, with a minimum of four (4) weeks and a maximum cap of twenty-six (26) weeks.

6.3 Equipment Return: All company-issued hardware, including laptops, security keys, and corporate credit cards, must be returned to IT Asset Management within five (5) business days of the termination date via prepaid courier box."""
            )
        ]
    },

    "infosec_policy": {
        "title": "Information Security, Data Protection & IT Acceptable Use Policy",
        "dept": "InfoSec Legal",
        "path": "data/corporate_docs/infosec_policy/document.pdf",
        "save_dir": "data/corporate_docs/infosec_policy",
        "sections": [
            (
                "1. Data Classification Scheme & Encryption Standards",
                """All digital and physical information assets owned or processed by Acme Global Enterprise must be categorized into one of four classification tiers:

1.1 Tier 1 - Public: Marketing collaterals, press releases, public open-source repositories. No confidentiality controls required.

1.2 Tier 2 - Internal: Internal memos, company wiki pages, organizational rosters. Access limited to authenticated Acme staff.

1.3 Tier 3 - Confidential: Customer Personally Identifiable Information (PII), employee payroll files, proprietary source code, and commercial contract drafts. Mandatory AES-256 encryption at rest and TLS 1.3 encryption in transit.

1.4 Tier 4 - Highly Restricted: Executive M&A strategies, master database cryptographic keys, banking credentials, and pre-release quarterly financial reports. Access requires two-person approval and zero-trust hardware attestation."""
            ),
            (
                "2. Password Standards, MFA & Endpoint Device Security",
                """2.1 Passphrase Complexity: Passphrases must be a minimum of sixteen (16) characters in length, containing uppercase letters, lowercase letters, numbers, and symbols. Automated rotation is enforced every 90 days.

2.2 Multi-Factor Authentication (MFA): MFA is mandatory across all cloud applications, VPN endpoints, and email suites. Hardware FIDO2 security keys (YubiKeys) or certified authenticator apps (Google/Microsoft Authenticator) are required. SMS-based 2FA is strictly banned due to vulnerability to SIM-swap attacks.

2.3 Screen Inactivity Lock: All workstations and laptops must automatically lock after five (5) minutes of user inactivity. 

2.4 Full-Disk Encryption: BitLocker (Windows) and FileVault (macOS) full-disk encryption with 256-bit keys is mandatory. Removable USB flash drives and external storage devices are blocked by endpoint management software."""
            ),
            (
                "3. Incident Response Protocol & Data Breach SLAs",
                """3.1 SLA for Incident Reporting: Any team member who discovers or suspects a security breach, lost corporate laptop, ransomware infection, or unauthorized data access must notify the Security Operations Center (SOC) within sixty (60) minutes of discovery at incident-response@acmeglobal.com.

3.2 Containment SLA: The SOC must acknowledge P1 critical breaches within 15 minutes and achieve network isolation/containment within four (4) hours.

3.3 Proprietary Information: Under the signed Proprietary Information and Inventions Agreement (PIIA), all code, models, and technical architecture authored during employment remain exclusive corporate property."""
            ),
            (
                "4. Artificial Intelligence (AI) & LLM Acceptable Use Policy",
                """4.1 Public AI Tools Prohibition: Entering Tier 3 (Confidential) or Tier 4 (Highly Restricted) corporate data, customer data, or internal source code into public, consumer AI tools (e.g., standard ChatGPT, Claude, Gemini) is strictly forbidden.

4.2 Approved Enterprise AI: Employees may only use corporate-contracted enterprise AI accounts where data privacy agreements guarantee zero data retention and zero training on Acme inputs.

4.3 AI Code Assistants: AI code completion tools (e.g., GitHub Copilot Enterprise) are permitted only on approved repositories and all AI-generated code must undergo mandatory automated Static Application Security Testing (SAST) and peer review before merging."""
            ),
            (
                "5. Remote Access, VPN & Wi-Fi Security Protocols",
                """5.1 Public Wi-Fi Warning: When working from coffee shops, hotels, or airports, connecting via the Acme Corporate WireGuard VPN is mandatory prior to opening any business application.

5.2 Split Tunneling Prohibited: Split tunneling on corporate VPN connections is permanently disabled to ensure all network traffic passes through cloud threat inspection and intrusion detection filters.

5.3 Mobile Device Management (MDM): Personal mobile devices accessing corporate email or Slack must enroll in Acme MDM. Rooted or jailbroken mobile devices are blocked by automated compliance policies."""
            ),
            (
                "6. Cloud Infrastructure, Secrets & Key Management",
                """6.1 Zero-Trust IAM: Access to AWS, Google Cloud, and Azure environments is governed by Least Privilege Role-Based Access Control (RBAC). Permanent administrative credentials are prohibited; just-in-time (JIT) elevated access expires after four (4) hours.

6.2 Secret Key Rotation: Production API keys, database credentials, and service account tokens must be rotated every 90 days. 

6.3 Hardcoded Secrets Ban: Storing plaintext passwords or API keys in Git source repositories is a critical security violation. All repositories enforce automated pre-commit secret scanning hooks (GitLeaks) that block commits containing credentials."""
            )
        ]
    },

    "procurement_policy": {
        "title": "Corporate Financial Governance, Travel & Procurement Manual",
        "dept": "Finance Governance",
        "path": "data/corporate_docs/procurement_policy/document.pdf",
        "save_dir": "data/corporate_docs/procurement_policy",
        "sections": [
            (
                "1. Business Travel, Flight Classes, Hotels & Meal Per Diems",
                """1.1 Air Travel Rules: Economy class is mandatory for all domestic flights and international flights under six (6) hours in duration. Premium Economy or Business Class is authorized for non-stop international flights exceeding six (6) hours, subject to prior VP authorization.

1.2 Hotel Nightly Caps: Standard room rates must not exceed $220 USD per night in secondary cities, or $350 USD per night in designated tier-one metropolitan areas (New York, San Francisco, London, Tokyo, Zurich).

1.3 Meals Per Diem: Employees are reimbursed for reasonable business meals up to $75 USD per day for domestic travel and up to $120 USD per day for international travel. Alcoholic beverages are non-reimbursable unless directly tied to pre-approved client entertainment dinners."""
            ),
            (
                "2. Corporate Purchasing Thresholds & Authorization Matrix",
                """Every purchase order, software license subscription, and vendor contract must adhere to the formal Delegation of Financial Authority:

2.1 Tier 1 (Up to $1,000 USD): Authorized by the immediate Direct Manager. Payable via corporate credit card.

2.2 Tier 2 ($1,001 to $10,000 USD): Requires written approval from the Department Director and Finance Manager. A formal Purchase Order (PO) must be issued before vendor engagement.

2.3 Tier 3 ($10,01 to $50,000 USD): Requires written sign-off from the Functional Vice President (VP) and Head of Procurement. A minimum of two (2) independent vendor bids is required.

2.4 Tier 4 (Exceeding $50,000 USD): Requires competitive Request for Proposal (RFP), Legal counsel contract review, and final sign-off from the Chief Financial Officer (CFO)."""
            ),
            (
                "3. Corporate Cards, Expense Filing Deadlines & Receipts",
                """3.1 Filing Deadline: All corporate card transactions and out-of-pocket expenses must be submitted via the Concur expense portal within fifteen (15) calendar days from the transaction date. Reports submitted past 30 days are subject to rejection.

3.2 Receipt Requirements: Itemized receipts are mandatory for all single expenditures exceeding $25 USD. Credit card summary slips that do not list purchased items are unacceptable.

3.3 Prohibited Expenses: Corporate funds must never be utilized for personal entertainment, traffic/parking violations, airline lounge memberships, luxury upgrades, or political contributions."""
            ),
            (
                "4. Corporate Gifts, Entertainment & Hospitality Limits",
                """4.1 Receiving Gifts: Employees may accept promotional gifts of nominal value under $50 USD (e.g., branded pens, mugs). Gifts valued between $51 and $200 USD must be disclosed to the Ethics & Compliance Office. Gifts exceeding $200 USD must be politely declined.

4.2 Cash Prohibition: Accepting or giving cash, gift certificates, or cryptocurrency is strictly prohibited regardless of amount.

4.3 Anti-Bribery Compliance: In strict compliance with the U.S. Foreign Corrupt Practices Act (FCPA) and UK Bribery Act, offering, promising, or giving anything of value to a foreign government official or commercial client to obtain improper business advantage is punishable by immediate dismissal and criminal referral."""
            ),
            (
                "5. Client Billing, Invoicing & Late Payment Terms",
                """5.1 Payment Terms: Standard payment terms for all client invoices are Net 30 days from the invoice issuance date. Any custom credit terms (Net 60 or Net 90) require written approval from the Corporate Controller.

5.2 Late Payment Penalty: Client accounts overdue past 45 days incur an automatic late payment interest charge of 1.5% per month (or maximum rate permitted by law) until settled in full.

5.3 Invoice Dispute Window: Clients must register billing disputes in writing within fourteen (14) calendar days of invoice receipt; undisputed portions must be remitted on schedule."""
            ),
            (
                "6. Hardware Replacements & Monthly Communication Stipend",
                """6.1 Mobile Phone Reimbursement: Employees in eligible client-facing, sales, and on-call engineering roles receive a monthly mobile communication stipend of $75 USD.

6.2 Home Internet Stipend: All remote and hybrid employees receive a monthly home broadband stipend of $50 USD reimbursed automatically through payroll.

6.3 Laptop Lifecycle Refresh: Standard corporate laptops are replaced on a mandatory 36-month (3-year) lifecycle. Earlier replacements require Director approval demonstrating hardware failure."""
            )
        ]
    },

    "engineering_standards": {
        "title": "Software Engineering, Architecture & Operational Standards",
        "dept": "Engineering Tech",
        "path": "data/corporate_docs/engineering_standards/document.pdf",
        "save_dir": "data/corporate_docs/engineering_standards",
        "sections": [
            (
                "1. SDLC, Git Branching & Code Review Mandates",
                """1.1 Branching Model: Acme Engineering operates on Trunk-Based Development. Feature branches must be short-lived (maximum lifespan of 48 hours) before being merged into the 'main' trunk.

1.2 Peer Review SLA: Every Pull Request (PR) requires a minimum of two (2) peer code reviews and approval from a designated Code Owner. PR reviews must be completed within 24 hours of submission.

1.3 Automated Test Coverage: Pull requests must achieve a minimum of 80% automated unit test code coverage. PRs that reduce overall repository test coverage are blocked automatically by CI pipelines."""
            ),
            (
                "2. API Design Guidelines & Semantic Versioning",
                """2.1 Semantic Versioning: All internal and public microservice APIs must strictly follow Semantic Versioning (vMAJOR.MINOR.PATCH). Breaking schema changes require incrementing the MAJOR version.

2.2 Schema Definitions: All REST APIs must publish comprehensive OpenAPI 3.0 (Swagger) specifications. High-throughput inter-service communication must utilize gRPC with compiled Protocol Buffers (Protobuf).

2.3 Deprecation Policy: When an API endpoint is deprecated, it must be supported for a minimum of six (6) months with active deprecation headers (Sunset: date) prior to complete decommissioning."""
            ),
            (
                "3. Incident Severity Levels & On-Call Response SLAs",
                """Engineering outages and software degradation are classified under four formal incident tiers:

3.1 SEV-1 (Critical Outage): Core service down, affecting >10% of global active users. Response SLA: On-call engineer must acknowledge within 15 minutes and join the incident bridge. Status updates published every 30 minutes.

3.2 SEV-2 (Major Degradation): Major feature impaired with no immediate workaround. Response SLA: 30 minutes acknowledgement, hourly status updates.

3.3 SEV-3 (Minor Defect): System degraded but workarounds exist. Response SLA: 4 hours.

3.4 Blameless Post-Mortems: For all SEV-1 and SEV-2 incidents, a blameless post-mortem document and root-cause analysis (RCA) must be published within 72 hours of incident resolution."""
            ),
            (
                "4. Deployment Windows, Canary Releases & Freezes",
                """4.1 Allowed Deployment Windows: Production software releases are permitted Monday through Thursday between 9:00 AM and 3:00 PM EST. 

4.2 Friday Deployments Banned: Production releases are strictly prohibited on Fridays after 12:00 PM EST and throughout weekends, except for urgent SEV-1 hotfixes approved by the Engineering VP.

4.3 Canary Rollouts: Microservice deployments must use progressive canary releases (5% traffic for 30 minutes, 25% for 1 hour, then 100% full rollout), automatically rolling back if error rates exceed 0.5%.

4.4 Holiday Deployment Freeze: A global change freeze takes effect annually from December 15 through January 3 to safeguard holiday commercial stability."""
            ),
            (
                "5. Observability, Logging & PII Masking Standards",
                """5.1 Structured Logging: All applications must emit structured JSON logs with correlation IDs (`trace_id`, `span_id`) propagated across microservices using OpenTelemetry.

5.2 Strict PII Sanitization: Application logs must never contain sensitive customer PII, credit card numbers, passwords, or authentication bearer tokens. Automated regex scrubbing filters are enforced at the logging agent level.

5.3 Retention Windows: Production application logs are retained for 30 days in hot storage (Datadog) and archived for 365 days in cold cloud storage (AWS S3 Glacier) for regulatory compliance."""
            )
        ]
    }
}


def main():
    print("Generating rich Corporate Documents for CorpLens...")

    for key, info in DOCS.items():
        pdf_path = info["path"]
        save_dir = info["save_dir"]

        # 1. Create PDF
        create_pdf(pdf_path, info["title"], info["dept"], info["sections"])

        # 2. Build FAISS index & Chunks
        print(f"Indexing {key} into {save_dir}...")
        build_index_for_paper(pdf_path, save_dir)

    print("\nAll 4 comprehensive corporate documents generated and indexed successfully.")


if __name__ == "__main__":
    main()
