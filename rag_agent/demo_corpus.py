"""Synthetic demo corpus: internal wiki pages for a fictional company,
"Aurora Robotics". All content is fabricated for demonstration."""

WIKI = {
    "KB-101": ("Aurora Robotics -- Remote Work Policy",
               "Aurora Robotics allows full-time employees to work remotely up to three days per week. "
               "Remote work requests are approved by the direct manager and recorded in the HR portal. "
               "Fully remote arrangements require VP approval and are reviewed every six months. "
               "Employees working remotely must be reachable on Slack during core hours of 10:00 to 16:00 "
               "in their registered time zone. The company provides a home-office stipend of 500 dollars "
               "per year, reimbursed upon submission of receipts to finance. Tax implications of the "
               "stipend vary by country and are described in the finance handbook."),
    "KB-102": ("Aurora Robotics -- Vacation and Leave",
               "Full-time employees receive 25 paid vacation days per year, accruing at approximately "
               "2.08 days per month. Unused vacation days can be carried over, up to a maximum of five "
               "days into the next year; anything beyond five days expires on March 31. Sick leave is "
               "separate from vacation and requires no advance approval; a medical certificate is needed "
               "only after five consecutive sick days. Parental leave grants 16 weeks at full pay for "
               "the primary caregiver and 6 weeks for the secondary caregiver. During parental leave, "
               "health insurance continues at company expense."),
    "KB-103": ("Aurora Robotics -- Expense Reimbursement",
               "Expense reports must be submitted within 30 days of the expense date through the finance "
               "portal. Reimbursements under 75 dollars do not require receipts, though itemization is "
               "still mandatory. The per-diem meal cap is 60 dollars for domestic travel and 80 dollars "
               "for international travel. Alcohol is not reimbursable. Flights must be booked in economy "
               "class unless the total flight time exceeds eight hours, in which case premium economy is "
               "permitted. Reimbursements are paid out with the next payroll cycle after approval."),
    "KB-104": ("Aurora Robotics -- Security Incident Reporting",
               "Any suspected security incident must be reported to security@aurora-robotics.example "
               "within 30 minutes of discovery. Do not attempt to investigate or contain the incident "
               "yourself unless you are part of the incident response team. The on-call security engineer "
               "acknowledges reports within 15 minutes and decides severity: SEV1 incidents page the CTO "
               "immediately, SEV2 within one hour. Post-incident reviews are mandatory for SEV1 and SEV2 "
               "and must be completed within five business days. Customers are notified of any incident "
               "involving their data within 72 hours, in line with contractual SLAs."),
    "KB-105": ("Aurora Robotics -- Office and Facilities",
               "The headquarters is located at 42 Harbor Lane and is open on weekdays from 07:00 to "
               "20:00 with badge access. The office has six meeting rooms bookable through the room "
               "dashboard, two quiet rooms for focused work, and a wellness room on floor three. "
               "Parking in the underground garage costs 8 dollars per day, deducted automatically from "
               "payroll for registered vehicles. The office is closed on public holidays listed in the "
               "shared calendar. Standing desks can be requested through facilities and are typically "
               "delivered within two weeks."),
}


def build_corpus():
    from rag_agent.corpus import Corpus

    corpus = Corpus()
    for doc_id, (title, text) in WIKI.items():
        corpus.add_document(doc_id, title, text)
    return corpus
