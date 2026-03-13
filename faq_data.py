"""
Mobitel HR Assistant - FAQ Data
Hardcoded FAQ question-answer pairs extracted from HR documents.
These are displayed in the UI accordion panel and used for exact-match lookups.

To update: Add/modify entries in the FAQ_DATA dictionary below.
Categories group FAQs for easy browsing in the UI.
"""

# FAQ data organized by category
# Each category has a list of (question, answer) tuples
FAQ_DATA = {
    "🏥 Medical Insurance": [
        (
            "What is the objective of Staff Medical Insurance?",
            "The main objective is to provide financial assistance to staff members and their families for medical expenses through Continental Insurance Lanka Ltd (CIL)."
        ),
        (
            "Can my spouse and children be included in the medical insurance?",
            "Yes. Your family unit (you, spouse, and children) are eligible. Children must be aged 0–25, unmarried and unemployed. Adults aged 18–70 are covered. Submit the New Member Inclusion Form through HR. Newborn babies are covered from birth but must be registered within 3 months."
        ),
        (
            "What does the insurance cover for hospitalization?",
            "Indoor hospitalization covers room charges, nursing, lab tests, X-rays, prescribed medicines, ICU, surgery fees, ambulance, blood transfusion, and more. Psychiatric treatment is also covered including counselor charges."
        ),
        (
            "How do I get hospitalized under the insurance plan?",
            "For planned hospitalization: Call Continental Insurance at 011 5 200 700 with your membership card number, patient name, hospital, contact details, and ailment. An agent will meet you at the hospital. For emergencies: Use your Continental Insurance Membership Card, get admitted, and inform CIL within 24 hours."
        ),
        (
            "How do I submit medical claims?",
            "Log in to the medi-portal at https://mediportal.cilanka.com/ to submit claims. Reimbursements are credited to your salary account after 10 working days. Check claim status under 'My Claims' in the portal. Use SLTMOPD<employee no.>A as member code."
        ),
        (
            "What is the deadline for submitting medical claims?",
            "Claims should be submitted within 3 months of the bill date (not the collection date). Delays may be entertained for unforeseeable reasons like pandemic situations."
        ),
        (
            "Does the insurance cover spectacles?",
            "Yes. Cost of spectacles, lenses including contact lenses are covered up to LKR 25,000 once in two years, for employees only."
        ),
        (
            "What dental treatments are covered?",
            "Dental treatments including extractions and nerve fillings are covered under OPD. Dental treatment under anesthesia is covered under indoor limit on reimbursement basis. Hospitalization for dental surgeries including anesthetic charges are covered under cashless basis."
        ),
        (
            "Which hospitals offer direct (cashless) settlement?",
            "Please refer to the Continental Insurance cashless hospital list available on the HR portal. For any questions, contact Continental Insurance at 011 5 200 700."
        ),
    ],
    "📱 Staff Family Package": [
        (
            "What is the Staff Family Package?",
            "These are special mobile packages introduced by Mobitel for employees and their family members as a benefit."
        ),
        (
            "Who is eligible for the Staff Family Package?",
            "The package is only available for staff members currently employed at Mobitel (Pvt) Ltd."
        ),
        (
            "What documents are required for the Staff Family Package?",
            "You need: (1) Duly filled and signed customer application form, (2) Staff ID (mandatory), (3) Signed copy of NIC, (4) Duly filled and signed consent form."
        ),
        (
            "How many connections can I get under the Staff Family Package?",
            "Each employee can obtain up to 5 connections under the Staff Family Package via their NIC. This is in addition to existing demo voice and data connections."
        ),
        (
            "Where can I get the Staff Family Package?",
            "These packages can be obtained at all Mobitel Main & Mini branches, SLT & Singer locations."
        ),
        (
            "Can I change my current connection to the Staff Family Package?",
            "Yes, package change is allowed for postpaid (Voice & Data) connections. Staff can also nominate a family member and change their connection. Ownership transfer will need to be done to the staff member's details."
        ),
    ],
}


def get_all_faqs() -> dict:
    """Return the complete FAQ data dictionary."""
    return FAQ_DATA


def search_faqs(query: str) -> str | None:
    """
    Search hardcoded FAQs for an exact or near-exact match.
    
    Args:
        query: User's question
        
    Returns:
        Answer string if match found, None otherwise
    """
    query_lower = query.lower().strip()
    
    # Remove common question starters for better matching
    for prefix in ["what is", "how do", "how can", "can i", "what are", "who is", "where can"]:
        if query_lower.startswith(prefix):
            query_lower = query_lower[len(prefix):].strip()
            break
    
    best_match = None
    best_score = 0
    
    for category, faqs in FAQ_DATA.items():
        for question, answer in faqs:
            q_lower = question.lower()
            
            # Exact match
            if query_lower == q_lower or query_lower.rstrip("?") == q_lower.rstrip("?"):
                return f"📋 **{question}**\n\n{answer}"
            
            # Word overlap matching
            query_words = set(query_lower.split())
            q_words = set(q_lower.split())
            
            # Remove common stop words
            stop_words = {"the", "is", "a", "an", "to", "for", "of", "in", "my", "i", "do", "can", "how", "what", "are"}
            query_words -= stop_words
            q_words -= stop_words
            
            if not query_words or not q_words:
                continue
            
            overlap = len(query_words & q_words)
            score = overlap / max(len(query_words), len(q_words))
            
            if score > best_score:
                best_score = score
                best_match = (question, answer)
    
    # Return if we have a strong enough match (70%+ word overlap)
    if best_score >= 0.7 and best_match:
        q, a = best_match
        return f"📋 **{q}**\n\n{a}"
    
    return None