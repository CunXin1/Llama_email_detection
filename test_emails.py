"""
Anti-Fraud Email Analysis — Test Suite
5 brand-new emails: 3 legitimate, 2 realistic-looking fraud emails.
Sends each to the running service at http://localhost:8000/predict and prints results.
"""

import json
import sys
import requests

BASE_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# 5 Test Emails
# ---------------------------------------------------------------------------
# Label: True = expected fraud, False = expected legitimate

TEST_CASES = [
    # ── LEGITIMATE EMAIL 1 ──────────────────────────────────────────────────
    # UC Merced IT department scheduled maintenance notice
    {
        "label": "LEGITIMATE",
        "expected_fraud": False,
        "email": {
            "date": "2026-02-25T09:00:00Z",
            "sender": "it-support@ucmerced.edu",
            "recipient": "rsun@ucmerced.edu",
            "copy": "helpdesk@ucmerced.edu",
            "subject": "Scheduled Maintenance — Campus VPN & MyUCMerced Portal (Feb 28, 2:00–4:00 AM)",
            "content": (
                "Hi Ruibo,\n\n"
                "The UC Merced IT team will be performing scheduled maintenance on the Campus VPN service "
                "and the MyUCMerced portal on Saturday, February 28, 2026, from 2:00 AM to 4:00 AM PST.\n\n"
                "During this window, you may experience brief interruptions when connecting to on-campus "
                "resources remotely. No action is required on your part — services will resume automatically "
                "after the maintenance window closes.\n\n"
                "If you experience issues after 4:00 AM, please contact the IT Help Desk at "
                "helpdesk@ucmerced.edu or call (209) 228-HELP (4357).\n\n"
                "Thank you for your patience.\n\n"
                "UC Merced IT Services\n"
                "University of California, Merced\n"
                "helpdesk@ucmerced.edu | (209) 228-4357"
            ),
        },
    },

    # ── LEGITIMATE EMAIL 2 ──────────────────────────────────────────────────
    # GitHub pull request review request from a real colleague
    {
        "label": "LEGITIMATE",
        "expected_fraud": False,
        "email": {
            "date": "2026-02-25T16:42:00Z",
            "sender": "notifications@github.com",
            "recipient": "rsun@ucmerced.edu",
            "copy": "",
            "subject": "[ML-Research/fraud-detector] Review requested: Add transformer-based feature extractor (#47)",
            "content": (
                "Hi rsun,\n\n"
                "jchen-ucm has requested your review on Pull Request #47 in ML-Research/fraud-detector:\n\n"
                "  Title : Add transformer-based feature extractor\n"
                "  Branch: feature/transformer-encoder → main\n"
                "  Files : 6 files changed, +312 −48\n\n"
                "Please visit the pull request to review the changes:\n"
                "https://github.com/ML-Research/fraud-detector/pull/47\n\n"
                "You are receiving this notification because you are listed as a code owner for "
                "the files changed in this pull request.\n\n"
                "— GitHub"
            ),
        },
    },

    # ── LEGITIMATE EMAIL 3 ──────────────────────────────────────────────────
    # Payroll department notifying about pay stub availability
    {
        "label": "LEGITIMATE",
        "expected_fraud": False,
        "email": {
            "date": "2026-02-25T08:15:00Z",
            "sender": "payroll@ucmerced.edu",
            "recipient": "rsun@ucmerced.edu",
            "copy": "",
            "subject": "Your February 2026 Pay Stub Is Now Available",
            "content": (
                "Dear Ruibo Sun,\n\n"
                "Your pay stub for the pay period ending February 22, 2026 is now available in "
                "the UC Merced Employee Self-Service portal. You can access it by logging into "
                "UCPath at https://ucpath.universityofcalifornia.edu using your UC credentials.\n\n"
                "Pay Date     : February 28, 2026\n"
                "Net Pay      : Please log in to view your earnings statement\n\n"
                "If you have questions about your pay, contact the Payroll Office at "
                "payroll@ucmerced.edu or visit us in Kolligian Library, Room 100.\n\n"
                "Regards,\n"
                "UC Merced Payroll Services\n"
                "payroll@ucmerced.edu | (209) 228-2101"
            ),
        },
    },

    # ── FRAUD EMAIL 1 (Realistic) ────────────────────────────────────────────
    # Convincing fake Microsoft 365 "unusual sign-in detected" alert.
    # Designed to look like a real security notification: proper tone,
    # correct-looking layout, personalized detail — but sender domain is
    # microsoft365-alerts.com (not microsoft.com) and the verification link
    # points to microsoft-account-secure.net (not microsoft.com).
    {
        "label": "FRAUD",
        "expected_fraud": True,
        "email": {
            "date": "2026-02-25T04:37:00Z",
            "sender": "security-noreply@microsoft365-alerts.com",
            "recipient": "rsun@ucmerced.edu",
            "copy": "",
            "subject": "Microsoft Account: Unusual sign-in activity detected",
            "content": (
                "Dear Ruibo Sun,\n\n"
                "We detected a sign-in to your Microsoft 365 account from an unrecognized device.\n\n"
                "  Date and time : February 25, 2026, 4:31 AM UTC\n"
                "  Location      : Chengdu, China\n"
                "  App           : Microsoft Outlook\n"
                "  IP address    : 103.72.14.88\n\n"
                "If this was you, you can safely ignore this message.\n\n"
                "If you did not sign in, your account may be compromised. Please verify your "
                "identity and reset your password immediately by visiting the Microsoft Account "
                "Security Center:\n\n"
                "  https://microsoft-account-secure.net/verify?token=RSN8821UCM\n\n"
                "This link will expire in 30 minutes for your protection. If you need assistance, "
                "contact Microsoft Support through the official portal.\n\n"
                "Microsoft Security Team\n"
                "Microsoft Corporation"
            ),
        },
    },

    # ── FRAUD EMAIL 2 (Realistic) ────────────────────────────────────────────
    # Convincing fake Wells Fargo transaction alert asking to confirm a
    # large wire transfer.  The email mimics the tone of a real bank fraud
    # alert — calm, professional, uses the recipient's name — but the sender
    # domain is wellsfargo-fraudalerts.com and the "confirm / deny" button
    # resolves to wellsfargo-secure-center.com (not wellsfargo.com).
    {
        "label": "FRAUD",
        "expected_fraud": True,
        "email": {
            "date": "2026-02-25T13:22:00Z",
            "sender": "fraud-alerts@wellsfargo-fraudalerts.com",
            "recipient": "rsun@ucmerced.edu",
            "copy": "",
            "subject": "Wells Fargo Alert: Did you authorize a $3,800.00 wire transfer?",
            "content": (
                "Dear Ruibo Sun,\n\n"
                "Our fraud monitoring system has flagged a potentially unauthorized outgoing wire "
                "transfer on your Wells Fargo Everyday Checking account.\n\n"
                "  Transaction date : February 25, 2026\n"
                "  Amount           : $3,800.00\n"
                "  Recipient        : INTL TRADE PARTNERS LTD\n"
                "  Reference #      : WF-2026-TXN-00441\n\n"
                "To confirm or deny this transaction and secure your account, please verify your "
                "identity using the link below. This step protects you from unauthorized activity.\n\n"
                "  https://wellsfargo-secure-center.com/verify?ref=WF2026441\n\n"
                "You will be asked to confirm your Online Banking credentials and the last 4 digits "
                "of your debit card. If you did authorize this transfer, no further action is needed "
                "after verification.\n\n"
                "If we do not receive a response within 2 hours, we may temporarily restrict your "
                "account to prevent further unauthorized activity.\n\n"
                "Wells Fargo Fraud Prevention Team\n"
                "Wells Fargo Bank, N.A."
            ),
        },
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RISK_BAR_WIDTH = 30

def risk_bar(score: int) -> str:
    filled = round(score / 100 * RISK_BAR_WIDTH)
    bar = "#" * filled + "." * (RISK_BAR_WIDTH - filled)
    return f"[{bar}] {score}/100"


def verdict_icon(is_fraud: bool) -> str:
    return "[FRAUD]" if is_fraud else "[SAFE ]"


def correct_icon(predicted: bool, expected: bool) -> str:
    return "CORRECT  " if predicted == expected else "INCORRECT"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ── Health check ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("  Anti-Fraud Email Analysis — Test Run")
    print("=" * 70)
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=10)
        r.raise_for_status()
        health = r.json()
        print(f"  Service status : {health.get('status')}")
        print(f"  Model          : {health.get('model')}")
        print(f"  dtype          : {health.get('dtype')}")
    except requests.exceptions.ConnectionError:
        print(f"  ERROR: Cannot reach {BASE_URL}. Is the Docker container running?")
        sys.exit(1)
    print("=" * 70)
    print()

    results = []

    for idx, case in enumerate(TEST_CASES, start=1):
        email = case["email"]
        label = case["label"]
        expected = case["expected_fraud"]

        print(f"  -- Email {idx}/5  [{label}] {'-' * (50 - len(label))}")
        print(f"  From   : {email['sender']}")
        print(f"  Subject: {email['subject'][:65]}")
        print()

        payload = {
            "date":      email["date"],
            "sender":    email["sender"],
            "recipient": email["recipient"],
            "copy":      email.get("copy", ""),
            "subject":   email["subject"],
            "content":   email["content"],
        }

        try:
            resp = requests.post(
                f"{BASE_URL}/predict",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            result = resp.json()
        except requests.exceptions.RequestException as e:
            print(f"  [ERROR] Request failed: {e}\n")
            results.append({"idx": idx, "error": str(e)})
            continue

        is_fraud        = result["is_fraud"]
        risk_score      = result["risk_score"]
        confidence      = result["confidence_level"]
        threats         = result["detected_threats"]
        reason          = result["reason"]
        suggestion      = result["suggestion"]

        print(f"  Verdict    : {verdict_icon(is_fraud)}")
        print(f"  Risk Score : {risk_bar(risk_score)}")
        print(f"  Confidence : {confidence:.0%}")
        if threats:
            print(f"  Threats    : {', '.join(threats)}")
        else:
            print(f"  Threats    : (none detected)")
        print()
        print(f"  Reason     : {reason}")
        print()
        print(f"  Suggestion : {suggestion}")
        print()
        print(f"  Expected   : {'FRAUD' if expected else 'SAFE '}")
        print(f"  Result     : {correct_icon(is_fraud, expected)}")
        print()
        print("  " + "-" * 66)
        print()

        results.append({
            "idx": idx,
            "label": label,
            "expected": expected,
            "predicted": is_fraud,
            "correct": is_fraud == expected,
            "risk_score": risk_score,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    total   = len([r for r in results if "error" not in r])
    correct = sum(1 for r in results if r.get("correct", False))
    errors  = len([r for r in results if "error" in r])
    print(f"  Total emails tested : {len(TEST_CASES)}")
    print(f"  Successful calls    : {total}")
    print(f"  Correct predictions : {correct}/{total}")
    print(f"  Request errors      : {errors}")
    print()
    for r in results:
        if "error" in r:
            print(f"  Email {r['idx']}: ERROR - {r['error']}")
        else:
            icon = "OK" if r["correct"] else "FAIL"
            print(
                f"  Email {r['idx']} [{r['label']:10s}] "
                f"expected={'FRAUD' if r['expected'] else 'SAFE ':6s} "
                f"predicted={'FRAUD' if r['predicted'] else 'SAFE ':6s} "
                f"risk={r['risk_score']:3d}  {icon}"
            )
    print("=" * 70)


if __name__ == "__main__":
    main()
