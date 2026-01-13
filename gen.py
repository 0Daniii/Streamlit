import pandas as pd
import random

# --- Configuration ---
NUM_ROWS = 500  # How many emails to generate

# --- Templates ---

# HAM (Legitimate) Components
ham_subjects = ["Project Update", "Meeting", "Lunch?", "Invoice", "Q3 Report", "Resume", "Family", "Vacation", "Notes"]
ham_greetings = ["Hi team,", "Hello,", "Dear colleague,", "Hey,", "Good morning,", "Just checking in,"]
ham_bodies = [
    "Please find the attached document regarding the Q3 sales.",
    "Are we still on for the meeting at 10 AM tomorrow?",
    "Can you review my pull request when you have a moment?",
    "I'll be out of office next week for vacation.",
    "The invoice #1023 has been paid successfully.",
    "Let's grab lunch at the new place downstairs.",
    "Here are the notes from yesterday's brainstorming session.",
    "Can you send over the updated slide deck?",
    "Great job on the presentation today!"
]
ham_closings = ["Best,", "Thanks,", "Cheers,", "Regards,", "See you soon,"]

# SPAM Components
spam_subjects = ["URGENT", "Winner!", "Verify Account", "Free Gift", "Dating", "Crypto", "Investment", "Loan Offer"]
spam_greetings = ["Dear Customer,", "URGENT NOTICE:", "Congratulations!", "Hello Friend,", "Attention!"]
spam_bodies = [
    "You have won a lottery prize of $5,000,000! Click here to claim.",
    "Your account has been flagged for suspicious activity. Verify now.",
    "Hot singles in your area want to meet you tonight!",
    "Invest in Bitcoin now and double your money in 24 hours.",
    "Nigerian Prince needs assistance transferring funds. Keep 10%.",
    "Lose 10 lbs in 2 days with this miracle pill!",
    "You have a new secure message. Click link to read.",
    "Your credit score has changed. Update your info immediately.",
    "Exclusive offer: Get a free iPhone if you complete this survey."
]
spam_closings = ["Click here now!", "Don't wait!", "Act immediately!", "Apply now,", "Unsubscribe here."]

# --- Generator Function ---
data = []

for _ in range(NUM_ROWS):
    is_spam = random.choice([True, False])
    
    if is_spam:
        text = f"{random.choice(spam_subjects)}: {random.choice(spam_greetings)} {random.choice(spam_bodies)} {random.choice(spam_closings)}"
        label = "spam"
    else:
        text = f"{random.choice(ham_subjects)}: {random.choice(ham_greetings)} {random.choice(ham_bodies)} {random.choice(ham_closings)}"
        label = "ham"
    
    data.append([text, label])

# --- Save to CSV ---
df = pd.DataFrame(data, columns=["text", "label"])
filename = "large_email_dataset.csv"
df.to_csv(filename, index=False)

print(f"✅ Successfully generated '{filename}' with {NUM_ROWS} emails!")