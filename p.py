import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- 1. Session State Initialization ---
if 'df' not in st.session_state:
    # Initialize with default synthetic dataset
    ham_messages = [
        "Hi team, please find the Q3 report attached.",
        "Meeting at 10 AM in conference room B. Don't be late!",
        "Could you review my pull request by end of day? Thanks!",
        "Your Amazon order #12345 has been shipped.",
        "Recipe for a delicious chocolate cake: 2 cups flour, 1 cup sugar.",
        "Reminder: Your doctor's appointment is scheduled for tomorrow.",
        "Monthly newsletter from your favorite tech blog.",
        "Project update: We've successfully integrated the new module.",
        "Family dinner this Sunday at my place, hope you can make it!",
        "Regarding your query, please refer to the attached documentation."
    ]
    spam_messages = [
        "Win a free iPhone now! Click this link to claim your prize!",
        "URGENT: Your account has been suspended. Verify your details here.",
        "Congratulations! You've won a cash prize of $1,000,000.",
        "Enlarge your manhood by 5 inches with our revolutionary product!",
        "Nigerian Prince needs your help to transfer funds.",
        "Limited time offer: Get rich quick! Invest in crypto now.",
        "You have a new message from a secret admirer. Click here!",
        "Claim your free vacation to a tropical island.",
        "Your credit card has been compromised. Update immediately.",
        "Unlock exclusive discounts! Click this phishing link now!"
    ]
    texts = ham_messages + spam_messages
    labels = ['ham'] * len(ham_messages) + ['spam'] * len(spam_messages)
    st.session_state['df'] = pd.DataFrame({'text': texts, 'label': labels})

# --- 2. Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predict Message", "Manage Dataset", "Model Training & Eval"])

# --- 3. Page: Manage Dataset ---
if page == "Manage Dataset":
    st.header("🗂️ Manage Dataset")
    st.write("View the current dataset, add new examples manually, or upload a CSV file.")

    st.subheader("Current Dataset")
    st.write(f"Total records: **{len(st.session_state['df'])}**")
    
    dist_df = st.session_state['df']['label'].value_counts()
    st.bar_chart(dist_df)
    
    with st.expander("View Raw Data"):
        st.dataframe(st.session_state['df'])

    st.markdown("---")
    st.subheader("Add New Email Manually")
    col1, col2 = st.columns([3, 1])
    with col1:
        new_text = st.text_input("Email Text", placeholder="e.g., Discount 50% off...")
    with col2:
        new_label = st.selectbox("Label", ["spam", "ham"])
    
    if st.button("Add Entry"):
        if new_text:
            new_row = pd.DataFrame({'text': [new_text], 'label': [new_label]})
            st.session_state['df'] = pd.concat([st.session_state['df'], new_row], ignore_index=True)
            st.success("Added successfully!")
            st.rerun()
        else:
            st.warning("Please enter text.")

    st.markdown("---")
    st.subheader("Upload External Dataset")
    st.info("Format Required: A CSV file with two columns named **'text'** and **'label'**.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    # Track processed files to prevent loops
    if 'processed_files' not in st.session_state:
        st.session_state['processed_files'] = []

    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}-{uploaded_file.size}"
        
        if file_id not in st.session_state['processed_files']:
            try:
                new_data = pd.read_csv(uploaded_file)
                if 'text' in new_data.columns and 'label' in new_data.columns:
                    st.session_state['df'] = pd.concat([st.session_state['df'], new_data], ignore_index=True)
                    
                    # Mark file as processed
                    st.session_state['processed_files'].append(file_id)
                    
                    st.success(f"Successfully added {len(new_data)} rows from file!")
                    st.rerun()
                else:
                    st.error("Error: CSV must contain 'text' and 'label' columns.")
            except Exception as e:
                st.error(f"Error reading file: {e}")

# --- 4. Page: Model Training & Evaluation ---
elif page == "Model Training & Eval":
    st.header("⚙️ Model Training & Evaluation")
    
    test_size = st.slider("Test Set Size (Percentage)", min_value=10, max_value=50, value=20, step=5)
    
    if st.button("Train and Evaluate Model"):
        df = st.session_state['df']
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df['text'])
        y = df['label']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=42
        )

        model = MultinomialNB()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        st.subheader("Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model Accuracy", f"{accuracy:.2%}")
        
        st.write("### Confusion Matrix")
        unique_labels = sorted(list(set(y)))
        cm_df = pd.DataFrame(cm, index=[f"Actual {l}" for l in unique_labels], columns=[f"Pred {l}" for l in unique_labels])
        st.table(cm_df)

        st.write("### Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

# --- 5. Page: Predict Message (Home) ---
elif page == "Predict Message":
    st.header("📧 Spam Detector")
    st.caption(f"Current Model trained on {len(st.session_state['df'])} examples.")

    # Train model on full dataset
    df = st.session_state['df']
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['text'])
    y = df['label']
    model = MultinomialNB()
    model.fit(X, y)

    # Input Area
    user_message = st.text_area("Enter email text:", height=150)

    # We use session_state to store the result so it persists when we click feedback buttons
    if st.button("Analyze"):
        if user_message:
            vec_message = vectorizer.transform([user_message])
            prediction = model.predict(vec_message)[0]
            probs = model.predict_proba(vec_message)
            
            classes = model.classes_
            spam_index = list(classes).index("spam")
            ham_index = list(classes).index("ham")
            
            # Save result to session state
            st.session_state['last_analysis'] = {
                'text': user_message,
                'prediction': prediction,
                'spam_prob': probs[0][spam_index],
                'ham_prob': probs[0][ham_index]
            }
        else:
            st.warning("Please enter text.")

    # Display Result and Feedback Loop (if an analysis exists)
    if 'last_analysis' in st.session_state:
        res = st.session_state['last_analysis']
        
        st.write("---")
        st.subheader("Analysis Result")
        
        if res['prediction'] == "spam":
            st.error(f"🚨 **SPAM** Detected ({res['spam_prob']*100:.2f}% confidence)")
        else:
            st.success(f"✅ **HAM** (Legitimate) ({res['ham_prob']*100:.2f}% confidence)")

        st.write("") 
        st.write("### 🧠 Improve the Model")
        st.write("Was this result correct? Add this email to the training data with the **correct** label:")

        # Feedback Buttons
        col1, col2 = st.columns(2)
        
        # Button 1: Add as SPAM
        if col1.button("Add to dataset as SPAM"):
            new_row = pd.DataFrame({'text': [res['text']], 'label': ['spam']})
            st.session_state['df'] = pd.concat([st.session_state['df'], new_row], ignore_index=True)
            st.success("✅ Added to dataset as SPAM! The model will learn from this.")
            # Clear the analysis so the user can start fresh
            del st.session_state['last_analysis']
            st.rerun()

        # Button 2: Add as HAM
        if col2.button("Add to dataset as HAM"):
            new_row = pd.DataFrame({'text': [res['text']], 'label': ['ham']})
            st.session_state['df'] = pd.concat([st.session_state['df'], new_row], ignore_index=True)
            st.success("✅ Added to dataset as HAM! The model will learn from this.")
            del st.session_state['last_analysis']
            st.rerun()