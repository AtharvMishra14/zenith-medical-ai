import pandas as pd
import re
from datasets import load_dataset

print("1. 📥 Downloading 20,000 dirty clinical records from HuggingFace...")
# We use a massive dataset that mimics real, unstructured hospital notes
dataset = load_dataset("starmpcc/Asclepius-Synthetic-Clinical-Notes", split="train[:20000]")

# Convert to a Pandas DataFrame for heavy-duty cleaning
df = pd.DataFrame(dataset)

# The dataset has a 'note' column containing the raw doctor's text
# Let's see how messy it is before we start
print(f"\n📊 Raw dataset size: {df.shape[0]} rows")
print("First 100 characters of raw data:")
print(f"[{df['note'].iloc[0][:100]}]...\n")

print("2. 🧹 Initiating the Data Cleaning Pipeline...")

# A. Drop any corrupted or empty records
initial_rows = len(df)
df = df.dropna(subset=['note'])
print(f"   - Dropped {initial_rows - len(df)} empty/corrupted rows.")

# B. The Regex Scrubbing (Fixing formatting and OCR errors)
def scrub_text(text):
    if not isinstance(text, str):
        return ""
    
    # Remove multiple spaces, tabs, and weird line breaks
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters that confuse the AI (keep punctuation)
    text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
    
    # C. Expand Medical Abbreviations (The "Flex" for your professor)
    # \b ensures we only replace exact words, not parts of words (like 'pt' inside 'optic')
    replacements = {
        r'\bpt\b': 'patient',
        r'\bhx\b': 'history',
        r'\bdx\b': 'diagnosis',
        r'\brx\b': 'prescription',
        r'\btx\b': 'treatment',
        r'\bsx\b': 'symptoms',
        r'\bp/w\b': 'presents with',
        r'\bc/o\b': 'complains of'
    }
    
    for abbr, full_word in replacements.items():
        text = re.sub(abbr, full_word, text, flags=re.IGNORECASE)
        
    return text.strip()

# Apply the scrubbing engine to the entire dataset
df['clean_note'] = df['note'].apply(scrub_text)

print("3. ✨ Cleaning Complete!")
print("First 100 characters of CLEANED data:")
print(f"[{df['clean_note'].iloc[0][:100]}]...\n")

# 4. Save the pristine data for the Vector Database
output_file = "cleaned_massive_clinical_data.csv"
# We only save the clean column to save space
df[['clean_note']].to_csv(output_file, index=False)
print(f"💾 Saved pristine dataset to {output_file}")