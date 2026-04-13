import os

print("Starting the database radar...")
found = False

# Walk through every single folder and sub-folder
for root, dirs, files in os.walk("."):
    if "chroma.sqlite3" in files:
        found = True
        filepath = os.path.join(root, "chroma.sqlite3")
        size_kb = os.path.getsize(filepath) / 1024
        
        print(f"\n🎯 FOUND IT!")
        print(f"Exact Path: {root}")
        print(f"File Size: {size_kb:.2f} KB")
        
        if size_kb < 50:
            print("⚠️ WARNING: This database file is tiny! It might be empty.")
        else:
            print("✅ Size looks good. This database contains data.")

if not found:
    print("\n❌ chroma.sqlite3 was NOT found anywhere in this folder. It might have extracted somewhere else!")