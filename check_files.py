import pickle

def verify_pickle(file_path, attribute=None):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            if attribute and hasattr(data, attribute):
                return getattr(data, attribute)
            return data
    except Exception as e:
        return f"Error loading {file_path}: {e}"

# Check all three files
cols = verify_pickle('columns.pkl')
model_features = verify_pickle('model.pkl', 'n_features_in_')
scaler_features = verify_pickle('scaler.pkl', 'n_features_in_')

print("--- Pickle File Verification ---")
if not isinstance(cols, str):
    print(f"File 'columns.pkl' has {len(cols)} columns.")
    print(f"First few columns: {list(cols)[:5]}...")
else:
    print(cols)

print(f"Model expects: {model_features} features")
print(f"Scaler expects: {scaler_features} features")