import pandas as pd
import os

class GenericIngester:
    @staticmethod
    def load_data(file_path):
        """
        Loads data based on file extension and handles basic encoding.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)
        
        print(f"--- [Ingester] Loading {ext.upper()}: {filename} ---")
        
        try:
            if ext == '.csv':
                return pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', low_memory=False)

            elif ext in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            elif ext == '.json':
                return pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
        except UnicodeDecodeError:
            # Fallback for weird encodings
            if ext == '.csv':
                return pd.read_csv(file_path, encoding='latin-1')
            raise
