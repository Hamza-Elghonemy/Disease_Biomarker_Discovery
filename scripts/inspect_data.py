
import pandas as pd
import os

data_dir = '/Users/hamzaelghonemy/Desktop/University/Senior/Bioinformatics/Project/Disease_Biomarker_Discovery/data'

files = [
    'abundance_tables.xlsx',
    'correlations.xlsx',
    'crc_gut_proteome.csv',
    'differential_analysis_results.xlsx',
    'rank_data.xlsx'
]

for file in files:
    file_path = os.path.join(data_dir, file)
    print(f"--- {file} ---")
    try:
        if file.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            excel_file = pd.ExcelFile(file_path)
            print(f"Sheet names: {excel_file.sheet_names}")
            df = pd.read_excel(file_path, sheet_name=0)
        
        print(df.info())
        print(df.head())
        print("\n")
    except Exception as e:
        print(f"Error reading {file}: {e}")
