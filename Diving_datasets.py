import pandas as pd

def diving_data(train, test, submission):
    # Obtener las dimensiones de los datasets
    num_train_rows, num_train_columns = train.shape
    num_test_rows, num_test_columns = test.shape
    num_submission_rows, num_submission_columns = submission.shape

    # Imprimir información sobre el dataset de entrenamiento
    print("Training Data:")
    print(f"Number of Rows: {num_train_rows}")
    print(f"Number of Columns: {num_train_columns}\n")

    # Imprimir información sobre el dataset de test
    print("Test Data:")
    print(f"Number of Rows: {num_test_rows}")
    print(f"Number of Columns: {num_test_columns}\n")

    # Imprimir información sobre el dataset de submission
    print("Submission Data:")
    print(f"Number of Rows: {num_submission_rows}")
    print(f"Number of Columns: {num_submission_columns}")

    # Contar valores nulos en los datasets
    train_null = train.isnull().sum().sum()
    test_null = test.isnull().sum().sum()

    print(f'\nNull Count in Train: {train_null}')
    print(f'Null Count in Test: {test_null}')

    # Contar filas duplicadas en los datasets
    train_duplicates = train.duplicated().sum()
    test_duplicates = test.duplicated().sum()
    submission_duplicates = submission.duplicated().sum()

    print(f"\nNumber of duplicate rows in train data: {train_duplicates}")
    print(f"Number of duplicate rows in test data: {test_duplicates}")
    print(f"Number of duplicate rows in submission data: {submission_duplicates}")

    # Imprimir número de valores únicos en las columnas del conjunto de entrenamiento
    print(f"\nNumber of unique values in train data:\n {train.nunique()}")

# Ejemplo de uso:
# diving_data(train, test, submission)




import numpy as np
missing_values = ['none', 'missing', '?', 'NA', 'null', 'na', 'undefined', 'N/A', 'none']

def replace_missing_values(df):
    return df.applymap(lambda x: np.nan if str(x).lower() in missing_values else x)
