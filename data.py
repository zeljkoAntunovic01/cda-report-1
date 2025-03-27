import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    def __init__(self):
        self.scaler = None
        self.num_imputer = None
        self.cat_imputer = None
        self.ordinal_maps = None  # Store per-column mapping
        self.dropped_cols = None

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Clean the dataset by removing unnecessary columns and handling missing values."""
        # Drop unnecessary columns
        if self.dropped_cols is None:
            self.dropped_cols = [col for col in df.columns if df[col].nunique() == 1]
        df.drop(columns=self.dropped_cols, errors='ignore')

        # Handle numerical missing values with KNN imputation, K = 5
        # 1. Identify x-columns (numerical)
        x_cols = [col for col in df.columns if col.startswith('x_')]

        # 2. Extract x-subset
        x_data = df[x_cols]

        # 3. Apply KNN imputation
        if self.num_imputer is None:
            self.num_imputer = KNNImputer(n_neighbors=5)
            x_imputed = pd.DataFrame(self.num_imputer.fit_transform(x_data), columns=x_cols)
        else:
            x_imputed = pd.DataFrame(self.num_imputer.transform(x_data), columns=x_cols)

        # 4. Replace original x_ columns with imputed ones
        df[x_cols] = x_imputed

        assert df[x_cols].isnull().sum() == 0, "Missing values in x_ columns after imputation"

        # Handle categorical missing values with KNN imputation, K = 5
        # 1. Identify categorical columns
        c_cols = [col for col in df.columns if col.startswith('C_')]

        # 2. Extract those columns
        c_data = df[c_cols]

        # 3. KNN impute (treating as numeric)
        if self.cat_imputer is None:
            self.cat_imputer = KNNImputer(n_neighbors=5)
            c_imputed = pd.DataFrame(self.cat_imputer.fit_transform(c_data), columns=c_cols)
        else:
            c_imputed = pd.DataFrame(self.cat_imputer.transform(c_data), columns=c_cols)

        # 4. Round to nearest valid class (assumed: 71 to 75)
        valid_classes = np.array([71, 72, 73, 74, 75])

        # Function to round each value to nearest valid class
        def round_to_valid_class(value):
            return valid_classes[np.argmin(np.abs(valid_classes - value))]

        # 5. Apply rounding
        for col in c_cols:
            c_imputed[col] = c_imputed[col].apply(round_to_valid_class)

        # 6. Replace original columns in df
        df[c_cols] = c_imputed

        assert df[c_cols].isnull().sum() == 0, "Missing values in categorical columns after imputation"
        return df

    def preprocess_train_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the dataset by cleaning the dataset, 
        standardizing numerical values of X_ columns 
        and mapping categorical C_ columns to a scale of 1-5 instead of 71-75."""

        # Clean data by removing unnecessary columns and handling missing values
        df = self.clean_data(df)

        # 1. Identify numerical x_ columns
        x_cols = [col for col in df.columns if col.startswith('x_')]

        # 2. Standardize
        self.scaler = StandardScaler()
        df[x_cols] = self.scaler.fit_transform(df[x_cols])
        # Note: For test data we need to use scaler.transform() with the same scaler fitted on train data, since it needs to use train data mean and std.

        # 3. Identify categorical C_ columns
        c_cols = [col for col in df.columns if col.startswith('C_')]

        # 4. Map categorical levels
        self.ordinal_maps = {}
        for col in c_cols:
            unique_sorted = sorted(df[col].dropna().unique())
            ordinal_map = {val: i + 1 for i, val in enumerate(unique_sorted)}
            self.ordinal_maps[col] = ordinal_map
            df[col] = df[col].map(ordinal_map)

        return df

    def preprocess_test_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the test dataset by cleaning the dataset and standardizing numerical values of X_ columns."""
        # Clean data by removing unnecessary columns and handling missing values
        df = self.clean_data(df)

        # 1. Identify numerical x_ columns
        x_cols = [col for col in df.columns if col.startswith('x_')]

        # 2. Standardize using the scaler fitted on train data
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Please fit the scaler using training data first.")
        df[x_cols] = self.scaler.transform(df[x_cols])

         # 3. Identify categorical C_ columns
        c_cols = [col for col in df.columns if col.startswith('C_')]

        # 4. Map categorical levels
        for col in c_cols:
            if self.ordinal_maps is None or col not in self.ordinal_maps:
                raise ValueError(f"Ordinal map for {col} not found. Train first!")
            df[col] = df[col].map(self.ordinal_maps[col])
        
        return df
    
    def reset(self):
        """Reset the DataProcessor to its initial state."""
        self.scaler = None
        self.num_imputer = None
        self.cat_imputer = None
        self.ordinal_maps = None
        self.dropped_cols = None