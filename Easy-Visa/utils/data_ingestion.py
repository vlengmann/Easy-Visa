import os
print("Running:", os.path.abspath(__file__))
import zipfile
import pandas as pd
from abc import ABC, abstractmethod

# ---------------------------------------------------------
# Abstract Base Class
# ---------------------------------------------------------
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Abstract method to ingest data from a given file."""
        pass

# ---------------------------------------------------------
# CSV Ingestor
# ---------------------------------------------------------
class CSVDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        if not file_path.endswith(".csv"):
            raise ValueError("The provided file is not a .csv file.")
        return pd.read_csv(file_path)

# ---------------------------------------------------------
# ZIP Ingestor
# ---------------------------------------------------------
class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        if not file_path.endswith(".zip"):
            raise ValueError("The provided file is not a .zip file.")

        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall("extracted_data")

        extracted_files = os.listdir("extracted_data")
        csv_files = [f for f in extracted_files if f.endswith(".csv")]

        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV file found in the extracted data.")
        if len(csv_files) > 1:
            raise ValueError("Multiple CSV files found. Please specify which one to use.")

        csv_path = os.path.join("extracted_data", csv_files[0])
        return pd.read_csv(csv_path)

# ---------------------------------------------------------
# Factory
# ---------------------------------------------------------
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        if file_extension == ".csv":
            return CSVDataIngestor()
        if file_extension == ".zip":
            return ZipDataIngestor()
        raise ValueError(f"No ingestor available for file extension: {file_extension}")