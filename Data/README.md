# 🗄️ Data

This directory serves as the primary data lake for the KBA Credit Risk & Alternative Data AutoML Engine.

## Subdirectories

- **Alternative_Data**: Stores datasets related to non-traditional credit risk signals, such as KNBS macroeconomic indicators, M-Pesa mobile money velocity, and geospatial telemetry (e.g., rainfall, crop health).
- **Loan_Data**: Stores historical loan book data, repayment performance, and traditional credit bureau (CRB) features.
- **Simulated_Data**: Contains synthetic datasets generated for testing, benchmarking, and algorithm validation.
- **WareHouse**: Serves as the persistence layer for DuckDB instances, analytical feature stores (`.parquet`), and processed datasets ready for model training.

*Note: All data processing strictly adheres to the Zero-PII Disk Persistence architecture, ensuring ephemeral and secure handling of sensitive information.*
