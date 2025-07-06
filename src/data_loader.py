# data_loader.py
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_data_in_chunks(file_path, chunk_size=10000):
    """
    Loads data from a CSV file in chunks.
    Args:
        file_path (str): The path to the CSV file.
        chunk_size (int): The number of rows per chunk.
    Returns:
        pandas.io.parsers.TextFileReader: An iterator for the DataFrame chunks.
    """
    try:
        return pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)
    except FileNotFoundError:
        logger.error(f"Error: The file {file_path} was not found.")
        raise
    except Exception as e:
        logger.error(f"Error loading data in chunks from {file_path}: {e}")
        raise


def filter_and_map_products(df):
    """
    Filters and maps raw CFPB product names to the five target chatbot
    product categories.

    Args:
        df (pd.DataFrame): The input DataFrame containing 'Product' and
            'Consumer complaint narrative' columns.

    Returns:
        pd.DataFrame: The DataFrame with a new 'Mapped_Product' column and
            filtered to target products.
    """
    # Define the comprehensive mapping from raw CFPB product names to your
    # target categories
    product_mapping = {
        'Credit card or prepaid card': 'Credit Cards',
        'Payday loan, title loan, or personal loan': 'Personal Loans',
        'Money transfer, virtual currency, or money service':
            'Money Transfers',
        'Checking or savings account': 'Savings Accounts',
        'Mortgage': 'Mortgages',  # Keeping this as its a major category in the
        # raw data
        'Debt collection': 'Debt collection',  # Keeping this too
        (
            'Credit reporting, credit repair services, or other personal '
            'consumer reports'
        ): (
            'Credit reporting, credit repair services, or other personal '
            'consumer reports'
        ),  # Keeping this
    }

    # Apply the direct mapping
    if 'Product' in df.columns:
        df['Mapped_Product'] = df['Product'].map(product_mapping)
    else:
        logger.warning("No 'Product' column found for mapping.")
        df['Mapped_Product'] = None  # Or handle as appropriate

    # --- BNPL Identification Logic ---
    # Since BNPL might not be a direct 'Product' category,
    # infer it from narratives,
    # This is a simple keyword-based approach;
    # a more robust solution might use NLP models
    bnpl_keywords = [
        'buy now pay later', 'bnpl', 'klarna', 'afterpay', 'affirm',
        'sezzle', 'quadpay'
    ]

    # Check if 'Consumer complaint narrative' column exists
    if 'Consumer complaint narrative' in df.columns:
        # Identify BNPL complaints where
        # 'Mapped_Product' is still NaN or not a core target
        # and narrative contains BNPL keywords.
        # Ensure we only process non-null strings in the narrative
        narrative_col = 'Consumer complaint narrative'
        bnpl_mask = (
            df[narrative_col].astype(str).str.lower().str.contains(
                '|'.join(bnpl_keywords), na=False
            ) &
            (df['Mapped_Product'].isna() | ~df['Mapped_Product'].isin([
                'Credit Cards', 'Personal Loans', 'Savings Accounts',
                'Money Transfers'
            ]))
        )
        df.loc[bnpl_mask, 'Mapped_Product'] = 'Buy Now, Pay Later (BNPL)'
    else:
        logger.warning(
            "No 'Consumer complaint narrative' column found for BNPL "
            "inference."
        )

    # Define your final target chatbot product categories
    target_chatbot_products = [
        'Credit Cards',
        'Personal Loans',
        'Buy Now, Pay Later (BNPL)',
        'Savings Accounts',
        'Money Transfers'
    ]

    # Filter the DataFrame to include only the target chatbot products
    if 'Mapped_Product' in df.columns:
        initial_rows = len(df)
        df_filtered = df[
            df['Mapped_Product'].isin(target_chatbot_products)
        ].copy()
        logger.info(f"Filtered from {initial_rows} rows.")
        logger.info(
            f"To {len(df_filtered)} rows for target chatbot products."
        )
        return df_filtered
    else:
        logger.warning("No 'Mapped_Product' column to filter on.")
        logger.warning("Returning original DataFrame.")
        return df
