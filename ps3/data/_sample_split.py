import hashlib
import pandas as pd
# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation.
def create_sample_split(df, id_column, training_frac=0.8):
    """Create sample split based on ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_column : str
        Name of ID column
    training_frac : float, optional
        Fraction to use for training, by default 0.9

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split based on IDs.
    """
    def hash_id(val):
        str_encoded=str(val).encode()
        hashid= int(hashlib.md5(str_encoded).hexdigest(), 16) %(2**32)
        fraction = hashid/ (2**32)
        return fraction
    hash_values=df[id_column].apply(hash_id)
    df['sample']=hash_values.apply(lambda x: 'train' if x<training_frac else 'test')

    return df