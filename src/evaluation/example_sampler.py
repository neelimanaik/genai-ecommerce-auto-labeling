import pandas as pd

def create_examples(examples_df, n=1):

    """
    Return a JSON list of randomized examples of size 2n with two classes.
    Create subsets of each class, choose random samples from the subsets,
    merge and randomize the order of samples in the merged list.
    Each run of this function creates a different random sample of examples
    chosen from the training data.

    Args:
        dataset (DataFrame): A DataFrame with examples (text + label)
        n (int): number of examples of each class to be selected

    Output:
        randomized_examples (JSON): A JSON with examples in random order
    """

    hc_reviews = (examples_df.Category == 'Hair Care')
    sc_reviews = (examples_df.Category == 'Skin Care')

    cols_to_select = ["Product Description","Category"]
    hc_examples = examples_df.loc[hc_reviews, cols_to_select].sample(n)
    sc_examples = examples_df.loc[sc_reviews, cols_to_select].sample(n)

    examples = pd.concat([hc_examples,sc_examples])
    # sampling without replacement is equivalent to random shuffling
    randomized_examples = examples.sample(2*n, replace=False)

    return randomized_examples.to_json(orient='records')