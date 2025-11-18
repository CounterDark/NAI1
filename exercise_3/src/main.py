"""
Movie Recommendation Engine

This script implements a user-based collaborative filtering recommender system
enhanced with user clustering to provide movie recommendations. It fetches
additional movie metadata from the OMDb API.

Compared to previous projects this one is based on functions instead of classes.

API Key for OMDb:
You need to get an API key from https://www.omdbapi.com/apikey.aspx
and set it as an environment variable `OMDB_API_KEY`, or place it
directly in the `OMDB_API_KEY` constant below.

After setting the API key, you can run the script using:
```bash
python src/main.py --user "Paweł Czapiewski"
```
or
```bash
make run ARGS="--user 'Paweł Czapiewski'"
```

Authors: Mateusz Anikiej and Aleksander Kunkowski
"""

import os
import json
import argparse
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd  # type: ignore
import httpx
from sklearn.cluster import KMeans  # type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error  # type: ignore
from sklearn.model_selection import KFold  # type: ignore

# --- Constants ---
OMDB_API_KEY = os.getenv("OMDB_API_KEY", "")
OMDB_API_URL = "http://www.omdbapi.com/"


# --- Function Definitions ---


def load_ratings(path: str) -> dict:
    """Loads ratings.json and returns a dictionary."""
    # Open and read the JSON file with UTF-8 encoding.
    with open(path, "r", encoding="utf-8") as f:
        # Parse the JSON content into a Python dictionary.
        return json.load(f)


def build_rating_matrix(
    ratings: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Builds a user × item matrix and computes per-user mean ratings.
    Returns raw and mean-centered matrices.
    """
    # Convert into a pandas DataFrame
    df = pd.DataFrame.from_dict(ratings, orient="index")
    # Calculate the mean rating for each user
    user_means = df.mean(axis=1)
    # Create a new DataFrame with mean-centered ratings by subtracting
    # the user's mean from each of their ratings
    df_mean_centered = df.subtract(user_means, axis=0)
    return df, df_mean_centered, user_means


def cluster_users(rating_matrix: pd.DataFrame, n_clusters: int) -> tuple[dict, KMeans]:
    """
    Runs clustering (k-means)on user vectors.

    Args:
        rating_matrix: User-item rating matrix (users x items).
        n_clusters: The number of clusters to form.

    Returns:
        - A dictionary mapping user to cluster label.
        - The fitted clustering model.
    """
    # Fill any missing ratings for the clustering algorithm
    data_for_clustering = rating_matrix.fillna(0)
    # Initialize the K-Means clustering model
    model = KMeans(n_clusters=n_clusters, n_init=10)
    # Fit the model and predict the cluster labels for each user
    labels = model.fit_predict(data_for_clustering)
    # Create a dictionary mapping each user to their assigned cluster label
    user_cluster_map = dict(zip(rating_matrix.index, labels))
    # Return the user-cluster map and the fitted model
    return user_cluster_map, model


def compute_user_similarity(
    rating_matrix: pd.DataFrame, metric: str, min_common_items: int = 3
) -> pd.DataFrame:
    """
    Computes pairwise similarity between users based on co-rated items.

    Args:
        rating_matrix: Mean-centered user-item rating matrix.
        metric: Similarity metric ('pearson', 'cosine', or 'euclidean').
        min_common_items: Minimum number of co-rated items to consider a
        similarity valid.

    Returns:
        A DataFrame with pairwise user similarities.
    """
    if metric == "pearson":
        # Use pandas' built-in correlation function for Pearson similarity
        similarity_df = rating_matrix.T.corr(
            method="pearson", min_periods=min_common_items
        )
    elif metric == "cosine":
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

        # Calculate cosine similarity on the matrix
        similarity_matrix = cosine_similarity(rating_matrix.fillna(0))
        # Convert the resulting numpy array back to a DataFrame
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=rating_matrix.index,
            columns=rating_matrix.index,
        )
        # Manually enforce the minimum number of common items for the similarity
        common_items = rating_matrix.notna().astype(int)
        common_count = common_items.dot(common_items.T)
        similarity_df[common_count < min_common_items] = np.nan
    elif metric == "euclidean":
        # Import the euclidean distances function from scikit-learn.
        from sklearn.metrics.pairwise import euclidean_distances  # type: ignore

        # Calculate the euclidean distance between all pairs of users.
        dist_matrix = euclidean_distances(rating_matrix.fillna(0))
        # Convert distance to similarity
        # higher values are better (0 distance -> 1 similarity)
        similarity_matrix = 1 / (1 + dist_matrix)
        # Convert the result back to a DataFrame
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=rating_matrix.index,
            columns=rating_matrix.index,
        )
        # Manually enforce the minimum common items threshold
        common_items = rating_matrix.notna().astype(int)
        common_count = common_items.dot(common_items.T)
        similarity_df[common_count < min_common_items] = np.nan
    else:
        raise ValueError(
            "Unsupported similarity metric. Choose 'pearson', 'cosine', or 'euclidean'."
        )

    # Fill the diagonal of the similarity matrix with NaN to ignore self-similarity
    np.fill_diagonal(similarity_df.values, np.nan)
    return similarity_df


def predict_rating(
    user: str,
    item: str,
    ratings_df: pd.DataFrame,
    ratings_df_mean_centered: pd.DataFrame,
    user_means: pd.Series,
    user_cluster_map: dict,
    similarity_df: pd.DataFrame,
    k_neighbors: int,
    use_cluster_restriction: bool = True,
) -> float:
    """
    Predicts rating for a specific user and item.
    """
    # Get the target user's average rating to use as a baseline or fallback
    user_mean = user_means.get(user, 0)
    # Find all users who have rated the target item
    users_who_rated_item = ratings_df.index[ratings_df[item].notna()]
    # If the user has already rated this item, we don't need to predict it
    if user in users_who_rated_item:
        return np.nan

    # Get the similarity scores between the target user and all other users
    user_similarities = similarity_df[user].drop(user, errors="ignore").dropna()
    # Find users who are similar and have rated the item
    potential_neighbors = user_similarities.index.intersection(users_who_rated_item)

    final_neighbors = potential_neighbors
    # If cluster restriction is enabled, try to narrow down the neighbors
    if use_cluster_restriction:
        # Get the cluster label for the target user
        user_cluster = user_cluster_map.get(user)
        if user_cluster is not None:
            # Find all other users belonging to the same cluster
            cluster_members = {
                u for u, c in user_cluster_map.items() if c == user_cluster
            }
            # Filter the potential neighbors to include only those in the same cluster
            cluster_neighbors = potential_neighbors.intersection(cluster_members)

            # Only use the smaller cluster-based list if it's not empty
            # Otherwise, fall back to the list of neighbors to avoid empty predictions
            if not cluster_neighbors.empty:
                final_neighbors = cluster_neighbors

    # Get the similarity scores for the final list of neighbors
    similar_neighbors = user_similarities.loc[final_neighbors]
    # Select the top K most similar neighbors
    top_k_neighbors = similar_neighbors.nlargest(k_neighbors)
    # If there are no neighbors to use, fall back to the user's average rating
    if top_k_neighbors.empty:
        return user_mean

    # Calculate the weighted average of neighbor ratings
    # Numerator: sum of (similarity * neighbor's mean-centered rating)
    numerator = (
        top_k_neighbors * ratings_df_mean_centered.loc[top_k_neighbors.index, item]
    ).sum()
    # Denominator: sum of the absolute similarity values
    denominator = top_k_neighbors.abs().sum()

    # Avoid division by zero
    if denominator == 0:
        return user_mean

    # The final prediction is the user's mean plus the weighted average adjustment
    prediction = user_mean + (numerator / denominator)
    return prediction


def fetch_movie_metadata(title: str) -> dict:
    """
    Uses the OMDb API to fetch movie metadata.
    """
    if not OMDB_API_KEY:
        print("Warning: OMDb API key is not set. Metadata fetching will be skipped.")
        return {}

    try:
        params = {"t": title, "apikey": OMDB_API_KEY}
        with httpx.Client() as client:
            response = client.get(OMDB_API_URL, params=params)
            response.raise_for_status()
            data = response.json()

        if data.get("Response") == "True":
            metadata = {
                "title": data.get("Title"),
                "year": data.get("Year"),
                "genres": data.get("Genre"),
                "plot": data.get("Plot"),
                "director": data.get("Director"),
                "external_rating": data.get("imdbRating"),
            }
            return metadata
        else:
            return {}
    except httpx.RequestError as e:
        print(f"Error fetching metadata for '{title}': {e}")
        return {}


def enrich_recommendations(
    recommendation_list: list[tuple[str, float]],
) -> list[dict]:
    """
    Given a list of (title, predicted_rating), returns a list of enriched dicts.
    """
    enriched = []
    for title, pred_rating in recommendation_list:
        metadata = fetch_movie_metadata(title)
        entry = {"title": title, "predicted_rating": pred_rating, **metadata}
        enriched.append(entry)
    return enriched


def generate_recommendations(
    user: str,
    ratings_df: pd.DataFrame,
    ratings_df_mean_centered: pd.DataFrame,
    user_means: pd.Series,
    user_cluster_map: dict,
    similarity_df: pd.DataFrame,
    best_params: dict,
    n_pos: int = 5,
    n_neg: int = 5,
) -> tuple[list, list]:
    """
    Computes predicted ratings for all unseen items and returns top/bottom N.
    """
    # Get the set of items the user has already rated
    user_rated_items = set(ratings_df.loc[user].dropna().index)
    # Get the set of all possible items
    all_items = set(ratings_df.columns)
    # Determine the set of items the user has not yet rated
    unseen_items = all_items - user_rated_items

    # Create a dictionary to store predicted ratings for unseen items
    predictions = {}
    # Loop through each unseen item and predict its rating
    for item in unseen_items:
        pred = predict_rating(
            user,
            item,
            ratings_df,
            ratings_df_mean_centered,
            user_means,
            user_cluster_map,
            similarity_df,
            best_params["k_neighbors"],
            best_params["use_cluster_restriction"],
        )
        # If the prediction is a valid number, store it
        if pd.notna(pred):
            predictions[item] = pred

    # Sort the predictions from highest to lowest rating
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    # Get the top N predictions as positive recommendations
    positive_recs = sorted_predictions[:n_pos]
    # Get the bottom N predictions as negative recommendations
    negative_recs = sorted_predictions[-n_neg:]
    # Return both lists
    return positive_recs, negative_recs


def recommend_for_user(
    user: str,
    ratings_df: pd.DataFrame,
    best_params: dict,
    n_pos: int = 5,
    n_neg: int = 5,
):
    """
    High-level function that runs the whole pipeline for a user.
    """
    # Train the final model using the full dataset and the best parameters found
    print("\n--- Training final model with best parameters ---")
    # First, build the rating matrices from the raw ratings data
    ratings, ratings_mean_centered, user_means = build_rating_matrix(
        ratings_df.to_dict(orient="index")
    )

    # Cluster all users based on the best number of clusters found
    user_cluster_map, _ = cluster_users(
        ratings_mean_centered,
        best_params["n_clusters"],
    )

    # Compute the final user-similarity matrix
    similarity_df = compute_user_similarity(
        ratings_mean_centered,
        best_params["similarity_metric"],
        best_params["min_common_items"],
    )

    # Generate the positive and negative recommendations for the user
    print(f"\n--- Generating recommendations for {user} ---")
    pos_recs, neg_recs = generate_recommendations(
        user,
        ratings,
        ratings_mean_centered,
        user_means,
        user_cluster_map,
        similarity_df,
        best_params,
        n_pos,
        n_neg,
    )

    print("Enriching positive recommendations...")
    enriched_pos = enrich_recommendations(pos_recs)
    print("Enriching negative recommendations...")
    enriched_neg = enrich_recommendations(neg_recs)

    return {
        "user": user,
        "positive": enriched_pos,
        "negative": enriched_neg,
    }


def tune_parameters(ratings_df: pd.DataFrame):
    """
    Finds the best parameters using k-fold cross-validation.
    """
    # Parameters to search through.
    param_grid: dict[str, list] = {
        # number of clusters to form
        "n_clusters": [2, 3, 4, 5],
        # similarity metric to use
        "similarity_metric": ["pearson", "cosine", "euclidean"],
        # number of neighbors to consider for prediction
        "k_neighbors": [3, 5, 7, 10],
        # minimum number of common items to consider for similarity
        "min_common_items": [2, 3],
        # whether to use cluster restriction for prediction
        "use_cluster_restriction": [True, False],
    }

    # Generate all combinations of parameters
    param_combinations = list(product(*param_grid.values()))

    print(f"Starting parameter tuning with {len(param_combinations)} combinations...")
    results = []

    # Get all existing ratings as a flat list of (user, item, rating) tuples
    all_ratings = []
    for user in ratings_df.index:
        for item in ratings_df.columns:
            if pd.notna(ratings_df.loc[user, item]):
                all_ratings.append((user, item, ratings_df.loc[user, item]))

    # Set up 5-fold cross-validation to split the data
    kf = KFold(n_splits=5, shuffle=True)

    # Loop through each combination of parameters
    for i, params_tuple in enumerate(param_combinations):
        # Create a dictionary for the current set of parameters
        params = dict(zip(param_grid.keys(), params_tuple))

        # Store the error scores for each of the 5 folds
        # RMSE - Root Mean Squared Error
        # MAE - Mean Absolute Error
        fold_rmse = []
        fold_mae = []

        # Split the data into training and validation sets for the current fold
        for train_index, val_index in kf.split(all_ratings):
            # Separate the rating tuples into training and validation sets
            train_set = [all_ratings[i] for i in train_index]
            val_set = [all_ratings[i] for i in val_index]

            # Reconstruct a training DataFrame from the training set tuples
            train_ratings: defaultdict[str, dict[str, float]] = defaultdict(dict)
            for name, title, rating in train_set:
                train_ratings[name][title] = rating

            # Align the new training DataFrame with the original's structure
            train_df = pd.DataFrame.from_dict(train_ratings, orient="index")
            train_df = train_df.reindex(
                index=ratings_df.index, columns=ratings_df.columns
            )

            # Get the true ratings from the validation set for error calculation
            val_true = [r for _, _, r in val_set]
            # This list will store the predictions for the validation set
            val_preds = []

            # Train the model components on the current training fold
            _, train_df_mean_centered, train_user_means = build_rating_matrix(
                train_ratings
            )

            # Reindex the trained components to maintain consistent shape
            train_df_mean_centered = train_df_mean_centered.reindex(
                index=ratings_df.index, columns=ratings_df.columns
            )
            train_user_means = train_user_means.reindex(index=ratings_df.index)

            # Cluster users and compute similarities using the training data
            user_cluster_map, _ = cluster_users(
                train_df_mean_centered,
                params["n_clusters"],
            )
            similarity_df = compute_user_similarity(
                train_df_mean_centered,
                params["similarity_metric"],
                params["min_common_items"],
            )

            # Predict the rating for each item in the validation set
            for user, item, _ in val_set:
                pred = predict_rating(
                    user,
                    item,
                    train_df,
                    train_df_mean_centered,
                    train_user_means,
                    user_cluster_map,
                    similarity_df,
                    params["k_neighbors"],
                    params["use_cluster_restriction"],
                )
                if pd.notna(pred):
                    # Append the valid prediction
                    val_preds.append(pred)
                else:
                    # If prediction is not possible, fall back to the average rating
                    val_preds.append(
                        train_user_means.get(user, ratings_df.mean().mean())
                    )

            # If we have predictions, calculate the error for this fold
            if len(val_preds) == len(val_true):
                fold_rmse.append(np.sqrt(mean_squared_error(val_true, val_preds)))
                fold_mae.append(mean_absolute_error(val_true, val_preds))

        # After all folds, if we have results, calculate the average error
        if fold_rmse:
            avg_rmse = np.mean(fold_rmse)
            avg_mae = np.mean(fold_mae)
            # Store the parameters and their average performance
            results.append({"params": params, "rmse": avg_rmse, "mae": avg_mae})
            progress_msg = "Finished combination {}/{}. RMSE: {:.4f}".format(
                i + 1, len(param_combinations), avg_rmse
            )
            print(progress_msg)

    # Parameters that resulted in the lowest RMSE
    best_params = min(results, key=lambda x: x["rmse"])

    print("\n--- Parameter Tuning Complete ---")
    print(f"Best RMSE: {best_params['rmse']:.4f}")
    print(f"Best MAE: {best_params['mae']:.4f}")
    print("Best parameters:")
    print(json.dumps(best_params["params"], indent=4))
    return best_params["params"]


if __name__ == "__main__":
    # Set up argument parser to accept a user name from the command line.
    parser = argparse.ArgumentParser(description="Movie Recommendation Engine")
    parser.add_argument(
        "--user",
        type=str,
        required=True,
        help="The name of the user to recommend movies for.",
        default="Paweł Czapiewski",
    )
    args = parser.parse_args()
    user = args.user

    # Load data from the JSON file
    ratings_dict = load_ratings("src/ratings.json")

    # Check if the user exists in the dataset# Check if the user exists in the dataset
    if user not in ratings_dict.keys():
        print(f"User '{user}' not found in the dataset.")
        exit(1)

    # Build the main ratings DataFrame
    ratings_df, _, _ = build_rating_matrix(ratings_dict)

    # Run the parameter tuning process to find the best model configuration
    best_params = tune_parameters(ratings_df)

    # Generate and enrich recommendations for the user
    results = recommend_for_user(user, ratings_df, best_params)

    # Print the final recommendations
    print("\n\n--- Final Recommendations ---")
    print(json.dumps(results, indent=4))
