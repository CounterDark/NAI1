# Movie recommendation engine

The program tries to find 5 movies the given user will probably like and 5 movies they will probably dislike from a list of movies they haven't seen yet.

Authors: Mateusz Anikiej and Aleksander Kunkowski

## Prerequisites

1. Python 3.13 installed (ensure `python --version` shows 3.13+)
2. `uv` installed (https://docs.astral.sh/uv/getting-started/installation/)
   - Linux/macOS (curl): `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

## Setup

Use `uv` to create a virtual environment and install dependencies from `pyproject.toml`.

Option A: using Makefile

```sh
make setup
```

Option B: using uv directly

```sh
uv sync
```

## Running the program

### API Key for OMDb

You need to get an API key from https://www.omdbapi.com/apikey.aspx
and set it as an environment variable `OMDB_API_KEY`, or place it
directly in the `OMDB_API_KEY` constant below.

### After setting the API key, you can run the script using

```bash
python src/main.py --user "Paweł Czapiewski"
```

or

```bash
make run ARGS="--user 'Paweł Czapiewski'"
```

## How it works

### Step 1: Understanding user's Tastes

First, the program reads the `ratings.json` file. It organizes this data into a DataFrame where:

- Each row represents a user.
- Each column represents a movie.
- The cells contain the rating a user gave to a movie.
- Many cells are empty, which simply means a user hasn't rated that particular movie. This table gives a complete picture of who has rated what.

### Step 2: Finding the Best Recommendation Strategy

1. The program first tries out different strategies to find the most accurate one for this specific dataset.
2. Grouping Similar Users (Clustering): The program groups users with similar tastes into clusters.
3. Measuring Similarity: It calculates a "similarity score" between users. A high score means two users have very similar tastes.
4. Choosing "Taste Buddies" (Neighbors): When predicting a rating, it decides how many of the most similar users (or "taste buddies") to listen to. It tries using 3, 5, 7, or 10 buddies.
5. To test which of the strategies is best, for each strategy, program repeatedly hides a few known ratings, asks the strategy to predict them, and checks how wrong the predictions were. The strategy that consistently makes the least amount of error is declared the winner.

### Step 3: Predicting a Rating

Once the best strategy is chosen, the program is ready to make predictions.
It finds all the users who have rated given movie.
From that group, it identifies user's closest "taste buddies" based on the winning similarity score. It focuses especially on buddies from his own "taste cluster."
It looks at how those buddies rated the movie. The ratings from more similar buddies are given more weight.
It calculates a weighted average of their ratings to produce a final predicted score for user.

### Step 4: Finding the Top 5 and Bottom 5 Movies

The program repeats Step 3 for every single movie that user hasn't rated yet.
The 5 movies with the highest predicted scores become the "positive recommendations."
The 5 movies with the lowest predicted scores become the "negative recommendations."

### Step 5: Getting Movie Details

To make the recommendations more useful, the program takes the title of each recommended movie and looks it up on the Open Movie Database (OMDb). This information is added to the final output.
