
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import svds
import gradio as gr


data_path = "/kaggle/input/movielens-100k-dataset/ml-100k/"
 

ratings = pd.read_csv(data_path + "u.data",
                      sep='	',
                      names=["user_id", "movie_id", "rating", "timestamp"])

movie_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL']
movies = pd.read_csv(data_path + "u.item",
                     sep='|',
                     names=movie_cols,
                     encoding='latin-1',
                     usecols=range(5))

df = pd.merge(ratings, movies[['movie_id', 'title']], on='movie_id')
df = df.drop(columns=['timestamp'])


train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)


user_item_matrix = train_ratings.pivot_table(index='user_id',
                                             columns='movie_id',
                                             values='rating').fillna(0)


user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity,
                                  index=user_item_matrix.index,
                                  columns=user_item_matrix.index)

def recommend_user_based(user_id, user_item_matrix, user_similarity_df, movies, N=5, return_scores=False):
    if user_id not in user_item_matrix.index:
        return pd.DataFrame(columns=['movie_id','title'])

    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:]
    most_similar_user = similar_users.index[0]

    user_ratings = user_item_matrix.loc[user_id]
    similar_user_ratings = user_item_matrix.loc[most_similar_user]

    unrated_movies = user_ratings[user_ratings == 0].index
    recommendation = similar_user_ratings.loc[unrated_movies].sort_values(ascending=False).head(N)

    recommendation_df = movies.set_index('movie_id').loc[recommendation.index][['title']].reset_index()
    recommendation_df['movie_id'] = recommendation.index
    if return_scores:
        recommendation_df['score'] = recommendation.values
    return recommendation_df


U, s, Vt = svds(user_item_matrix.values, k=50)
S = np.diag(s)

U_df = pd.DataFrame(U, index=user_item_matrix.index)
Vt_df = pd.DataFrame(Vt, columns=user_item_matrix.columns)

def recommend_svd(user_id, user_item_matrix, movies, U_df, S, Vt_df, N=5, return_scores=False):
    if user_id not in user_item_matrix.index:
        return pd.DataFrame(columns=['movie_id','title'])

    user_idx = user_item_matrix.index.get_loc(user_id)
    preds = pd.Series(U_df.iloc[user_idx, :] @ S @ Vt_df.values, index=user_item_matrix.columns)

    user_ratings = user_item_matrix.loc[user_id]
    unrated_mask = user_ratings == 0
    candidates = preds[unrated_mask].sort_values(ascending=False).head(N)

    recs_df = movies.set_index('movie_id').loc[candidates.index][['title']].reset_index()
    recs_df['movie_id'] = candidates.index
    if return_scores:
        recs_df['score'] = candidates.values
    return recs_df


def recommend(user_id, method="UserCF"):
    try:
        if method == "UserCF":
            recs = recommend_user_based(int(user_id), user_item_matrix, user_similarity_df, movies, N=5)
        else:
            recs = recommend_svd(int(user_id), user_item_matrix, movies, U_df, S, Vt_df, N=5)
        return recs[['title']].to_string(index=False)
    except Exception as e:
        return f"Error: {str(e)}"

iface = gr.Interface(
    fn=recommend,
    inputs=[gr.Number(label="User ID"), gr.Radio(["UserCF", "SVD"])],
    outputs="text",
    title="Movie Recommendation System"
)

if __name__ == "__main__":
    iface.launch()
