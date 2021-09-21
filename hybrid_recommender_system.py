#####  Data Understanding

# Importing Libraries
import pandas as pd
pd.set_option('display.max_columns', 20)


# Importing Data
movie = pd.read_csv("datasets/movie_lens_dataset/movie.csv")
rating = pd.read_csv("datasets/movie_lens_dataset/rating.csv")

df = movie.merge(rating, how="left", on="movieId")

##### Descriptive Statistics

df.shape  # Dimension of dataframe

df.dtypes  # Data type of each variable

df.info  # Print a concise summary of a DataFrame

df.head()  # First 5 observations of dataframe

df.tail()  # Last 5 observations of dataframe



####################### Item-Based Collaborative Filtering #######################
##### Data Preparation

# Creating the User Movie Df

df["title"].nunique()  # unique number of movies
df["title"].value_counts().head(10)  # how many comments each movie has

rating_counts = pd.DataFrame(df["title"].value_counts())


# I don't use movies with less than 2000 reviews
rare_movies = rating_counts[rating_counts["title"] <= 2000].index

# Movies with over 2000 reviews
comon_movies = df[~df["title"].isin(rare_movies)]

comon_movies.head()
comon_movies.shape

comon_movies["title"].nunique()  # unique number of movies


user_movie_df = comon_movies.pivot_table(index=["userId"], columns=["title"], values="rating")

user_movie_df


# Item-Based Film Recomended

user = 108170
movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]
movie_name = pd.Series(df[df["movieId"] == movie_id][["title"]].values[0])

# Random choosing movies
# movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]
# movie_name = user_movie_df[movie_name]

movies_from_item_based = user_movie_df.corrwith(movie_name).sort_values(ascending=False)

##################################
movies_from_item_based[1:6].index
##################################



####################### User-Based Collaborative Filtering #######################


# Importing Data
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


movie = pd.read_csv("datasets/movie_lens_dataset/movie.csv")
rating = pd.read_csv("datasets/movie_lens_dataset/rating.csv")

df = movie.merge(rating, how="left", on="movieId")

### WordCloud
text = " ".join(word for word in df.title)

wordcloud = WordCloud(max_font_size=50,
                      max_words=50,
                      background_color='black', colormap='Set2').generate(text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud.to_file("wordcloud.png")


def create_user_movie_df(dataframe):
    comment_counts = pd.DataFrame(dataframe["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 2000].index
    comon_movies = dataframe[~dataframe["title"].isin(rare_movies)]
    user_movie_df = comon_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df(df)
user_movie_df.head()


# Random choosing user
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)



# Movies watched by randomly selected user
random_user_df = user_movie_df[user_movie_df.index == random_user]
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

movies_watched[0:5]

len(movies_watched)


# Accessing Data and Ids of Other Users Watching the Same Movies

movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.shape

user_movie_count = movies_watched_df.T.notnull().sum()  # total number of movies watched by each user
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]

users_same_movies = user_movie_count[user_movie_count["movie_count"] > len(movies_watched) * 0.7]["userId"]

users_same_movies.count()


# Identification of Users with the Most Similar Behaviors to the User to be Suggested
# For this we will perform 3 steps:
# 1. We will aggregate the data of the selected User and other users.
# 2. We will create the correlation df.
# 3. We will find the most similar users (Top Users).

final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])
final_df.head()
final_df.shape

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()

# Users with a correlation of 0.7 percent or more with the user we selected

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] > 0.7)][["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values(by='corr', ascending=False)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]


# Weighted Average Recommendation Score's Calculate

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings.head()

# Grouping the score given to each movie
recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

recommendation_df = recommendation_df.reset_index()

# Ranking of values whose weight average is greater than 3.6
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.6].sort_values("weighted_rating", ascending=False)

movies_to_be_recommend_final = movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"].head()

################################
movies_to_be_recommend_final
################################




