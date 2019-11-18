#Goal: Create an average weighted recommender system based from movies and credits
#We are going to consider the voting average and vote count to determine recommendations
#%%
#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

#%%

credits= pd.read_csv("tmdb_5000_credits.csv")
#%%
movies_df= pd.read_csv("tmdb_5000_movies.csv")
#%%
print(display(credits.head()))

#%%
print(display(movies_df.head()))

#%%
print("Credits: ", credits.shape)
print("Movies Dataframe: ", movies_df.shape)

#%%
#So what we want to do now is merge the movie and credits datasets id columns

credits_column_renamed= credits.rename(index=str, columns={"movie_id": "id"})
#%%
movies_df_merge= movies_df.merge(credits_column_renamed, on='id')

#%%
print(display(movies_df_merge.head()))

#%%
#Now we need to drop the irrelevant features that will not help with algorithm
movies_cleaned_df= movies_df_merge.drop(columns=['homepage', 'title_x', 'title_y', 'status', 'production_countries'])

#%%
print(display(movies_cleaned_df.head()))

#%%
movies_cleaned_df.info()

#%%
#We need to calculate the components to do the weighted average formula
v= movies_cleaned_df['vote_count']
R= movies_cleaned_df['vote_average']
C= movies_cleaned_df['vote_average'].mean()
m= movies_cleaned_df['vote_count'].quantile(0.70)
#for quantile we're only considering movies that are more than 70th percentile or higher

#%%
#Now we can apply the formula to find out the weights
movies_cleaned_df['weighted_average']= ((R*v)+(C*m))/(v+m)
print(display(movies_cleaned_df.head()))

#%%
print(movies_cleaned_df.shape)
#%%
#So now let's sort the movies by their weighted average and popularity
movie_sorted_ranking= movies_cleaned_df.sort_values('weighted_average', ascending=False)
print(display(movie_sorted_ranking[['original_title', 'vote_count', 'vote_average', 'weighted_average', 'popularity']].head(20)))

#%%
#Now let's plot the weighted average results.
weight_average= movie_sorted_ranking.sort_values('weighted_average', ascending=False)
plt.figure(figsize=(12,6))
axis1=sns.barplot(x=weight_average['weighted_average'].head(10), y=weight_average['original_title'].head(10))
plt.xlim(4,10)
plt.title('Best Movies by average votes', weight='bold')
plt.xlabel('Weighted Average Score', weight='bold')
plt.ylabel('Movie Title', weight='bold')
plt.savefig('best_movies.png')

#%%
#We can also plot the data based on the most popular movies
popularity=movie_sorted_ranking.sort_values('popularity', ascending=False)
plt.figure(figsize=(12,6))
ax=sns.barplot(x=popularity['popularity'].head(10), y=popularity['original_title'].head(10), data=popularity)
plt.title('Most Popular by Votes', weight='bold')
plt.xlabel('Score of Popularity', weight='bold')
plt.ylabel('Movie Title', weight='bold')
plt.savefig('best_popular_movies.png')
#%%
#So Now we can build our recommendation system based on scaled weighted avg and popularity score
#Both popularity and weighted average will get 50% importance in the system
scaling=MinMaxScaler()#use this function because the values between pop. and weight avg are way too different
movie_scaled_df= scaling.fit_transform(movies_cleaned_df[['weighted_average', 'popularity']])
movie_normalized_df=pd.DataFrame(movie_scaled_df, columns=['weighted_average','popularity'])
print(display(movie_normalized_df.head()))
#%%
#Now let's create the columns that will be displayed that show recommended
#movies based on the two scoring criteria
movies_cleaned_df[['normalized_weight_average', 'normalized_popularity']]= movie_normalized_df
print(display(movies_cleaned_df.head()))
#%%
#Let's now make the equation that will give equal priority to normalized
#weighted average and to normalized popularity
movies_cleaned_df['score']=movies_cleaned_df['normalized_weight_average']*0.5+movies_cleaned_df['normalized_popularity']*0.5
movies_scored_df= movies_cleaned_df.sort_values(['score'], ascending=False)
movies_scored_df[['original_title', 'normalized_weight_average', 'normalized_popularity', 'score']].head(20)
#%%
#Now we have a nice list of the top 20 movies that would be 
#recommended by both vote score and popularity
#Let's finish off by plotting the data with the top 10
scored_df= movies_scored_df.sort_values('score', ascending=False)
plt.figure(figsize=(16,6))
ax=sns.barplot(x=scored_df['score'].head(10), y=scored_df['original_title'].head(10), data=scored_df, palette='deep')
plt.title('Best Rated and Most Popular Movie Blend', weight='bold')
plt.xlabel('Score', weight='bold')
plt.ylabel('Movie Title', weight='bold')
plt.savefig('scored_movies.png')

