
#-------------------------------------------------------------------------
#---- downloading the data and creating 'edx' and 'validation' sets
#-------------------------------------------------------------------------
# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

# Downloading the data set:
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

# Extracting data from the downloaded file:
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind = "Rounding" )
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#-------------------------------------------------------------------------
#---- Summary of data
#-------------------------------------------------------------------------

head(edx)
dim(edx)
dim(validation)

#-------------------------------------------------------------------------
#---- Adding "year" column to the 'edx' and 'validation' sets
#-------------------------------------------------------------------------

benchmark = as.Date('1970-1-1')
edx <- edx %>%
  mutate(year = format(as.Date(timestamp/86400,benchmark),format = "%Y"))
validation <- validation %>%
  mutate(year = format(as.Date(timestamp/86400,benchmark),format = "%Y"))
rm(benchmark)
dim(edx)
dim(validation)

#-------------------------------------------------------------------------
#---- Splitting 'edx' to train_set and test_set
#-------------------------------------------------------------------------
set.seed(1, sample.kind = "Rounding" )
index <- createDataPartition(edx$rating, 1, p = 0.1, list = F)
train_set <- edx[-index,]
temp <- edx[index,]

test_set <- temp %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

rm(index, temp, removed)

#-------------------------------------------------------------------------
#---- 1st try : Creating bias terms by just averaging the data
#-------------------------------------------------------------------------

# Calculating average rating:
mu <- mean(train_set$rating)

# Movie effect Calculation:
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# Calculating User effects:
user_avgs <- train_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Calculating Genres effects:
genre_effect <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u))

genre_effect[1,2] <- 0   #deleting the effect of "(no genres listed)"

# Calculating year effect:
year_effect <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_effect, by='genres') %>%
  group_by(year) %>%
  summarize(b_y = mean(rating - mu - b_i - b_u - b_g))

#--------------------
#making predictions:
predicted_ratings <- test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_effect, by='genres') %>%
  left_join(year_effect, by= 'year') %>%
  mutate(pred = mu + b_i + b_u + b_g + b_y) %>% .$pred

try1_rmse <- RMSE(predicted_ratings, test_set$rating, na.rm = T)
results <- data.frame(Method = 'simple average', RMSE = try1_rmse)
results

#-------------------------------------------------------------------------
#---- 2nd try : regularizing bias terms with Lambda
#-------------------------------------------------------------------------

regularize <- function(l){
  mu <- mean(train_set$rating)
  
  # Movie effect Calculation
  movie_avgs <- train_set %>%
    group_by(movieId) %>% 
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  # User effects Calculation
  user_avgs <- train_set %>%
    left_join(movie_avgs, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i)/(n()+l))
  
  # Genres effects Calculation
  genre_effect <- train_set %>% 
    left_join(movie_avgs, by='movieId') %>%
    left_join(user_avgs, by='userId') %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - b_i - b_u)/(n()+l))
  
  genre_effect[1,2] <- 0   #deleting the effect of "(no genres listed)"
  
  # year effect Calculation
  year_effect <- train_set %>%
    left_join(movie_avgs, by='movieId') %>%
    left_join(user_avgs, by='userId') %>%
    left_join(genre_effect, by='genres') %>%
    group_by(year) %>%
    summarize(b_y = sum(rating - mu - b_i - b_u - b_g)/(n()+l))
  
  # Prediction and RMSE
  predicted_ratings <- test_set %>%
    left_join(movie_avgs, by = 'movieId') %>%
    left_join(user_avgs, by = 'userId') %>%
    left_join(genre_effect, by = 'genres') %>%
    left_join(year_effect, by = 'year') %>%
    mutate(pred = mu + b_i + b_u + b_g + b_y) %>% .$pred
  
  return(RMSE(predicted_ratings, test_set$rating, na.rm = T))
}


#Optimizing lambda:
options(digits = 12)
l <- c()              #due to memory allocation limits on my PC
l[1] <- regularize(1)
l[2] <- regularize(2)
l[3] <- regularize(3)
l[4] <- regularize(4)
l[5] <- regularize(5)
l[6] <- regularize(6)
l[7] <- regularize(7)
l[8] <- regularize(8)
l[9] <- regularize(9)
which.min(l)
min(l)

# checking for more improvements in lambda:
options(digits = 10)
regularize(4.8)
regularize(5.2)
regularize(4.9)

# << lambda = 5.0 is optimized >>

results <- bind_rows(results,data_frame(Method = "Regularized", RMSE = l[5] ))
results


#-------------------------------------------------------------------------
#---- Final Analysis on "edx" set + Calculating RMSE on "validation" set
#-------------------------------------------------------------------------

mu <- mean(edx$rating)
l <- 5.0

movie_avgs <- edx %>%                     # Movie effect Calculation
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+l))

user_avgs <- edx %>%                      # User effects Calculation
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i)/(n()+l))

genre_effect <- edx %>%                   # Genres effects Calculation 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - b_i - b_u)/(n()+l))

genre_effect[1,2] <- 0   #deleting the effect of "(no genres listed)":

year_effect <- edx %>%                    # year effect Calculation
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_effect, by='genres') %>%
  group_by(year) %>%
  summarize(b_y = sum(rating - mu - b_i - b_u - b_g)/(n()+l))

predicted_ratings <- validation %>%               # Prediction and RMSE
  left_join(movie_avgs, by = 'movieId') %>%
  left_join(user_avgs, by = 'userId') %>%
  left_join(genre_effect, by = 'genres') %>%
  left_join(year_effect, by = 'year') %>%
  mutate(pred = mu + b_i + b_u + b_g + b_y) %>% .$pred



#Final RMSE:
final_RMSE <- RMSE(predicted_ratings, validation$rating, na.rm = T)
print("The final RMSE:")
final_RMSE
