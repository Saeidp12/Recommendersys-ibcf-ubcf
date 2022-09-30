library(tidyverse) # metapackage of all tidyverse packages
library(LaF)
library(lubridate)
library(chunked)
library(cluster)
library(nomclust)
library(recommenderlab)
library(caret)
library(lsa)

# the path here is on Kaggle database.
path <- "../input/h-and-m-personalized-fashion-recommendations"

# reading articles
# article ids start with 0. But R reads them as numeric and discards the 0. 
# solution grabbed from stackoverflow
# we define a new class of characters that whenever the console identifies a numeric
# value that starts with zero, it reads the value as a character
setClass("character0")

setAs("character", "character0",
      function(from) {
        from2 <- type.convert(from, as.is = TRUE)
        if (is.numeric(from2) && any(grepl("^0", from))) from else from2
      })


articles_df <- read.csv(paste0(path,'/articles.csv'), header=TRUE, colClasses='character0')

# the same thing happens with customer ids and we use the character0 class here too.
customers_df <- read.csv(paste0(path,'/customers.csv'), header=TRUE, colClasses='character0')

# I use LaF to read data chunk by chunk and then select those rows that are after 2020-Aug-26 (last 4 weeks of data)
transacs_file <- paste0(path, '/transactions_train.csv')
model <- detect_dm_csv(transacs_file, sep=",", header=TRUE, colClasses = c('character','character0','character0','numeric','numeric'))

df.laf <- laf_open(model)
nrow(df.laf)/10
nrow(df.laf)/100
transactions_df <- df.laf[1,][0,]
for(i in 1:10){
  raw <- next_block(df.laf,nrows=(nrow(df.laf)/10)) %>% 
    filter(t_dat >= '2020-08-26')
  transactions_df <- rbind(transactions_df, raw)
}

# turn date variables from character into date type. Create a week before variable
# to distinguish the weeks of data for further splitting. 
# last week would be week_before=1, and second to last week would be week_before=2
transacs_df <- transactions_df %>% mutate(t_dat = ymd(t_dat), week_before = max(week(t_dat)) - week(t_dat) + 1)
df <- transacs_df %>% filter(week_before == 2)
df_test <- transacs_df %>% filter(week_before == 1)

# In articles_df, some columns are just names and codes of the same characteristic so we select one of them
art_df <- articles_df %>% select(article_id, product_code, product_type_no, product_group_name, 
                                 graphical_appearance_no,colour_group_code, department_no, 
                                 index_code, index_group_no, section_no, garment_group_no) %>%
  filter(article_id %in% df$article_id)

# we have some missing values in age, we set those missing values equal to median(age)
df <- left_join(df, customers_df[,c(1,6, 7)], by='customer_id')
median_age <- median(df$age, na.rm=TRUE)
df$age[is.na(df$age)] <- median_age

# we create a new variable, mean age of customers who buy a specific article. We find
# this mean for each of the articles and then add categorized version of this variable 
# to our articles data (item-based feature)mean_age_per_article <- df %>% group_by(article_id) %>% summarize(mean_age = mean(age)) 
art_df <- left_join(art_df, mean_age_per_article, by='article_id') %>% 
  mutate(age_group_per_article = cut(mean_age, breaks=c(15, 22.5, 30.5, 38, 45.5, 56, 100),
                                     labels=c(1, 2, 3, 4, 5, 6))) %>%
  select(-mean_age)

# Create a new variable based on the average sales channel of each article (item-based feature)
mean_sales_channel <- df %>% group_by(article_id) %>% summarize(mean_sales_channel = round(mean(sales_channel_id))) %>% ungroup()
art_df <- left_join(art_df, mean_sales_channel, by='article_id')

# calculate the days passed after last purchase of each customer (user-based feature)
max_date <- max(df$t_dat)
days_since_last_purchase <- df %>% group_by(customer_id) %>% summarize(days_since_last_purchase = as.numeric(max_date - max(t_dat)))

# add the categorized version of the above variable to the customers data
cust_df <- left_join(customers_df, days_since_last_purchase, by='customer_id') %>% filter(customer_id %in% df$customer_id)
cust_df <- cust_df %>% mutate(days_since_last_purchase = cut(days_since_last_purchase,
                                                             breaks = c(-1, 1.5, 4, 10, 17, 23, 31, Inf),
                                                             labels = c('1 day', 'around 3 days', 'around a week', 'around 2 weeks',
                                                                        'around 3 weeks', 'around a month', 'more than a month')))

# create a new diff function to deal with NAs when using R diff function
new_diff <- function(x){
  if(length(diff(na.omit(unique(x)))) == 0 ){return(0)}
  else {return(diff(na.omit(unique(x))))}
}
avg_diff <- df %>% group_by(customer_id) %>% summarize(avg_days_diff = round(mean(as.numeric(new_diff(t_dat)))))

# average days difference between purchases for each customer (user-based feature)
cust_df <- left_join(cust_df, avg_diff, by='customer_id')
cust_df <- cust_df %>% mutate(avg_days_diff = cut(avg_days_diff,
                                                  breaks=c(-1, 0.9, 1.5, 3.5, 7.5, 14.5, 21.5, 30.5), labels=c(0, 1, 3, 7, 14, 21, 30)))

# normalize the price feature and categorize it into a new variable for average 
# price spent (user-based feature)df2 <- df %>% mutate(price = (price-mean(price))/sd(price))
df2 <- df2 %>% group_by(customer_id) %>% mutate(price = mean(price)) %>% ungroup()
cust_df <- left_join(cust_df, df2[,c(2, 4)], by='customer_id')
cust_df <- cust_df  %>% mutate(avg_price_spent = cut(price,
                                                     breaks = c(-Inf, -2, -0.5, 0.5, 2, Inf), labels = c('Very less than avg', 'less than avg', 'around avg',
                                                                                                         'more than avg', 'a lot more than avg'))) %>%
  select(-price)
# here cust_df makes duplicate rows. why???

# categorize customers into bins of ages.
cust_df <- cust_df %>% mutate(customer_age_groups = cut(age,
                                                        breaks = c(15,21.9, 29.9, 36.1, 45.1, 55.1, 100),
                                                        labels = c('15-22 years', '22-30 years', '30-36 years', '36-45 years', '45-55 years','55-100 years'))) %>%
  select(-age)

cust_df <- cust_df %>% select(-FN, -Active) # since they have a lot of missing values.
cust_df <- distinct(cust_df) # to remove duplicate rows.

# sampling the data
customers_sample <- cust_df[1:2000,]
customer_ids <- customers_sample$customer_id
customers_sample <- customers_sample %>% select(-postal_code)
transacs_sample <- df %>% filter(customer_id %in% customer_ids)
article_ids <- unique(transacs_sample$article_id)
articles_sample <- art_df %>% filter(article_id %in% article_ids)

# we factorize each variables in our data
customers_sample$avg_days_diff <- factor(customers_sample$avg_days_diff, levels = c(0, 1, 3, 7, 14, 21, 30))
customers_sample$days_since_last_purchase <- factor(customers_sample$days_since_last_purchase, 
                                                    levels = c('1 day', 'around 3 days', 'around a week', 
                                                               'around 2 weeks','around 3 weeks', 
                                                               'around a month', 'more than a month'))
# these variables should be added for the spatio-temporal model, so we temporarily remove them.
customers_sample <- customers_sample %>% select(-avg_days_diff, -days_since_last_purchase)

customers_sample$club_member_status <- factor(customers_sample$club_member_status, levels = c('ACTIVE','PRE-CREATE'))
customers_sample$fashion_news_frequency <- factor(customers_sample$fashion_news_frequency, levels = c('NONE', 'Regularly'))
customers_sample$customer_age_groups <- factor(customers_sample$customer_age_groups, levels = c('15-22 years', 
                                                                                                '22-30 years', 
                                                                                                '30-36 years', 
                                                                                                '36-45 years', 
                                                                                                '45-55 years',
                                                                                                '55-100 years'))
customers_sample$avg_price_spent <- factor(customers_sample$avg_price_spent, levels = c('Very less than avg', 
                                                                                        'less than avg', 
                                                                                        'around avg',
                                                                                        'more than avg', 
                                                                                        'a lot more than avg'))


customer_ids <- customers_sample$customer_id
# dummyVars is used to make our data binary. 
dummy <- dummyVars(" ~ .", data=customers_sample[,-1])
df_fin_1 <- as.matrix(predict(dummy, newdata = customers_sample[,-1]))
dim(df_fin_1)

# removing irrelevant features. 
df_fin_2 <- transacs_sample %>% select(-c(t_dat, price, sales_channel_id, age, postal_code, week_before)) %>% 
  group_by(customer_id, article_id) %>% mutate(purchase=1) %>% ungroup() %>% 
  distinct()
# here we spread our data to add new features based on the purchases of each customer.
# a new feature for each article available. if a customer has bought an article, it will
# be 1 under the feature for that article, if the customer has not bought that article, 
# it would be 0 
df_mat <- spread(df_fin_2, article_id, purchase, fill=0) %>% select(-c(customer_id))
df_cust_ids <- unique(transacs_sample$customer_id)

df_mat <- cbind(df_cust_ids, df_mat)

df_fin_1 <- cbind(customer_ids, df_fin_1)

rownames(df_fin_1) <- 1:nrow(df_fin_1)
rownames(df_mat) <- 1:nrow(df_mat)

# Now we join all features in df_fin_1 and df_mat into one data frame based on the
# customer ids. You can use colnames() to see which features are in df_fin_1, and df_mat
df_fin <- left_join(data.frame(df_fin_1), data.frame(df_mat),  by=c('customer_ids'='df_cust_ids')) %>% distinct()
df_fin_ <- apply(df_fin[,-1], 2, as.numeric)
df_fin_ <- apply(df_fin_, 2, replace_na, 0)
df_brm <- as(df_fin_, 'binaryRatingMatrix')
sim_mat <- similarity(df_brm, method='cosine', which='users')
sim_mat <- as.matrix(sim_mat)
sim_mat <- apply(sim_mat, 2, replace_na, 1)

# top k similar users for a given user_id based on the similarity matrix that is given
top_k_similar_users <- function(user_id, similarity_matrix, k = 4){
  user_row_index <- which(rownames(sim_mat) == user_id)
  ordered <- order(sim_mat[user_row_index,], decreasing=TRUE)[1:k+1]
  top_k <- rownames(sim_mat)[ordered]
  return(top_k[top_k != user_id])
}

rownames(sim_mat) <- customer_ids
colnames(sim_mat) <- customer_ids
sim_mat[1:3,1:3]

top_k_recs <- function(user_id, similarity_matrix, transactions_df, nn, k){
  # for user_id find top nn common items from top k similar users
  # if the number of common items do not reach nn, select additional
  # items from the most similar user until the number totals to nn items
  top_k_sim_users <- top_k_similar_users(user_id, similarity_matrix, k)
  filtered_transacs <- transactions_df %>% filter(customer_id %in% top_k_sim_users)
  user_id_items <- transactions_df %>% filter(customer_id == user_id) %>% select(article_id) %>% unique()
  user_id_items <- user_id_items$article_id
  rec_items <- filtered_transacs %>% filter(!(article_id %in% user_id_items)) %>% group_by(article_id) %>%
    filter(n_distinct(customer_id) == n_distinct(filtered_transacs$customer_id)) %>% select(article_id) %>%
    unique()
  rec_items <- rec_items[[1]]
  len <- length(rec_items)
  if(len >= nn) {return(rec_items[1:nn])}
  i <- 1
  while((len < nn) && (i <= k)) {
    items <- filtered_transacs %>% filter(customer_id == top_k_sim_users[i]) %>% 
      filter(!(article_id %in% rec_items) && !(article_id %in% user_id_items)) %>% select(article_id) %>% unique()
    items <- items[[1]]
    rec_items <- append(rec_items, items)
    len <- length(rec_items)
    i <- i + 1
  }
  return(rec_items[1:nn])
}

# because of small sample we need to reduce our test set to users that have been previously seen. 
df_test_small <- df_test %>% filter(customer_id %in% customers_sample$customer_id)
test_user_ids <- unique(df_test_small$customer_id)
length(test_user_ids)

prediction <- function(test_users, similarity_matrix, transactions_df, nn, k){
  if(is.vector(test_users)){
    test_users <- data.frame(customer_id = test_users)
  }
  if(!('customer_id' %in% colnames(test_users)) && is.vector(test_users)){
    stop('please provide a dataframe or vector that includes customer ids')
  }
  else{
    
    customer_ids <- test_users$customer_id
    users_recommendations <- t(sapply(customer_ids, top_k_recs, similarity_matrix=similarity_matrix, 
                                      transactions_df=transactions_df, nn=nn, k=k))
  }
  return(users_recommendations)
}

df_test_preds <- prediction(test_users=test_user_ids, similarity_matrix=sim_mat, transactions_df=transacs_sample, nn=4, k=4)

# evaluation of results
df_test_true <- df_test_small %>% group_by(customer_id) %>% summarize(article_id = paste(sort(article_id), collapse=' '))



# these two data frames should have identical dimensions
dim(df_test_true)
dim(df_test_preds)


# map@12 function
map12 <- function(predictions, true_values) {
  n_users <- nrow(predictions) # number of users
  if(n_users != nrow(true_values)) {stop('Number of users in predicted values differs from number of users in true values')}
  
  n_preds <- ncol(predictions) # number of predictions per user
  # average precision for user u
  avg_precision_u <- function(u){
    precision <- c()
    rel <- c()
    true_items <- true_values %>% filter(customer_id %in% u) %>% select(article_id)
    true_items <- true_items$article_id
    true_items <- strsplit(true_items, ' +')[[1]]
    m <- length(true_items) # number of ground truth items user 'u' actually bought
    
    for(j in 1:min(n_preds, 12)) {
      pred_items <- predictions[u,1:j]
      true_positives <- length(which(pred_items %in% true_items)) # number of recommendations that are relevant
      n_rec <- j # total number of recommended items
      precision[j] <- true_positives/n_rec
      rel[j] <- predictions[u, j] %in% true_items*1 # indicator function: 1 if item j is relevant, 0 o.w
    }
    sum(precision*rel)/(1/min(m, 12))
  }
  
  # mean average precision (map)
  return(mean(sapply(rownames(predictions), avg_precision_u)))
}

# for the data without spatiotemporal features
map12_1 <- map12(df_test_preds, df_test_true)
map12_1

# # UBCF with spatio-temporal data

# sampling the data
customers_sample <- cust_df[1:2000,]
customer_ids <- customers_sample$customer_id
customers_sample <- customers_sample %>% select(-postal_code)
transacs_sample <- df %>% filter(customer_id %in% customer_ids)
article_ids <- unique(transacs_sample$article_id)
articles_sample <- art_df %>% filter(article_id %in% article_ids)

customers_sample$avg_days_diff <- factor(customers_sample$avg_days_diff, levels = c(0, 1, 3, 7, 14, 21, 30))
customers_sample$days_since_last_purchase <- factor(customers_sample$days_since_last_purchase, 
                                                    levels = c('1 day', 'around 3 days', 'around a week', 
                                                               'around 2 weeks','around 3 weeks', 
                                                               'around a month', 'more than a month'))
customers_sample$club_member_status <- factor(customers_sample$club_member_status, levels = c('ACTIVE','PRE-CREATE'))
customers_sample$fashion_news_frequency <- factor(customers_sample$fashion_news_frequency, levels = c('NONE', 'Regularly'))
customers_sample$customer_age_groups <- factor(customers_sample$customer_age_groups, levels = c('15-22 years', 
                                                                                                '22-30 years', 
                                                                                                '30-36 years', 
                                                                                                '36-45 years', 
                                                                                                '45-55 years',
                                                                                                '55-100 years'))
customers_sample$avg_price_spent <- factor(customers_sample$avg_price_spent, levels = c('Very less than avg', 
                                                                                        'less than avg', 
                                                                                        'around avg',
                                                                                        'more than avg', 
                                                                                        'a lot more than avg'))

customer_ids <- customers_sample$customer_id
dummy <- dummyVars(" ~ .", data=customers_sample[,-1])
df_fin_1 <- as.matrix(predict(dummy, newdata = customers_sample[,-1]))
dim(df_fin_1)

df_fin_2 <- transacs_sample %>% select(-c(t_dat, price, sales_channel_id, age, postal_code, week_before)) %>% 
  group_by(customer_id, article_id) %>% mutate(purchase=1) %>% ungroup() %>% 
  distinct()
df_mat <- spread(df_fin_2, article_id, purchase, fill=0) %>% select(-c(customer_id))
df_cust_ids <- unique(transacs_sample$customer_id)

df_mat <- cbind(df_cust_ids, df_mat)
df_fin_1 <- cbind(customer_ids, df_fin_1)
rownames(df_fin_1) <- 1:nrow(df_fin_1)
rownames(df_mat) <- 1:nrow(df_mat)

df_fin <- left_join(data.frame(df_fin_1), data.frame(df_mat),  by=c('customer_ids'='df_cust_ids')) %>% distinct()
df_fin_ <- apply(df_fin[,-1], 2, as.numeric)
df_fin_ <- apply(df_fin_, 2, replace_na, 0)
df_brm <- as(df_fin_, 'binaryRatingMatrix')
sim_mat <- similarity(df_brm, method='cosine', which='users')
sim_mat <- as.matrix(sim_mat)
sim_mat <- apply(sim_mat, 2, replace_na, 1)

rownames(sim_mat) <- customer_ids
colnames(sim_mat) <- customer_ids

df_test_preds_new <- prediction(test_users=test_user_ids, similarity_matrix=sim_mat, transactions_df=transacs_sample, nn=4, k=4)

map12_2 <- map12(df_test_preds_new, df_test_true)
map12_2

map12_1