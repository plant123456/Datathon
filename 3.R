# Are there any road properties (intersections, location types, road condition), weather conditions that have a higher incidence of fatal crashes? Can we use predictive modeling to identify these high-risk areas and suggest road repair or redesign to mitigate the risks?
# Load the required library
library(dplyr)
library(caret)
library(neuralnet)
library(gbm)
library(ggplot2)

gen.dat<-read.csv("crash_info_general.csv", header=T)

# filter
merged.dat <- subset(gen.dat, select=c(FATAL_COUNT,
                                       WEATHER1, WEATHER2, INTERSECT_TYPE, ILLUMINATION,
                                       INTERSECTION_RELATED, RELATION_TO_ROAD,
                                       TCD_FUNC_CD, ROAD_CONDITION))

# Convert to factor variable
merged.dat$TCD_FUNC_CD <- as.factor(merged.dat$TCD_FUNC_CD)


merged.dat$RELATION_TO_ROAD <- as.factor(merged.dat$RELATION_TO_ROAD)
table(merged.dat$RELATION_TO_ROAD)
class(merged.dat$RELATION_TO_ROAD)

merged.dat$ILLUMINATION <- as.factor(merged.dat$ILLUMINATION)
table(merged.dat$ILLUMINATION)
class(merged.dat$ILLUMINATION)

merged.dat$WEATHER1 <- as.factor(merged.dat$WEATHER1)
table(merged.dat$WEATHER1)
class(merged.dat$WEATHER1)

merged.dat$WEATHER2 <- as.factor(merged.dat$WEATHER2)
table(merged.dat$WEATHER2)
class(merged.dat$WEATHER2)

merged.dat$INTERSECT_TYPE <- as.factor(merged.dat$INTERSECT_TYPE)
table(merged.dat$INTERSECT_TYPE)
class(merged.dat$INTERSECT_TYPE)

merged.dat$INTERSECTION_RELATED <- as.factor(merged.dat$INTERSECTION_RELATED)
table(merged.dat$INTERSECTION_RELATED)
class(merged.dat$INTERSECTION_RELATED)

merged.dat$ROAD_CONDITION <- as.factor(merged.dat$ROAD_CONDITION)
table(merged.dat$ROAD_CONDITION)
class(merged.dat$ROAD_CONDITION)

# create a fatal predictor (binary)
merged.dat$FATALITY <- ifelse(merged.dat$FATAL_COUNT > 0, 1, 0)

# set baseline
merged.dat$INTERSECTION_RELATED <- relevel(merged.dat$INTERSECTION_RELATED,
                                           ref="N")

# Fit a logistic regression model
model1 <- glm(FATALITY ~ WEATHER1 + WEATHER2 + INTERSECT_TYPE + ILLUMINATION +
                INTERSECTION_RELATED + ROAD_CONDITION +
                RELATION_TO_ROAD + TCD_FUNC_CD, 
              data = merged.dat, family = binomial(link="logit"))

# Print the model summary
summary(model1)

#linear regression
model2 <- lm(FATAL_COUNT ~ WEATHER1 + WEATHER2 +
               INTERSECT_TYPE + ILLUMINATION + INTERSECTION_RELATED +
               RELATION_TO_ROAD + TCD_FUNC_CD + ROAD_CONDITION, data = merged.dat)
summary(model2)

# cross-validation
merged.dat <- na.omit(merged.dat)

# Define the formula for the logistic regression model
formula1 <- FATALITY ~ WEATHER1 + WEATHER2 + INTERSECT_TYPE + ILLUMINATION +
  INTERSECTION_RELATED + ROAD_CONDITION +
  RELATION_TO_ROAD + TCD_FUNC_CD

# Define the formula for the linear regression model
formula2 <- FATAL_COUNT ~ WEATHER1 + WEATHER2 +
  INTERSECT_TYPE + ILLUMINATION + INTERSECTION_RELATED +
  RELATION_TO_ROAD + TCD_FUNC_CD + ROAD_CONDITION

# Define the models
model1 <- train(formula1, data = merged.dat, method = "glm", family = "binomial", trControl = trainControl(method = "cv", number = 10))
model2 <- train(formula2, data = merged.dat, method = "lm", trControl = trainControl(method = "cv", number = 10))

# Print the results
print(model1)
print(model2)






# Gradient Boosting for Logistic Regression Model
# Prepare data
merged.dat$FATALITY <- as.factor(merged.dat$FATALITY) # convert FATALITY to factor
train.idx <- sample(nrow(merged.dat), nrow(merged.dat) * 0.8) # index for training data
train <- merged.dat[train.idx, ]
test <- merged.dat[-train.idx, ]


gbm.fit <- gbm(
  formula = FATALITY ~ WEATHER1 + WEATHER2 + INTERSECT_TYPE + ILLUMINATION + INTERSECTION_RELATED + ROAD_CONDITION + RELATION_TO_ROAD + TCD_FUNC_CD, 
  distribution = "bernoulli",
  data = train,
  n.trees = 1000,
  interaction.depth = 3,
  shrinkage = 0.01,
  cv.folds = 5,
  n.cores = NULL,
  verbose = TRUE
)

# Predict on test data
gbm.pred <- predict(gbm.fit, newdata = test, n.trees = 1000)

# Evaluate the model
library(ROCR)
auc <- performance(prediction(gbm.pred, test$FATALITY), "auc")@y.values[[1]]
cat("AUC: ", auc, "\n")




# Gradient Boosting for Linear Regression Model
# Prepare data
train.idx <- sample(nrow(merged.dat), nrow(merged.dat) * 0.8) # index for training data
train <- merged.dat[train.idx, ]
test <- merged.dat[-train.idx, ]

# Train the model
gbm.fit <- gbm(
  formula = FATAL_COUNT ~ WEATHER1 + WEATHER2 + INTERSECT_TYPE + ILLUMINATION + INTERSECTION_RELATED + ROAD_CONDITION + RELATION_TO_ROAD + TCD_FUNC_CD, 
  distribution = "gaussian",
  data = train,
  n.trees = 1000,
  interaction.depth = 3,
  shrinkage = 0.01,
  cv.folds = 5,
  n.cores = NULL,
  verbose = TRUE
)

# Predict on test data
gbm.pred <- predict(gbm.fit, newdata = test, n.trees = 1000)

# Evaluate the model
library(Metrics)
rmse <- rmse(gbm.pred, test$FATAL_COUNT)
cat("RMSE: ", rmse, "\n")

