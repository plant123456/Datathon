# Are there any road properties (intersections, location types, road condition), weather conditions that have a higher incidence of fatal crashes? Can we use predictive modeling to identify these high-risk areas and suggest road repair or redesign to mitigate the risks?
# Load the required library
library(dplyr)
library(caret)
library(gbm)
library(car)
library(psych)  # Load the psych package for EFA
library(heatmaply)

gen.dat<-read.csv("crash_info_general.csv", header=T)
flag.dat<-read.csv("crash_info_flag_variables.csv", header=T)
# Merge the two datasets based on CRN variable
merged.dat <- merge(gen.dat, flag.dat, by="CRN")

# filter
merged.dat <- subset(merged.dat, select=c(FATAL_COUNT, HIT_TREE_SHRUB,
                                       WEATHER1, WEATHER2, INTERSECT_TYPE, ILLUMINATION,
                                       INTERSECTION_RELATED, RELATION_TO_ROAD,
                                       TCD_FUNC_CD, ROAD_CONDITION))
merged.dat$INTERSECTION_RELATED <- ifelse(merged.dat$INTERSECTION_RELATED == "Y", 1, 
                                          ifelse(merged.dat$INTERSECTION_RELATED == "N", 0, NA))

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



# Fit a logistic regression model
model1 <- glm(FATALITY ~ WEATHER1 + WEATHER2 + INTERSECT_TYPE + ILLUMINATION + HIT_TREE_SHRUB +
                INTERSECTION_RELATED + ROAD_CONDITION +
                RELATION_TO_ROAD + TCD_FUNC_CD, 
              data = merged.dat, family = binomial(link="logit"))

# Print the model summary
summary(model1)
vif(model1)
# VIF of WEATHER1>10, consider to remove

model2 <- glm(FATALITY ~ WEATHER2 + INTERSECT_TYPE + ILLUMINATION + HIT_TREE_SHRUB +
                INTERSECTION_RELATED + ROAD_CONDITION +
                RELATION_TO_ROAD + TCD_FUNC_CD, 
              data = merged.dat, family = binomial(link="logit"))
summary(model2)
vif(model2)




# Exploratory Factor Analysis
merged.dat$WEATHER2 <- as.numeric(as.character(merged.dat$WEATHER2))
merged.dat$INTERSECT_TYPE <- as.numeric(as.character(merged.dat$INTERSECT_TYPE))
merged.dat$ILLUMINATION <- as.numeric(as.character(merged.dat$ILLUMINATION))
merged.dat$INTERSECTION_RELATED <- as.numeric(as.character(merged.dat$INTERSECTION_RELATED))
merged.dat$ROAD_CONDITION <- as.numeric(as.character(merged.dat$ROAD_CONDITION))
merged.dat$RELATION_TO_ROAD <- as.numeric(as.character(merged.dat$RELATION_TO_ROAD))
merged.dat$TCD_FUNC_CD <- as.numeric(as.character(merged.dat$TCD_FUNC_CD))

dat <- merged.dat[, c("WEATHER2", "INTERSECT_TYPE", "HIT_TREE_SHRUB", "ILLUMINATION", "ROAD_CONDITION", "INTERSECTION_RELATED", "RELATION_TO_ROAD", "TCD_FUNC_CD")]

efa_res <- fa(dat, nfactors = 3)

print(efa_res)



# Create a matrix of the loadings from your factor analysis
matrix_data <- matrix(c(0.00, 0.17, 0.25, 0.29, 0.07, -0.05, -0.10, 0.10, -0.07, -0.10, 0.20, 0.03, 0.00, 0.40, 0.54, 0.23, 0.04, -0.03, -0.52, 0.54, -0.36, 0.94, 0.29, -0.18),
                      ncol = 3, byrow = TRUE,
                      dimnames = list(c("WEATHER2", "INTERSECT_TYPE", "HIT_TREE_SHRUB", "ILLUMINATION",
                                        "ROAD_CONDITION", "INTERSECTION_RELATED", "RELATION_TO_ROAD", "TCD_FUNC_CD"),
                                      c("MR1", "MR2", "MR3")))

# Create a heatmap using heatmaply
heatmaply(matrix_data, xlab = "Factors", ylab = "Variables", 
          main = "Factor Analysis Heatmap",
          Rowv = FALSE, Colv = FALSE)



# Logistic regression using latent variables

# Add latent variables
dat$factor1 <- efa_res$scores[,1]
dat$factor2 <- efa_res$scores[,2]
dat$factor3 <- efa_res$scores[,3]
dat$FATALITY  <- merged.dat$FATALITY

# Fit a logistic regression model
model3 <- glm(FATALITY ~ factor1 + factor2 + factor3, data = dat, family = binomial(link="logit"))

# Output
summary(model3)







