# Are there any road properties (intersections, location types, road condition), weather conditions or times of day that have a higher incidence of fatal crashes? Can we use predictive modeling to identify these high-risk areas and suggest road repair or redesign to mitigate the risks?



# Load the required library
library(dplyr)
library(car)

gen.dat<-read.csv("crash_info_general.csv", header=T)


merged.dat <- subset(merged.dat, select=c(BELTED_DEATH_COUNT, BICYCLE_DEATH_COUNT, CHLDPAS_DEATH_COUNT, MCYCLE_DEATH_COUNT,
                                          NONMOTR_DEATH_COUNT, PED_DEATH_COUNT, UNB_DEATH_COUNT,
                                          WEATHER1, WEATHER2, INTERSECT_TYPE,
                                          INTERSECTION_RELATED, TIME_OF_DAY,
                                          ROAD_CONDITION))
merged.dat$WEATHER1 <- as.factor(merged.dat$WEATHER1)
table(merged.dat$WEATHER1)
class(merged.dat$WEATHER1)

merged.dat$WEATHER2 <- as.factor(merged.dat$WEATHER2)
table(merged.dat$WEATHER2)
class(merged.dat$WEATHER2)

merged.dat$INTERSECT_TYPE <- as.factor(merged.dat$INTERSECT_TYPE)
table(merged.dat$INTERSECT_TYPE)
class(merged.dat$INTERSECT_TYPE)

# Convert to factor variable
merged.dat$INTERSECTION_RELATED <- as.factor(merged.dat$INTERSECTION_RELATED)
table(merged.dat$INTERSECTION_RELATED)
class(merged.dat$INTERSECTION_RELATED)

merged.dat$ROAD_CONDITION <- as.factor(merged.dat$ROAD_CONDITION)
table(merged.dat$ROAD_CONDITION)
class(merged.dat$ROAD_CONDITION)

merged.dat <- merged.dat %>% 
  mutate(total_death = BELTED_DEATH_COUNT + BICYCLE_DEATH_COUNT + CHLDPAS_DEATH_COUNT + MCYCLE_DEATH_COUNT + NONMOTR_DEATH_COUNT + PED_DEATH_COUNT + UNB_DEATH_COUNT)

merged.dat <- merged.dat %>% 
  mutate(total_death = BELTED_DEATH_COUNT + BICYCLE_DEATH_COUNT + CHLDPAS_DEATH_COUNT + MCYCLE_DEATH_COUNT +
           NONMOTR_DEATH_COUNT + PED_DEATH_COUNT + UNB_DEATH_COUNT,
         fatal = ifelse(total_death > 0, 1, 0))

# Fit a logistic regression model
model1 <- glm(fatal ~ WEATHER1 + WEATHER2 + INTERSECT_TYPE + 
               INTERSECTION_RELATED + TIME_OF_DAY + ROAD_CONDITION, 
             data = merged.dat, family = binomial(link="logit"))

# Print the model summary
summary(model1)

# INTERSECT_TYPE, INTERSECTION_RELATE, TRAVEL_DIRECTION, TIME_OF_DAY, ROAD_CONDITION (7:snow)
