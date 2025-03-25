####### Sarah Martin

### AI such as Chat GPT should not be used on this project 
### Answers that are general and don't relate to the actual graphs and
### output produced in the code may result in a 0 for the entire exam.



#### Exam 1 File
#### Students are to complete this on their own. Not in groups or teams
#### You should be able to adapt this file as directed in comment lines.
import kagglehub
import pandas as pd
import os 
import seaborn as sns 
import matplotlib.pyplot as plt 
import statsmodels.api as sm


# Download latest version
path = kagglehub.dataset_download("adilshamim8/student-depression-dataset")

# Ensure the expected file exists
file_to_read = os.path.join(path, 'student_depression_dataset.csv')
print("File to read:", file_to_read)

if os.path.exists(file_to_read):
    student = pd.read_csv(file_to_read)
    print("Data loaded successfully.")
    print(student.head())  # Display first few rows
else:
    print("Error: File not found at", file_to_read)
    
    
### initial investigation:
print(list(student.columns))


######################################################################
####################### Part 1 #######################################
######################################################################



#######################################################################
# List the 3 worse variable names that would be poblematic for a data scientist #
######  Worst 3 Variable Names: sm, print, and plt

# And explain why #
######  I think these would be the worst variable names because it would make it so 
######  every time you used statsmodel.api, the print function, or matplotlib.pyplot you
######  would have to qualify it using the namespace as a prefix to avoid confusion with 
######  the variables.
#######################################################################

pd.plotting.scatter_matrix(student)
### Add a line of code to view the scatterplot matrix
plt.show()
#########################################################
#This scatterplot matrix from pandas while a useful starting point
# is fairly confusing.  Write a paragraph explaining the challenges
# in interpreting this plot.
######  It seems like the reason why this scatterplot matrix is difficult to interpret is 
######  because there are so many variables and it's showing too much information at once by 
######  showing the relationship between every variable and every other variable. This makes
######  it so that the scales of each pair of variables are really hard to see, and it's also
######  hard to tell which data is even relevant. On a smaller screen, the labels also overlap 
######  each other because there are so many of them.

#####################################
df = student['Gender'].value_counts() 
# Add a commmand to view the freqency table
# and report the males and females in this dataset
print(df)
###### Males:    15547
###### Females:  12354

# We can make a barplot from this data
# As follows.  But I would like you to 
# add a title and labels to the graph below

df.plot(kind = 'bar', xlabel="Gender", ylabel="Count", title="Count of Participants by Gender")
plt.show()

###############################################
#The seaborn library makes a better scatterplot matrix
# Read about the hue command and make male/female different colors
# https://seaborn.pydata.org/examples/scatterplot_matrix.html
# in the sns.pairplot(student) command below.
##################################################
# sns.pairplot(student, hue="Gender")
# plt.show()

### This scatterplot matrix has better attributes.
### Please note it only plots the numeric variables.
# Use the barplot created earlier to explain why the blue 
# distributions in the diagonals of the scatterplot matrix 
#are almost always higher than the orange

######  There is a greater number of male participants in the study (represented by
######  the blue) than female, which would make the blue distributions higher.

# A close inspection of the scatterplot matrix only reveals a few clues.
# Look closely at the CGPA row.  What sticks out on that row? Also what scale is it
# on?  Did you know many places in the world don't use a 4.0 GPA scale?

#######  It appears that pretty much all the data points on the CGPA row are at 5.0 
#######  and above. The scale goes from about 0 to 10. I was not aware many places in
#######  the world don't use a 4.0 GPA scale, but it doesn't surprise me since I know
#######  different countries do lots of things differently and also since humans like
#######  things being set up in base 10 because we have 10 fingers so it makes 
#######  counting easier.

# Make at least 4 other observations from investigating the scatterplot matrix
# Directly related to your data  If you choose you can remake it and color it based on 
# Depression instead of male/female...

#######  I put in my best effort to be able to see the graph well enough to do this; I
#######  spent over an hour between trying to get my laptop display to scale smaller 
#######  so it could display the whole graph on my screen only to find it is unable to
#######  scale down and can only scale up, attempting to use a tv as a monitor in hopes
#######  it could display it on a different scale (it couldn't), trying to run the code
#######  from the lab computers at the science building in hopes that it could display
#######  on those (after about 20 minutes of setting up Ubuntu so I'd be able to install
#######  Python packages, I learned that I still couldn't display the plots), and looking
#######  for other ways around the limitations of my hardware. I ended up stuck with 
#######  running the code from my laptop with the too-small screen, and because of how
#######  much memory the data set takes up with the pairplot and the almost 30,000 rows
#######  of data (about 80-90% of my laptop's available memory was being used just to gen-
#######  erate the plot), every time I tried to pan or zoom on the graph, the window would
#######  freeze up and stop responding; and it would take at least 30-45 seconds to re-
#######  spond at all to any click. I couldn't see anyof the labels on the bottom of the 
#######  graph, so I couldn't really understand the data well enough to make well-informed
#######  observations about it. I am terribly sorry, but my failed attempts are all I've 
#######  got. I hope you will have mercy on the points for this part.


######################################################################
####################### Part 2 #######################################
######################################################################

##### First suppose we want to predict CGPA from Study 
#The following model could be run:
  
# Define independent (X) and dependent (y) variables
y = student['CGPA']

# Use the correct column names
X = student['Work/Study Hours']

# Add a constant for the intercept
X = sm.add_constant(X)
X = X.astype(float)  # Forces everything to be a float

# Final check
print("Final columns in X:", X.columns)
print(X.isnull().sum()) #OLS can break with missing values

# Fit the regression model
model = sm.OLS(y, X).fit()

################ What does OLS mean in this stats model?
######  OLS stands for "Ordinary Least Squares" and refers to 
######  ordinary least squares regression.


########## How does this relate to calculus? ###########
#######  This relates to calculus because ordinary least squares regression works by
#######  minimizing the sum of the squared errors between the predicted and actual
#######  values, and minimization is a calculus concept.


# Print summary of the regression
print(model.summary())

###################################################
########### What is the model predicted? ##########
### Actually write the equation ###################

######  0.001("Work/Study Hours") + 7.6487

# Demonstrate how that would work if and individual
# reported 8 work/study hours #

############### Find 3 statistics in the output that show
############### This model is NOT good/ reliable at all!

######  Probability is 0.664
######  R-squared is 0
######  Adjusted R-squared is -0

### Here is a scatterplot showing this model ###
### Add a meaningful title to this plot
sns.scatterplot(data = student,
                x = 'Work/Study Hours',
                y = 'CGPA')
plt.title("Student CGPA Compared to Study Hours")
plt.show()

# Is there any transformation that you would recommend to improve
# The model? If so what would it be.  If not why not?...

######  No; it appears that the data is probably very clustered in such 
######  a way that there is no strong trend, so using a transformation
######  probably wouldn't help much. This especially seems to be the
######  case since the log likelihood is less than -50,000.


#### What is step down vs step up regression?  ### 
######  Step down is when variables are removed from the regression one
######  by one to get a better model; step up is when you strip it to a
######  single variable and then add them back in one by one to get a 
######  more predictive model that way.



# we are going to do an adaptive form of it.
# Separate features (X) and target (y)
# Select only numeric columns
numeric_data = student.select_dtypes(include=['number'])
X = numeric_data.drop('CGPA', axis=1)  # Replace 'target_variable_name'
y = numeric_data['CGPA']
X = sm.add_constant(X)
X = X.astype(float)  # Forces everything to be a float

# Final check
print("Final columns in X:", X.columns)
print(X.isnull().sum()) #OLS can break with missing values

# Fit the regression model
model = sm.OLS(y, X).fit()
print(model.summary())


# referencing your scatterplot matrix and the model summary
# find a numeric variable with a meaningul distribution
# DO NOT use depression: Even though it was recorded as 0 and 
# 1 it is still categorical.
# Commented below is the code for our original model
# Run a regression model to predict CGPA on the numeric variable
# of your choosing
# Uncomment and edit the section of code below as necessary to make it 
# Run.  

# Define independent (X) and dependent (y) variables
y = student['CGPA']
# Use the correct column names
X = student['Work/Study Hours']
# Add a constant for the intercept
X = sm.add_constant(X)
X = X.astype(float)  # Forces everything to be a float
# Final check
print("Final columns in X:", X.columns)
print(X.isnull().sum()) #OLS can break with missing values
# Fit the regression model
model = sm.OLS(y, X).fit()

# Report the model produced.  Give an example of how it could 
# be used and share some statistics to show if it is a valid estimation

print(model.summary())

#######################################################################
####################### Part 3 #######################################
######################################################################
######## Now lets build the model further!
# to make a multiple regression model the X should be passed a tuple
# The double brackets lets you pass more than 1 independent variable 
#X = student[['x1', 'x2']]
# Copy relevant code from above and expand it to be a multiple regression model









# Report the model produced.  Give an example of how it could 
# be used and share some statistics to show if it is a valid estimation
# Also mention if it is better or worse than the previuos model and how you know






# Explain what colinearity is and how it can impact multiple regression.  
# Look at the coefficients, the R-squared and other statistics the earlier full model produced
# Do you think you have colinearity?  Why or why not?



# Regardless of p-values use the following code to re-run the model
# Replace variable 1 and variable 2 with the variables you chose previously.
# What happens when we want to add a categorical data to the dataset? 
X = student[['Gender', 'variable_1', 'variable 2']]
X = pd.get_dummies(X, drop_first=True) 
# explain the get_dummies command above... What does it do? and how does it impact regression?



#### now run the following
X = sm.add_constant(X)
X = X.astype(float)  # Forces everything to be a float
# Final check
print("Final columns in X:", X.columns)
print(X.isnull().sum()) #OLS can break with missing values
# Fit the regression model
model = sm.OLS(y, X).fit()


# Report the model produced # 
# Clearly explain how to use the model for a male...



# Did gender improve the model?  Report 2 statistics to show evidence





######################################################################
####################### Part 4 #######################################
######################################################################
# Great Job making to here!  Now lets do some: logisitic regression !!!!


X_logit = sm.add_constant(student['Age'])
logit_model = sm.Logit(student['Depression'], X_logit).fit()
print(logit_model.summary())
### Run the above model # Report the formula produced








### Explain how to use the formula used.  Specifically
# If a student is 20 what is the probability of depression for this population
# If a student is 30 what is the probabilty of depression for this population?
# 


# Predictions
student['Predicted_Depression'] = logit_model.predict(X_logit)
plt.scatter(student['Age'], 
            student['Predicted_Depression'], 
            color='red', 
            alpha=0.5, 
            label='Predicted')
plt.xlabel("Age in years")
plt.ylabel("Probability of Depression")
plt.title("Logistic Regression Predictions")
plt.legend()
plt.show()
 
##### You can see the sigmoid function curving the graph
##### Why do you suppose the graph doesn't go all the way to 1
##### on the x-axis?  Is this a good thing or not in the context 
# of this data?




# Run the model
X_logit = sm.add_constant(student[['Age', 'Academic Pressure']])
logit_model = sm.Logit(student['Depression'], X_logit).fit()
print(logit_model.summary())


# Report the model produced #




# Add another independent variable to the model below.  If it is not 
# significant switch it out until you find one that is
# If you desire you can modify the code to add a categorical variable
X_logit = sm.add_constant(student[['Age', 'Academic Pressure']])
logit_model = sm.Logit(student['Depression'], X_logit).fit()
print(logit_model.summary())



# If you were an adminster at the academic instituion 
# that this data came from: What recommendations might 
# you make after seeing these models?


# We did not make a training/ testing dataset on this data
# Do we have enough data to do so?  Justify it based on the size of the datamatrix




# What would be the advantage of doing so?





################### End of Python Part of Exam ###################


