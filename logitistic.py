from sklearn.linear_model import logitisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

__author__='Research group of Huanyang Chen'

df=pd.read_excel(r"DataSource\pd_data\pd_file.xlsx",index_col=0)

X=df[df.columns[0:40]]    # Get the input variable of the data.
y=df[df.columns[40]]      # Get the laebl of the data.
xTrain,xTest,yTrain,yTest=train_test_split(X,y,test_size=0.2)

# Train the model.
model=logitisticRegression(multi_class="multinomial",solver="newton-cg",penalty='l2')   # Get the logitistic regression model. Used L2 regularization to prevent the model from overfitting.
model.fit(xTrain,yTrain.astype('int'))	 # Train the logitistic regression model.
# Evaluation the model.
yPre=model.predict(xTest)
accuracy=sum(yTest.tolist()==yPre)/len(yTest)
print('The accuracy of logitistic Regression is: ',accuracy)