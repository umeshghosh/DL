import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
		  
X=pd.read_csv('svm_input_1k.csv',index_col=0).T.values
y=to_categorical(pd.read_csv('class3.csv',index_col=0,names=['x']))

def ml():	
	m = Sequential()
	# init='normal',
	m.add(Dense(12, input_dim=X.shape[1], activation='relu'))
	m.add(Dense(3, activation='sigmoid'))
	
	m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return m
	
est = KerasClassifier(build_fn=ml, nb_epoch=100, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, shuffle=True, random_state=0)

res = cross_val_score(est, X, y, cv=kfold)

print("CV Score: %.2f%% (%.2f%%)" % (res.mean()*100, res.std()*100))




