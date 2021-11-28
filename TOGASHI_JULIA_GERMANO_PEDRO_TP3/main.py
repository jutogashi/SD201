import pandas as pd
import numpy as np
from decision_functions import *

#%%
###########################
#TP4
#Júlia Togashi de Miranda
#Pedro Germano Almeida Machado
###########################

df = pd.read_csv("data.csv")
df.rename(columns={'Survived':'Class'}, inplace=True)
t=BuildDeciosionTree(df,5,0)
printDecisionTree(t,"output_tree.txt")
g=generalizationError(df,t,0.5)
print(g,"\n\n\nPrune\n")
p=pruneTree(df,t,0.5,5,0)
printDecisionTree(p,"postpruned tree.txt")
g=generalizationError(df,p,0.5)
print(g)
