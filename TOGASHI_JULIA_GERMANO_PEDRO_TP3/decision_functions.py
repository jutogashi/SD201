# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 23:55:05 2020

@author: jutogashi
"""
#%%
import pandas as pd
import numpy as np
#%%
###########################
#TP4
#Júlia Togashi de Miranda
#Pedro Germano Almeida Machado
###########################


#Here we define the classes used in our decision tree. 
#First we have the Desision tree class, that is an assemble of nodes
class DecisionTree:
    def __init__(self,nodes):
        self.nodes=nodes

#in the node classe qe hava the useful informations from the nodes that will help in our functions     
class Node:
    def __init__(self,nodetype,level, attribute,constraint, children,parent, classval,gini):
        self.nodetype=nodetype
        self.level=level
        self.constraint=constraint
        self.attribute=attribute
        self.children=children
        self.parent=parent
        self.classval=classval
        self.gini=gini
        #most of these attributes are calculated in the BuildDecisionTree and used in the print function
#%%
#Here we have our function to calcul the GINI parametrer, that will be used to find the best split in the tree
#We define best split as the parametrer which maximizes the distance between classes, and minimizes the distance intra classes 
def GINI(df):
    gini=1
    total=df["Class"].count()
    class_occur=df["Class"].value_counts()
    for i in df["Class"].unique():
        gini= gini-(class_occur[i]/total)**2
    return gini
#Here we just calcul the gini, for the spliting based on gini we will calcul the weighted average

#%%  
#Here we define our function to Build the tree    
def BuildDeciosionTree(df,minNum,d):
    #We define a empty tree here
    Tree=DecisionTree(np.array([],dtype=Node))
    attributes= df.columns
    
    if (df["Class"].unique().size ==1):
        #Here we have the first case were all our entries are from the same class, so it's a leaf that represets that class
        node=Node("Leaf",0,None,None,np.array([],dtype=Node),None,df["Class"].unique()[0],0)
        Tree.nodes= np.insert(Tree.nodes,Tree.nodes.size,[node])
    elif (df.size<minNum):
        #Here we have the case where our sample is really small, that means that is not statisticly significant, so we attribute an standard value d for the class
        node=Node("Leaf",0,None,None,np.array([],dtype=Node),None,d,GINI(df))
        Tree.nodes= np.insert(Tree.nodes,Tree.nodes.size,[node])
    else:
        #Now we have the case where we should calcul the best parametrer to split
        gini=np.inf
        #In these two fors we decide de parameter and the value for the parameter
        for a in attributes:
            if(a!="Class"):
                for c in df[a].unique():
                    data1=df[df[a]<=c]
                    data2=df[df[a]>c]
                    g=(data1["Class"].count()/df["Class"].count())*GINI(data1)+ (data2["Class"].count()/df["Class"].count())*GINI(data2)
                    #if the gini calculated is the smallest, that should be the parameter
                    if g<=gini:
                        gini=g
                        att=a
                        val=c
                    #print(att,val,GINI(df))
                        
        #this is an extra condition for the case where we have entries with the same value for all atributes, but from different classes.
        #this will stop the algorithm from going into a infinite loop                        
        if(df[att].unique().size==1):
            c=df['Class'].value_counts().idxmax()
            node=Node("Leaf",0,None,None,np.array([],dtype=Node),None,c,GINI(df))
            Tree.nodes= np.insert(Tree.nodes,Tree.nodes.size,[node])
        else:
        #Now we will call the function recursivly 
            #we will create the nodes and the list of values of the attribute
            l=[]
            cons=df[att].unique()
            for c in cons:
                    if(c<=val):
                        l.append(c)
            node=Node("Root",0,att,l,None,None,None,GINI(df))
            
            #now we will call the node1 build function, and will calcul the levels recursivly
            b1=BuildDeciosionTree(df[df[att]<=val],minNum,d)
            b1.nodes[b1.nodes.size-1].parent=node
            for i in b1.nodes:
                i.level+=1
                
            #same for the second node 
            b2=BuildDeciosionTree(df[df[att]>val],minNum,d)
            b2.nodes[b2.nodes.size-1].parent=node
            for i in b2.nodes:
                i.level+=1
            
            #now we will have that the resulting trees will be the junction of the 2 sub-trees calculated
            Tree.nodes= np.insert(Tree.nodes,Tree.nodes.size,b1.nodes)
            if(Tree.nodes[Tree.nodes.size-1].nodetype=="Root"):
                #Here we set the nodetype to the correct value due to the recursivity
                Tree.nodes[Tree.nodes.size-1].nodetype="Intermediate"
            Tree.nodes= np.insert(Tree.nodes,Tree.nodes.size,b2.nodes)
            if(Tree.nodes[Tree.nodes.size-1].nodetype=="Root"):
                Tree.nodes[Tree.nodes.size-1].nodetype="Intermediate"
             
            #we define the children given the node1 and 2 results
            node.children=np.array([b1.nodes[b1.nodes.size-1],b2.nodes[b2.nodes.size-1]],dtype=Node)                    
            #now we add the node to the tree
            Tree.nodes= np.insert(Tree.nodes,Tree.nodes.size,[node])
                          
    return(Tree)
    
#%% 
def printDecisionTree(DecisionTree,file):
    #Here we have the function to print the Tree
    tree = open(file,"w+")
    level=0
    maxlevel=1
    count=0
    #We have to print as BFS, so go through all the levels one by one
    while(level<=maxlevel):
        for node in DecisionTree.nodes:
            if(node.level==level):
                if(count>0):
                    print("*****")
                    tree.write("*****\n")
                if(node.nodetype=="Leaf"): #informations needed if it's a leaf node
                    print(node.nodetype,"\nLevel ",node.level,"\nClass ",node.classval,"\nGini ", node.gini)
                    tree.write(node.nodetype+"\nLevel "+str(node.level)+"\nClass "+str(node.classval)+"\nGini "+str(node.gini)+"\n") 
                    count+=1
                else:#informations needed in it's a non leaf node
                    s = " ".join(map(str, node.constraint))
                    print(node.nodetype,"\nLevel ",node.level,"\nFeature ",node.attribute,*node.constraint,"\nGini ", node.gini)
                    tree.write(node.nodetype + "\nLevel " + str(node.level) + "\nFeature " + node.attribute +"  "+ s + "\nGini " + str(node.gini) + "\n")
                    count+=1
            elif(node.level>maxlevel):
                maxlevel=node.level
        level+=1
        count=0
        print("\n")
        tree.write("\n")
    tree.close()
    return()

    
#%%
def generalizationError(df,DecisionTree,alpha):
    #our generalization error should be the test erro plus alpha times the number of leafs
    num_leafs=0
    erro=0
    e = open("generalization_error.txt","w+")
    for node in DecisionTree.nodes: 
        if(node.nodetype=="Leaf"):
            num_leafs+=1 #here we count the number of leafs 
            x=node
            constrains=[]
            class_v=node.classval
            data=df
            #We go um the tree to find the constrains to a given leaf
            while(x.nodetype!="Root"):
                c=x
                x=x.parent
                if(c==x.children[0]):
                    constrains.append([x.attribute,x.constraint,0])
                else:
                    constrains.append([x.attribute,x.constraint,1])
            #Here we find the set of splited data that corresponds to that leaf
            for i in constrains:
                #print(i)
                m=i[1]
                if(i[2]==0):
                    data=data[data[i[0]] <= m[len(i[1])-1]]
                else:
                    data=data[data[i[0]] > m[len(i[1])-1]]
                #print(data)
            data=data[data["Class"]!=class_v] #If the class of an entry in not the same from the leaf class, we have a classification error
            #print(data["Class"].size)
            erro=erro+data["Class"].size #getting the number of errors
    gen_erro= erro+alpha*num_leafs
    
    e.write(str(gen_erro))
    e.close()
    
    return(gen_erro)
#%%
def pruneTree(df,DecisionTree,alpha,minNum,d):
    
    max_level=0
    for node in DecisionTree.nodes:
        if(node.level>max_level):
            max_level=node.level
    
    for i in range (max_level-1,0,-1):
        #we go in the tree from the bottom up
        check=[]
        for node in DecisionTree.nodes:
            #We won't prune leaf nodes, nor the root node
            if(node.nodetype =="Intermediate" and node.level==i):
                check.append(node)
            
                
        for node in check:
            prior_erro=generalizationError(df,DecisionTree,alpha)
            s=True
            #Here we "remove" the leafs that we will prune
            for child in node.children:
                if(child.nodetype!="Leaf"):
                    s=False
            if(s):
                for child in node.children:
                    if(child.nodetype=="Leaf"):
                        child.nodetype="?"
                    else:
                        s=False
                
                x=node
                constrains=[]
                data=df
                #we turn the node that we are pruning into a tree
                node.nodetype="Leaf"
    
                #here we will find the data that correspond to the constrains to find the class
                while(x.nodetype!="Root"):
                    c=x
                    x=x.parent
                    if(c==x.children[0]):
                        constrains.append([x.attribute,x.constraint,0])
                    else:
                        constrains.append([x.attribute,x.constraint,1])
                for i in constrains:
                    m=i[1]
                    if(i[2]==0):
                        data=data[data[i[0]] <= m[len(i[1])-1]]
                    else:
                        data=data[data[i[0]] > m[len(i[1])-1]] 
                        
                u=data["Class"].unique()
                #Here we define the classes given each special case
                if(u.size==1 and df.size>=minNum):
                    node.classval=u[0]
                elif(df.size < minNum):
                    node.classval=d
                else:
                    c=df['Class'].value_counts().idxmax()
                    node.classval=c
            
                after_erro=generalizationError(df,DecisionTree,alpha)
                #print(prior_erro,after_erro)
                if(prior_erro >= after_erro):
                    #If the error is smaller, we indeed turn it into a leaf
                    node.constraint=None
                    node.attribute=None
                    node.children=np.array([],dtype=Node)
                    
                else:
                    #else we undo
                    node.classval=None
                    node.nodetype="Intermediate"
                    for child in node.children:
                        child.nodetype="Leaf"
    dels=[]   
    #We delete the leafs from the node that we pruned             
    for node in DecisionTree.nodes:
        if(node.nodetype=="?"):
            index = np.argwhere(DecisionTree.nodes==node)
            dels.append(index)
    for i in range (len(dels)-1,-1,-1):
        DecisionTree.nodes = np.delete(DecisionTree.nodes, dels[i])   
                
    return DecisionTree
#%%
df = pd.read_csv("data.csv")
df.rename(columns={'Survived':'Class'}, inplace=True)
#df=df[:20]
t=BuildDeciosionTree(df,5,0)
printDecisionTree(t,"output_tree.txt")
g=generalizationError(df,t,0.5)
print(g,"\n\n\nPrune\n")
p=pruneTree(df,t,0.5,5,0)
printDecisionTree(p,"postpruned tree.txt")
g=generalizationError(df,p,0.5)
print(g)
#%%