#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python 
from IDAPICourseworkLibrary import *
from numpy import *
#
# Coursework 1 begins here
#
# Function to compute the prior distribution of the variable root from the data set
def Prior(theData, root, noStates):
    prior = zeros((noStates[root]), float )
# Coursework 1 task 1 should be inserted here
    for i in range(0,len(theData)):
      prior[theData[i,root]] = prior[theData[i,root]]+1
    prior/=len(theData)
# end of Coursework 1 task 1
    return prior
# Function to compute a CPT with parent node varP and xchild node varC from the data array
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT(theData, varC, varP, noStates):
    cPT = zeros((noStates[varC], noStates[varP]), float )
# Coursework 1 task 2 should be inserte4d here
    for k in range (0,noStates[varP]):
		alpha=0
    		for j in range (0,noStates[varC]):
    			for i in range (0, len(theData)):
				if theData[i,varC]==j and theData[i,varP]==k:
					alpha=alpha+1							
					cPT[j,k]=cPT[j,k]+1
		for j in range (0,noStates[varC]):
			cPT[j,k]=cPT[j,k]/alpha
		
		
		
# end of coursework 1 task 2
    return cPT
# Function to calculate the joint probability table of two variables in the data set
def JPT(theData, varRow, varCol, noStates):
    jPT = zeros((noStates[varRow], noStates[varCol]), float )
#Coursework 1 task 3 should be inserted here 
    for i in range(0,len(theData)):
	jPT[theData[i,varRow],theData[i,varCol]] = jPT[theData[i,varRow],theData[i,varCol]]+1
    jPT/=len(theData)
# end of coursework 1 task 3
    return jPT
#
# Function to convert a joint probability table to a conditional probability table
def JPT2CPT(aJPT):
#Coursework 1 task 4 should be inserted here
	alpha=zeros(aJPT.shape[1],float) 
	for j in range (0,aJPT.shape[1]):	 
		for i in range (0,aJPT.shape[0]):
			alpha[j]=alpha[j]+aJPT[i][j]

        aJPT=aJPT/alpha
# coursework 1 taks 4 ends here
        return aJPT

#
# Function to query a naive Bayesian network
def Query(theQuery, naiveBayes): 
    rootPdf = zeros((naiveBayes[0].shape[0]), float)
# Coursework 1 task 5 should be inserted here
    alpha=0
    for i in range(0,naiveBayes[0].shape[0]):
	for j in range (0,len(naiveBayes)):
		if j==0:
			rootPdf[i]=naiveBayes[j][i]
		else:
			rootPdf[i]=naiveBayes[j][theQuery[j-1]][i]*rootPdf[i]

    for k in range (0,naiveBayes[0].shape[0]):
	alpha=alpha+rootPdf[k]
	

    
    rootPdf=rootPdf/alpha
# end of coursework 1 task 5
    return rootPdf
#
# End of Coursework 1
#
# Coursework 2 begins here
#
# Calculate the mutual information from the joint probability table of two variables
def MutualInformation(jP):
    mi=0.0
    
# Coursework 2 task 1 should be inserted here
#
#Application of the formula Dep(A, B) = sum P(ai&bj )log2((P(ai&bj )/(P(ai)P(bj )))
#
    for i in range(0, jP.shape[0]):
	pi=0
	for k in range(0,jP.shape[1]):
		pi=pi+jP[i][k]
	for j in range(0,jP.shape[1]):
		pj=0
		for k in range(0,jP.shape[0]):
			pj=pj+jP[k][j]
		if jP[i][j]!=0 and pi!=0 and pj!=0:					
			mi=mi+jP[i][j]*math.log(jP[i][j]/(pj*pi),2)

# end of coursework 2 task 1
    return mi
#
# construct a dependency matrix for all the variables
def DependencyMatrix(theData, noVariables, noStates):
    MIMatrix = zeros((noVariables,noVariables))
# Coursework 2 task 2 should be inserted here
#
# We apply the function of the question to each variable and create a matrix for the dependency between each variable.
# The dependency between a variable and itself will be set to 0.  
#
    for i in range (0,noVariables):
	for j in range (0,noVariables):
		if i!=j:
			MIMatrix[i][j]=MutualInformation(JPT(theData,i,j,noStates))   
		else:
			MIMatrix[i][j]=0

# end of coursework 2 task 2
    return MIMatrix
# Function to compute an ordered list of dependencies 
def DependencyList(depMatrix):
    depList=[]
# Coursework 2 task 3 should be inserted here
#
# We put in a list the value of the dependency and the two node associated 
# by taking the values up to the diagonal of the matrix. Then we ordered this list
# according to the values of the dependecy
#
    for i in range (0,len(depMatrix)):
	for j in range (i+1,len(depMatrix)):
		if i!=j:
			depList.append([depMatrix[i][j],i,j])

    depList.sort(cmp=lambda x,y: cmp(x[0], y[0]))
    depList.reverse()
# end of coursework 2 task 3                          
    return array(depList)
#
# Functions implementing the spanning tree algorithm
# Coursework 2 task 4
#
# Exponation of the function searchLoop:
# This function searches if there is a loop in the tree.
# As a parameter, we have a list of nodes connected. We will separate this list
# in two lists in order to have two lists containing each half of all the nodes. 
# The same position of each one of the list indicates that 
# the nodes in these positions are connected. We will then look for a node that 
# appeares less than two times in theses two lists. It means it is connected 
# only one time to another node so it is not a part of a loop. We delete it and 
# apply again the function searchLoop to the new list. At the end, if there is no loop,
# the size of the last list should be two. If there is a loop, all the resulting node 
# should be connected at least twice to nodes
#
def searchLoop(list):
	if len(list)<3:
		return False
	node_left=[]
	node_right=[]
	list_bis=list[:]
	for i in range (0,len(list)):
		node_left.append(list[i][0])
		node_right.append(list[i][1])
	for i in range(0,len(node_left)):
		counter=0
		for j in range(0,len(node_left)):
			if node_left[i]==node_left[j]:
				counter=counter+1
			if node_left[i]==node_right[j]:
				counter=counter+1	
		if counter<2:
			list_bis.remove(list_bis[i])
			return searchLoop(list_bis)
	
	for i in range(0,len(node_left)):
		counter=0
		for j in range(0,len(node_left)):
			if node_right[i]==node_right[j]:
				counter=counter+1
			if node_right[i]==node_left[j]:
				counter=counter+1
		if counter<2:
			list_bis.remove(list_bis[i])
			return searchLoop(list_bis)
			
	return True
		
#
# Exponation of the function SpanningTree:
# From the array ordered by dependency we add each time
# a new connection between two nodes (which already exist in the actual tree or not).
# We apply then searchLoop to check if there is a loop in this new tree. 
# If there is a loop, we delete the new connection and continue the processus with the next connection
# If there is not a loop, we continue the processus with the next connection
#

def SpanningTreeAlgorithm(depList, noVariables):
    spanningTree = []
    for i in range(0,len(depList)):
    	spanningTree.append(depList[i])
    	if searchLoop(spanningTree):
		spanningTree.remove(depList[i])
		
    return array(spanningTree)
#	
# End of coursework 2
#
# Coursework 3 begins here
#
# Function to compute a CPT with multiple parents from he data set
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT_2(theData, child, parent1, parent2, noStates):
    cPT = zeros([noStates[child],noStates[parent1],noStates[parent2]], float )
# Coursework 3 task 1 should be inserted here
   

# End of Coursework 3 task 1           
    return cPT
#
# Definition of a Bayesian Network
def ExampleBayesianNetwork(theData, noStates):
    arcList = [[0],[1],[2,0],[3,2,1],[4,3],[5,3]]
    cpt0 = Prior(theData, 0, noStates)
    cpt1 = Prior(theData, 1, noStates)
    cpt2 = CPT(theData, 2, 0, noStates)
    cpt3 = CPT_2(theData, 3, 2, 1, noStates)
    cpt4 = CPT(theData, 4, 3, noStates)
    cpt5 = CPT(theData, 5, 3, noStates)
    cptList = [cpt0, cpt1, cpt2, cpt3, cpt4, cpt5]
    return arcList, cptList
# Coursework 3 task 2 begins here

# end of coursework 3 task 2
#
# Function to calculate the MDL size of a Bayesian Network
def MDLSize(arcList, cptList, noDataPoints, noStates):
    mdlSize = 0.0
# Coursework 3 task 3 begins here


# Coursework 3 task 3 ends here 
    return mdlSize 
#
# Function to calculate the joint probability of a single data point in a Network
def JointProbability(dataPoint, arcList, cptList):
    jP = 1.0
# Coursework 3 task 4 begins here


# Coursework 3 task 4 ends here 
    return jP
#
# Function to calculate the MDLAccuracy from a data set
def MDLAccuracy(theData, arcList, cptList):
    mdlAccuracy=0
# Coursework 3 task 5 begins here


# Coursework 3 task 5 ends here 
    return mdlAccuracy
#
# End of coursework 2
#
# Coursework 3 begins here
#
def Mean(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    mean = []
    # Coursework 4 task 1 begins here
	


    # Coursework 4 task 1 ends here
    return array(mean)


def Covariance(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    covar = zeros((noVariables, noVariables), float)
    # Coursework 4 task 2 begins here


    # Coursework 4 task 2 ends here
    return covar
def CreateEigenfaceFiles(theBasis):
    adummystatement = 0 #delete this when you do the coursework
    # Coursework 4 task 3 begins here

    # Coursework 4 task 3 ends here

def ProjectFace(theBasis, theMean, theFaceImage):
    magnitudes = []
    # Coursework 4 task 4 begins here

    # Coursework 4 task 4 ends here
    return array(magnitudes)

def CreatePartialReconstructions(aBasis, aMean, componentMags):
    adummystatement = 0  #delete this when you do the coursework
    # Coursework 4 task 5 begins here

    # Coursework 4 task 5 ends here

def PrincipalComponents(theData):
    orthoPhi = []
    # Coursework 4 task 3 begins here
    # The first part is almost identical to the above Covariance function, but because the
    # data has so many variables you need to use the Kohonen Lowe method described in lecture 15
    # The output should be a list of the principal components normalised and sorted in descending 
    # order of their eignevalues magnitudes

    
    # Coursework 4 task 6 ends here
    return array(orthoPhi)

#
# main program part for Coursework 2
#
 
noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
theData = array(datain)
AppendString("IDAPIResults02.txt","Coursework Two Results by Adrien Boukobza aeb115")
AppendString("IDAPIResults02.txt","") 

# Question 2.1

AppendString("IDAPIResults02.txt","Qu1.The mutual information for the HepatitisC data set")
jP = JPT(theData, 1, 2, noStates)
mi = MutualInformation(jP)
AppendString("IDAPIResults02.txt",mi)
AppendString("IDAPIResults02.txt","") 

# Question 2.2

AppendString("IDAPIResults02.txt","Qu2.Dependency matrix for the HepatitisC data set")
depMat = DependencyMatrix(theData, noVariables, noStates)
AppendArray("IDAPIResults02.txt", depMat)
AppendString("IDAPIResults02.txt","")

# Question 2.3

AppendString("IDAPIResults02.txt","Qu3.Dependency list for the HepatitisC data set")
listMat=DependencyList(depMat)
AppendArray("IDAPIResults02.txt", listMat)
AppendString("IDAPIResults02.txt","")

# Question 2.4

AppendString("IDAPIResults02.txt","Qu4.Spanning tree found for the HepatitisC data set")
list1=listMat.tolist()
for i in range(0,len(list1)):
	list1[i].remove(list1[i][0])

listMat2=SpanningTreeAlgorithm(list1, noVariables)

AppendArray("IDAPIResults02.txt", listMat2)






