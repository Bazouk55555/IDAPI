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
#
# We calculate the occurences for each states and we divide by the total of occurences for all the states
#
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
#
# We calculate the occurences for each state of varC knowing each state of varP and we divide by the total 
# of occurences for all the states of VarC knowing each states of VarP
#
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
#
# We calculate the occurences for each states of varCol and varRow and we divide by the total of occurences for 
# all the states of varRow and varC 
#
    for i in range(0,len(theData)):
	jPT[theData[i,varRow],theData[i,varCol]] = jPT[theData[i,varRow],theData[i,varCol]]+1
    jPT/=len(theData)
# end of coursework 1 task 3
    return jPT
#
# Function to convert a joint probability table to a conditional probability table
def JPT2CPT(aJPT):
#Coursework 1 task 4 should be inserted here
#
# We have to normalise the result of question 3 which is done by dividing by the vector alpha 
#
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
#
# Here, we applied the formula: P(D|S1&S2..&Sn) = αP(D)P(S1|D)P(S2|D)· · · P(Sn|D)
#
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
   

# end of coursework 2 task 1
    return mi
#
# construct a dependency matrix for all the variables
def DependencyMatrix(theData, noVariables, noStates):
    MIMatrix = zeros((noVariables,noVariables))
# Coursework 2 task 2 should be inserted here
    

# end of coursework 2 task 2
    return MIMatrix
# Function to compute an ordered list of dependencies 
def DependencyList(depMatrix):
    depList=[]
# Coursework 2 task 3 should be inserted here
    

# end of coursework 2 task 3
    return array(depList2)
#
# Functions implementing the spanning tree algorithm
# Coursework 2 task 4

def SpanningTreeAlgorithm(depList, noVariables):
    spanningTree = []
  
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
# main program part for Coursework 1
#

noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("Neurones.txt")
theData = array(datain)
AppendString("IDAPIResults01.txt","Coursework One Results by Adrien Boukobza aeb115")


# Question 1

AppendString("IDAPIResults01.txt","") #blank line
AppendString("IDAPIResults01.txt","1.The prior probability of node 0")
prior = Prior(theData, 0, noStates)
AppendList("IDAPIResults01.txt", prior)


# Question 2

AppendString("IDAPIResults01.txt","2. Matrix P(2|0) calculated from the data.")
CPT_2given0 = CPT(theData, 2, 0, noStates)
AppendArray("IDAPIResults01.txt", CPT_2given0)
AppendString("IDAPIResults01.txt","")


# Question 3 

AppendString("IDAPIResults01.txt","3. P(2&0) calculated from the data")
AppendString("IDAPIResults01.txt","")
jointDis = JPT(theData, 2, 0, noStates)
AppendArray("IDAPIResults01.txt", jointDis)
AppendString("IDAPIResults01.txt","")


# Question 4

AppendString("IDAPIResults01.txt","4. P(2|0) calculated from the joint probability matrix P(2&0).")
AppendString("IDAPIResults01.txt","")
conDis = JPT2CPT(jointDis)
AppendArray("IDAPIResults01.txt", conDis)
AppendString("IDAPIResults01.txt","")


# Question 5

# the conditional probabilities are: 
cpt1 = CPT(theData, 1, 0, noStates)
cpt2 = CPT(theData, 2, 0, noStates)
cpt3 = CPT(theData, 3, 0, noStates)
cpt4 = CPT(theData, 4, 0, noStates)
cpt5 = CPT(theData, 5, 0, noStates)
naiveBayes = [prior, cpt1, cpt2, cpt3, cpt4, cpt5]
 
AppendString("IDAPIResults01.txt","5. Output of Query[4,0,0,0,5]")
query1 = Query([4,0,0,0,5], naiveBayes)

AppendList("IDAPIResults01.txt", query1)

AppendString("IDAPIResults01.txt","")
AppendString("IDAPIResults01.txt","5. Output of Query[6,5,2,5,5]")
query2 = Query([6,5,2,5,5], naiveBayes)
AppendList("IDAPIResults01.txt", query2)



