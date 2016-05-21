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
# Coursework 1 task 2 should be inserted here
    for k in range (0,noStates[varP]):
		alpha=0
    		for j in range (0,noStates[varC]):
    			for i in range (0, len(theData)):
				if theData[i,varC]==j and theData[i,varP]==k:
					alpha=alpha+1							
					cPT[j,k]=cPT[j,k]+1
		for j in range (0,noStates[varC]):
			if alpha!=0:
				cPT[j,k]=cPT[j,k]/alpha
			else:
				cPT[j,k]=1.0/float(noStates[varC])	
		
		
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
# Function to compute a CPT with multiple parents from the data set
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT_2(theData, child, parent1, parent2, noStates):
    cPT = zeros([noStates[child],noStates[parent1],noStates[parent2]], float )
# Coursework 3 task 1 should be inserted here
#
# We calculate the occurences for each state of the child knowing each state of the parents parent1 and parent2 
# and we divide by the total of occurences for all the states of the child 
# knowing each state of the parents. When no occurence occures for some states of the parents, we put all
# the value to 1/noStates[child]
#
    for i in range(0,noStates[parent1]):
	for j in range(0,noStates[parent2]):
		alpha=0
		for k in range(0,noStates[child]):
			for l in range(0,len(theData)):
				if theData[l][child]==k and theData[l][parent1]==i and theData[l][parent2]==j:
					cPT[k,i,j]=cPT[k,i,j]+1
					alpha=alpha+1
		for k in range (0,noStates[child]):
			if alpha!=0:
				cPT[k,i,j]=cPT[k,i,j]/alpha
			else:
				cPT[k,i,j]=1.0/float(noStates[child])	

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
#
# We are doing like the example above with the hapatitis network 
#
def HepatitisNetwork(theData, noStates):
    arcList = [[0],[1],[2,0],[3,4],[4,1],[5,4],[6,1],[7,1,0],[8,7]]
    cpt0 = Prior(theData, 0, noStates)
    cpt1 = Prior(theData, 1, noStates)
    cpt2 = CPT(theData, 2, 0, noStates)
    cpt3 = CPT(theData, 3, 4, noStates)
    cpt4 = CPT(theData, 4, 1, noStates)
    cpt5 = CPT(theData, 5, 4, noStates)
    cpt6 = CPT(theData, 6, 1, noStates)
    cpt7 = CPT_2(theData, 7, 1, 0, noStates)
    cpt8 = CPT(theData, 8, 7, noStates)
    cptList = [cpt0, cpt1, cpt2, cpt3, cpt4, cpt5, cpt6, cpt7,cpt8]
    return arcList, cptList
# end of coursework 3 task 2
#
# Function to calculate the MDL size of a Bayesian Network
def MDLSize(arcList, cptList, noDataPoints, noStates):
    mdlSize = 0.0
    	
# Coursework 3 task 3 begins here
#
# We apply the formula : Size(Bn) = |Bn|(log2N)/2 
#
    for i in range(0,len(arcList)):
	variable_useful=noStates[arcList[i][0]]-1
	for j in range(1,len(arcList[i])):
		variable_useful=variable_useful*noStates[arcList[i][j]]
	mdlSize=mdlSize+variable_useful
    mdlSize=mdlSize*math.log(noDataPoints,2)/2

# Coursework 3 task 3 ends here 
    return mdlSize 
#
# Function to calculate the joint probability of a single data point in a Network
def JointProbability(dataPoint, arcList, cptList):
    jP = 1.0
# Coursework 3 task 4 begins here
#
# The join probability is the multiplication between all the conditional probabilities
# and all the prior probabilities. The dimension of cptList is different according to 
# the length of arcList so we need to separate the case.
#
    for i in range(0,len(cptList)):
	if len(arcList[i])==1:
		jp=jp*cptList[i][datapoint[i]]
	elif len(arcList[i])==2:
		jp=jp*cptList[i][datapoint[arcList[i][0]]][datapoint[arcList[i][1]]]
	else:
		jp=jp*cptList[i][datapoint[arcList[i][0]]][datapoint[arcList[i][1]]][datapoint[arcList[i][2]]]

# Coursework 3 task 4 ends here 
    return jP
#
# Function to calculate the MDLAccuracy from a data set
def MDLAccuracy(theData, arcList, cptList):
    mdlAccuracy=0
# Coursework 3 task 5 begins here
#
# We apply the formula : Size(Bn) = log2(P(Ds|Bn)) with P(Ds|Bn)= π(P(xi)P(yi)).
# The dimension of cptList is different according to arcList so we separate into 3 case depending of the 
# length of cpTlist.
#
    for i in range(0,len(theData)):
	p=1
	for j in range(0,len(cptList)):
		if len(arcList[j])==1:
			p=p*cptList[j][theData[i][arcList[j][0]]]
		if len(arcList[j])==2:
			p=p*cptList[j][theData[i][arcList[j][0]]][theData[i][arcList[j][1]]]
		if len(arcList[j])==3:
			p=p*cptList[j][theData[i][arcList[j][0]]][theData[i][arcList[j][1]]][theData[i][arcList[j][2]]]
	mdlAccuracy=mdlAccuracy+math.log(p,2)
# Coursework 3 task 5 ends here 
    return mdlAccuracy


# Coursework 3 : computation of the model score
#
# We apply the formula : MDLScore = ModelSize - ModelAccuracy
#

def MDLScore(mdlSize,mdlAccuracy):
	mdlScore=mdlSize-mdlAccuracy
	return mdlScore

# CourseWork 3 task 6
#
# We delete the arcs one by one (arcList[i].remove(arcList[i][1])) and we calculate the mdlscore.
# If it is better than the previous one calculated, then it becomes the next best_score, we add the node # # deleted and 
# delete the next one. Again, for a reason of compatibility with cptList, we need to separate the case when len(arcList)==2 and
# len(arcList)==3.
#
def BestScoringNetwork(theData,arcList,cptList,noStates):
	#Initialize the best score
	best_score=MDLScore(MDLSize(arcList, cptList, len(theData), noStates),MDLAccuracy(theData, arcList, cptList))
	arc_removed= 0

	for i in range(0,len(arcList)):
		
		# When the list of arcList has two elements
		if len(arcList[i])==2:
			#We removed the second element of arclist[i]
			buffer1=arcList[i][1]
			buffer2=cptList[i]
			arcList[i].remove(arcList[i][1])
			cptList[i]=Prior(theData, arcList[i][0], noStates)
			mdlscore=MDLScore(MDLSize(arcList, cptList, len(theData), noStates),MDLAccuracy(theData, arcList, cptList))
			if mdlscore<best_score:
				best_score= mdlscore
				arc_removed= [arcList[i][0],buffer1]
			#We put back the second element of arclist[i]
			arcList[i].insert(i,buffer1)
			cptList[i]=buffer2

		# When the list of arcList has three elements
		if len(arcList[i])==3:  

			#We removed the second element of arclist[i]
			buffer1=arcList[i][1]
			buffer2=cptList[i]
			arcList[i].remove(arcList[i][1])
			cptList[i]=CPT(theData, arcList[i][0], arcList[i][1], noStates)
			mdlscore=MDLScore(MDLSize(arcList, cptList, len(theData), noStates),MDLAccuracy(theData, arcList, cptList))
			if mdlscore<best_score:
				best_score= mdlscore
				arc_removed= [arcList[i][0],buffer1]
			#We put back the second element of arclist[i]
			arcList[i].insert(1,buffer1)
			cptList[i]=buffer2

			#We removed the third element of arclist[i]
			buffer1=arcList[i][2]
			buffer2=cptList[i]
			arcList[i].remove(arcList[i][2])
			cptList[i]=CPT(theData, arcList[i][0], arcList[i][1], noStates)
			mdlscore=MDLScore(MDLSize(arcList, cptList, len(theData), noStates),MDLAccuracy(theData, arcList, cptList))
			if mdlscore<best_score:
				best_score= mdlscore
				arc_removed= [arcList[i][0],buffer1]
			#We put back the third element of arclist[i]
			arcList[i].append(buffer1)
			cptList[i]=buffer2

	return best_score,arc_removed

#
# End of coursework 3
#
# Coursework 4 begins here
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
# main program part for Coursework 3
#

noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
theData = array(datain)
AppendString("IDAPIResults03.txt","Coursework trois Results by Adrien Boukobza aeb115")
AppendString("IDAPIResults03.txt","") #blank line

# Computation of cpt_2, arcList and cptList
cpt_2=CPT_2(theData, 7, 1, 0, noStates)
arcList, cptList=HepatitisNetwork(theData, noStates)

#2
AppendString("IDAPIResults03.txt", "2.The MDLSize of the network for Hepatitis C data set" )
mdlsize=MDLSize(arcList, cptList, len(theData), noStates)
AppendString("IDAPIResults03.txt",mdlsize)
AppendString("IDAPIResults03.txt","")#blank line

#3
AppendString("IDAPIResults03.txt", "3.The MDLAccuracy of the network for Hepatitis C data set" )
mdlaccuracy=MDLAccuracy(theData, arcList, cptList)
AppendString("IDAPIResults03.txt",mdlaccuracy)
AppendString("IDAPIResults03.txt","")#blank line

#4
AppendString("IDAPIResults03.txt", "4.The MDLScore of the network for Hepatitis C data set" )
mdlscore=MDLScore(mdlsize,mdlaccuracy)
AppendString("IDAPIResults03.txt",mdlscore)
AppendString("IDAPIResults03.txt","")#blank line

#5
AppendString("IDAPIResults03.txt", "5.The score of the best network with one arc removed" )
best_score, arc_removed=BestScoringNetwork(theData,arcList,cptList,noStates)
AppendString("IDAPIResults03.txt","a. best_score : ")
AppendString("IDAPIResults03.txt",best_score)
AppendString("IDAPIResults03.txt","b. arc_removed : ")
AppendString("IDAPIResults03.txt",arc_removed)
AppendString("IDAPIResults03.txt","")#blank line








