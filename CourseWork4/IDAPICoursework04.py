#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python 
from IDAPICourseworkLibrary import *
from numpy import *

#
# Coursework 4 begins here
#
def Mean(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    mean = []
    # Coursework 4 task 1 begins here
    #
    # The mean is calculated by summing all the rows in each column 
    # and dividing them by the number of rows
    #
    for i in range (0,theData.shape[1]):
	a=0
	for j in range (0,theData.shape[0]):
    		a=a+realData[j][i]
	a=a/theData.shape[0]
	mean.append(a)

    # Coursework 4 task 1 ends here
    return array(mean)


def Covariance(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    covar = zeros((noVariables, noVariables), float)
    
    # Coursework 4 task 2 begins here
    #
    # The covariance is calculated according to the formula :
    # Transpose(RealData-Mean)*(RealData-Mean)/N-1
    #
    mean=Mean(theData)
    for i in range (0,theData.shape[0]):
	realData[i]=realData[i]-mean
    covar=dot(realData.transpose(),realData)/(realData.shape[0]-1)
    # Coursework 4 task 2 ends here
    return covar

def CreateEigenfaceFiles(theBasis):
    # Coursework 4 task 3 begins here
    for i in range (0,theBasis.shape[0]):
	filename= "PrincipalComponent"+ str(i) + ".jpg"
	SaveEigenface(theBasis[i],filename)    

    # Coursework 4 task 3 ends here

def ProjectFace(theBasis, theMean, theFaceImage):
    magnitudes = []
    # Coursework 4 task 4 begins here
    #
    # The magnitudes are calculated according to the formula :
    # (ImageRead-Meanread)*Transpose(theBasis)
    #
    image_read=ReadOneImage(theFaceImage)
    image_centered=subtract(image_read,theMean)
    a=dot(image_centered,theBasis.transpose()) 
    magnitudes=a[:]
    # Coursework 4 task 4 ends here
    return array(magnitudes)

def CreatePartialReconstructions(aBasis, aMean, componentMags):
    # Coursework 4 task 5 begins here
    filename="Moyenne.jpg"
    SaveEigenface(array(aMean),filename) 
    for i in range(0,len(componentMags)):
	aMean=aMean+componentMags[i]*aBasis[i]
	filename="new_image_"+str(i)+".jpg"
    	SaveEigenface(aMean,filename) 
    
	
    # Coursework 4 task 5 ends here

def PrincipalComponents(theData):
    # Coursework 4 task 3 begins here
    # The goal is to calculate the Matrix:
    # Transpose(U)*eigenvectors(U*Transpose(U))
    #
    D=array(theData)
    mean=Mean(D)
    # theData is centered
    U= D-mean
    # the matrix of the eigenvectors of U*Transpose(U) are calculated:
    UUt=dot(U,U.transpose())
    eigenva, eigenvecUUt=numpy.linalg.eig(UUt)
    # The array Transpose(U)*eigenvectors(U*Transpose(U)) is calculated:
    eigenvec=dot(U.transpose(),eigenvecUUt)
    # This array is transposed
    eigenvec=eigenvec.transpose()
    # it is then normalized:
    for i in range(0,eigenvec.shape[0]):
	eigenvec[i]=eigenvec[i]/linalg.norm(eigenvec[i])
    # it is then classified according to the decreasing order of the eigenvalues:
    i=0
    buffer=zeros((1,eigenvec[0].shape[0]),float)
    while i<len(eigenva):
    	max=eigenva[i]
	rank=-1
    	for j in range(i,len(eigenva)):
		if eigenva[j]>max:
			max=eigenva[j]
			rank=j	
	if rank!=-1:
		buffervec=copy(eigenvec[i])
		bufferva=eigenva[i]
	      	eigenvec[i]=eigenvec[rank]
		eigenva[i]=eigenva[rank]
		eigenvec[rank]=buffervec
		eigenva[rank]=bufferva	
	i=i+1	
    
    # Coursework 4 task 6 ends here
    return eigenvec

#
# main program part for Coursework 2
#
 
noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
theData = array(datain)
AppendString("IDAPIResults04.txt","1.Coursework four Results by Adrien Boukobza aeb115")
AppendString("IDAPIResults04.txt","") 

#
# Qu 1
#

AppendString("IDAPIResults04.txt","2.The mean vector Hepatitis C data set is :") 
AppendString("IDAPIResults04.txt","") 
mean= Mean(theData)
AppendList("IDAPIResults04.txt",mean)
AppendString("IDAPIResults04.txt","") 

#
# Qu 2
#

AppendString("IDAPIResults04.txt","3.The covariance matrix of the Hepatitis C data set :") 
AppendString("IDAPIResults04.txt","") 
cov=Covariance(theData)
AppendArray("IDAPIResults04.txt",cov)
AppendString("IDAPIResults04.txt","") 

#
# Qu 3
#
AppendString("IDAPIResults04.txt","The images of each component of the basis :")
AppendString("IDAPIResults04.txt","") 
theBasis=ReadEigenfaceBasis()
CreateEigenfaceFiles(theBasis)
#
# Qu 4
#

AppendString("IDAPIResults04.txt","4.The component magnitudes for image “c.pgm” in the principal component basis used in task 4.4 :")
AppendString("IDAPIResults04.txt","") 
aMean=ReadOneImage("MeanImage.jpg")
magnitudes=ProjectFace(theBasis,aMean,"c.pgm")
AppendList("IDAPIResults04.txt",magnitudes)
AppendString("IDAPIResults04.txt","") 

#
# Qu5
#
AppendString("IDAPIResults04.txt","The images reconstructed are :")
AppendString("IDAPIResults04.txt","")
images_created=CreatePartialReconstructions(theBasis, aMean, magnitudes)

#
# Qu6
#
AppendString("IDAPIResults04.txt","Question6 :")
AppendString("IDAPIResults04.txt","")
theData=ReadImages()
p=PrincipalComponents(theData)
AppendString("IDAPIResults04.txt","The images of each component of the new basis :")
AppendString("IDAPIResults04.txt","") 
CreateEigenfaceFiles(p)
AppendString("IDAPIResults04.txt","The component magnitudes for image “c.pgm” in the principal component basis used in task 4.6 :")
AppendString("IDAPIResults04.txt","") 
aMean=Mean(array(theData))
magnitudes2=ProjectFace(p,aMean,"c.pgm")
AppendList("IDAPIResults04.txt",magnitudes2) 
AppendString("IDAPIResults04.txt","")
AppendString("IDAPIResults04.txt","The images reconstructed are :")
AppendString("IDAPIResults04.txt","")
images_created_new_basis=CreatePartialReconstructions(p, aMean, magnitudes2)



