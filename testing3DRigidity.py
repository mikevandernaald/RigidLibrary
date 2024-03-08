import numpy as np
import sys
import Configuration_altered as CF
import Pebbles as PB
import itertools
from matplotlib import pyplot
#import Hessian_3D

"""
Author:  Mike van der Naald and Finn Braaten
In this file we aim to test various pebble games and the 3D frictional hessian approach on various lattice packings with
different characteristics.  In the first portion we define helper functions that we will use in later portions.  In the
second portion we give the positions and radii of various lattices so that we can test the rigidity algorithms in the 
last portion.  Finally, in the third portion we run the rigidity algorithms on the aforementioned lattices and plot
the results.
"""

def rigidClusterDataGenerator(pebbleObj, returnClusterIDs=True):
    """
    This is a helper function that takes pebble objects and returns the size of the rigid clusters, the number of bonds
    in the cluster, as well as the contacts participating in the cluster (if returnClusterIDs=True).
    """

    # Load in all the relevant data.  The first column has the ID of the cluster and the second and third rows tell you the particles i and j which are in that cluster ID
    clusterIDHolder = np.transpose(np.vstack([pebbleObj.cluster, pebbleObj.Ifull, pebbleObj.Jfull]))

    # Remove all rows that have -1 in the first column.  Those are contacts that are not participating in a cluster
    clusterIDHolder = clusterIDHolder[clusterIDHolder[:, 0] != -1]

    numClusters = len(np.unique(clusterIDHolder[:, 0]))

    clusterSizes = np.zeros(numClusters)
    numBondsPerCluster = np.zeros(numClusters)
    if returnClusterIDs == True:
        clusterID = [0] * numClusters

    counter = 0
    for i in np.unique(clusterIDHolder[:, 0]):
        currentCluster = clusterIDHolder[clusterIDHolder[:, 0] == i][:, 1:]

        currentCluster = np.unique(np.sort(currentCluster, axis=1), axis=0)

        (numBonds, _) = np.shape(currentCluster)

        numBondsPerCluster[counter] = numBonds
        clusterSizes[counter] = len(np.unique(currentCluster.flatten()))
        if len(np.unique(currentCluster.flatten())) == 0:
            breakpoint()
        if returnClusterIDs == True:
            clusterID[counter] = currentCluster

        counter = counter + 1
    if returnClusterIDs == True:
        return clusterSizes, numBondsPerCluster, clusterID
    else:
        return clusterSizes, numBondsPerCluster

def rigidClusterGenerator(radii,contactList,k,l):
    numParticles = len(radii)
    dataType = 'simulation'
    ThisConf = CF.Configuration(dataType, numParticles, radii)
    ThisConf.readSimdata(contactList)
    ThisPebble = PB.Pebbles(ThisConf, k, l, 'nothing', False)
    ThisPebble.play_game()
    ThisPebble.rigid_cluster()
    # Extract the cluster sizes, number of contacts in each cluster, and which bonds are participating in each cluster.
    clusterSizes, numBondsPerCluster, clusterID = rigidClusterDataGenerator(ThisPebble)
    return clusterSizes, numBondsPerCluster, clusterID

def distNormalVecCalc(pos1,pos2,boxSize):

    normalVector = pos1-pos2/np.sqrt(np.sum((pos1-pos2)**2))

    distances = np.zeros(len(boxSize))
    for i in range(0,len(boxSize)):
        distances[i] = np.abs(pos1[i]-pos2[i])
        if distances[i] > boxSize[i]/2:
            distances[i] = boxSize[i]-distances[i]
            normalVector[i]=-normalVector[i]

    return np.sqrt(np.sum(distances**2)),normalVector

def hessianDataGenerator(positions,radii,boxSize,slidingOrNot=0):
    """
    Given the positions, radii, and the box size of a particle packing that has numParticles this returns the relevant
    information for the 3D frictional hessian.
    :param positions: 2D numpy array that has all the 3D positions of the particles, it's shape is (numParticles, 3)
    :param radii:  1D numpy array that contains all the radiis of the particles, it's length is numParticles.
    :param boxSize: 1D numpy array with three entries that gives the three dimensional box size that confines the packing
    :return: contactData, this is an array that has a row for each contact in the packing.
    """
    #Particles are neighbors if they're distance-epsilon is less than the sum of the particle radii's, set epsilon to be small
    epsilon=np.max(radii)/10000

    #Initiliaze the contactData holder.
    contactData = []

    #Particle counters for the for loops
    particleI = 0

    #This is an incredibly inefficient way to do this and absolutely should be optimized at some point but for proof of
    #concept stuff this will do for now.
    #We are going to find contacting particles by looping over every particle pair and calculating their pairwise
    #distances and comparing that to the sum of their radii.
    for pos1 in positions:

        #The outer loop needs to over every ith row in positions but the inner loop only needs to run over the (i+1)th
        #row in positions, because we don't want to calculate the distance for redundant pairs of particles or for
        #particles with themselves.
        particleJ = particleI+1
        for pos2 in positions[particleI+1:,:]:
            #Find the distance and normal vector between the two particles in PBC's
            distance,normalVector = distNormalVecCalc(pos1,pos2,boxSize)

            #If that distance is such that the particles much be in contact, record it in the contactData array.
            if distance-epsilon < radii[particleI]+radii[particleJ]:
                newRow = np.array([particleI,particleJ,slidingOrNot,normalVector[0],normalVector[1],normalVector[2],distance])
                if contactData == []:
                    contactData = newRow
                else:
                    contactData = np.vstack((contactData, newRow))

            #Increment  counters
            particleJ = particleJ+1
        particleI=particleI+1

    return contactData

def ifEmptySetZeroOtherwiseReturnMax(array):
    if array.size==0:
        return 0
    else:
        return np.max(array)

"""
Part 2:  Assembling contact data for SC, BCC, and FCC lattice.
Below is crafting the information required for the 3D Hessian and various pebble games with
simple cubic, body centered cubic, and face centered cubic lattices.
"""

#For all lattices we can set normal and tangential spring stiffness' as well as whether each contact constrains one or
#two degrees of freedom:
kn=1
k_t=kn
slidingOrNot = 0 #0 is not sliding, 1 is sliding


#Simple Cubic Lattice SC
positionsSCInit = np.array([[0,0,0],[0,1,0],[0,1,1],[0,0,1],[1,0,1],[1,1,1],[1,1,0],[1,0,0]])
positionsSC = positionsSCInit
radiiSC = 1/2
boxSizeSC = np.array([2,2,2])
#Expand the lattice so that particles don't interact with a single particle twice in periodic boundary conditions
shifts = np.array([[0,1,0],[1,1,0],[1,0,0],[0,1,1],[1,1,1],[1,0,1],[0,0,1]])
for shift in shifts:
    positionsSC = np.vstack((positionsSC,positionsSCInit+shift*np.max(boxSizeSC)))
#This naive way of shifting the lattice gives doubles of interior boundary particles
positionsSC = np.unique(positionsSC, axis=0)
#Expand the boxsize
boxSizeSC=2*boxSizeSC
radiiSC=radiiSC*np.ones(len(positionsSC))
contactDataSC = hessianDataGenerator(positionsSC,radiiSC,boxSizeSC,0)
contactListSC = contactDataSC[:,:3]
#Check number of neighbors of particle 0
print("Particle 0 has "+str(np.sum(contactListSC[:,:2]==0))+" neighbors")
#Plot the lattice for a sanity check
fig = pyplot.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(positionsSC[:,0], positionsSC[:,1], positionsSC[:,2], marker='v',color="green",label="SC")
pyplot.legend()



"""
Mike 3/7/2023
This is where we will run various pebble games and hessian analysis' on the SC lattice.  Right now we neither the pebble 
game or the hessian analysis' are working.  Once on of them is working I will uncomment out the relevant code.
"""
# #(6,6) pebble game
# (k,l)=(6,6)
# #All contacts are sliding - 1 dof constrained per contact
# contactListSC[:,2]=0
# contactListBCC[:,2]=0
# contactListFCC[:,2]=0
# clusterSizesSC66_1dof, numBondsPerClusterSC66_1dof, clusterIDSC66_1dof = rigidClusterGenerator(radiiSC,contactListSC,k,l)
# clusterSizesBCC66_1dof, numBondsPerClusterBCC66_1dof, clusterIDBCC66_1dof = rigidClusterGenerator(radiiBCC,contactListBCC,k,l)
# clusterSizesFCC66_1dof, numBondsPerClusterFCC66_1dof, clusterIDFCC66_1dof = rigidClusterGenerator(radiiFCC,contactListFCC,k,l)
#
# #Set all contacts to non sliding - 2 dof constrained per contact
# contactListSC[:,2]=1
# contactListBCC[:,2]=1
# contactListFCC[:,2]=1
# clusterSizesSC66_2dof, numBondsPerClusterSC66_2dof, clusterIDSC66_2dof = rigidClusterGenerator(radiiSC,contactListSC,k,l)
# clusterSizesBCC66_2dof, numBondsPerClusterBCC66_2dof, clusterIDBCC66_2dof = rigidClusterGenerator(radiiBCC,contactListBCC,k,l)
# clusterSizesFCC66_2dof, numBondsPerClusterFCC66_2dof, clusterIDFCC66_2dof = rigidClusterGenerator(radiiFCC,contactListFCC,k,l)
#
#
# #(5,6) pebble game
# (k,l)=(5,6)
# #All contacts are sliding - 1 dof constrained per contact
# contactListSC[:,2]=0
# contactListBCC[:,2]=0
# contactListFCC[:,2]=0
# clusterSizesSC56_1dof, numBondsPerClusterSC56_1dof, clusterIDSC56_1dof = rigidClusterGenerator(radiiSC,contactListSC,k,l)
# clusterSizesBCC56_1dof, numBondsPerClusterBCC56_1dof, clusterIDBCC56_1dof = rigidClusterGenerator(radiiBCC,contactListBCC,k,l)
# clusterSizesFCC56_1dof, numBondsPerClusterFCC56_1dof, clusterIDFCC56_1dof = rigidClusterGenerator(radiiFCC,contactListFCC,k,l)
#
# #Set all contacts to non sliding - 2 dof constrained per contact
# contactListSC[:,2]=1
# contactListBCC[:,2]=1
# contactListFCC[:,2]=1
# clusterSizesSC56_2dof, numBondsPerClusterSC56_2dof, clusterIDSC56_2dof = rigidClusterGenerator(radiiSC,contactListSC,k,l)
# clusterSizesBCC56_2dof, numBondsPerClusterBCC56_2dof, clusterIDBCC56_2dof = rigidClusterGenerator(radiiBCC,contactListBCC,k,l)
# clusterSizesFCC56_2dof, numBondsPerClusterFCC56_2dof, clusterIDFCC56_2dof = rigidClusterGenerator(radiiFCC,contactListFCC,k,l)
#
#
#
#
# """Plot Results"""
# fontsizeLabel=15
# #Plot size of cluster vs lattice type
# x=["Simple Cubic \n $Z=6$", "Body Centered Cubic \n $Z=8$","Face Centered Cubic \n $Z=12$"]
# clusterSizes66_1dof = [ifEmptySetZeroOtherwiseReturnMax(clusterSizesSC66_1dof),ifEmptySetZeroOtherwiseReturnMax(clusterSizesBCC66_1dof),ifEmptySetZeroOtherwiseReturnMax(clusterSizesFCC66_1dof)]
# clusterSizes66_2dof = [ifEmptySetZeroOtherwiseReturnMax(clusterSizesSC66_2dof),ifEmptySetZeroOtherwiseReturnMax(clusterSizesBCC66_1dof),ifEmptySetZeroOtherwiseReturnMax(clusterSizesFCC66_2dof)]
# clusterSizes56_1dof = [ifEmptySetZeroOtherwiseReturnMax(clusterSizesSC56_1dof),ifEmptySetZeroOtherwiseReturnMax(clusterSizesBCC56_1dof),ifEmptySetZeroOtherwiseReturnMax(clusterSizesFCC56_1dof)]
# clusterSizes56_2dof = [ifEmptySetZeroOtherwiseReturnMax(clusterSizesSC56_2dof),ifEmptySetZeroOtherwiseReturnMax(clusterSizesBCC56_2dof),ifEmptySetZeroOtherwiseReturnMax(clusterSizesFCC56_2dof)]
# pyplot.plot(x,clusterSizes66_1dof,marker='o',label="(6,6) All Contacts Sliding")
# pyplot.plot(x,clusterSizes66_2dof,marker='o',label="(6,6) All Contacts Not Sliding")
# pyplot.plot(x,clusterSizes56_1dof,marker='o',label="(5,6) All Contacts Sliding")
# pyplot.plot(x,clusterSizes56_2dof,marker='o',label="(5,6) All Contacts Not Sliding")
# pyplot.xlabel("Lattice Type",fontsize=fontsizeLabel)
# pyplot.ylabel(r"Normalized Cluster Size $\frac{S}{N}$",fontsize=fontsizeLabel)
# pyplot.legend(fontsize=fontsizeLabel)
# pyplot.ylim((-0.001,1))
# pyplot.tight_layout()




"""
Mike: 3/7/2023
Below are the BCC and FCC lattices that are more coordinated than the SC lattice above.  They're not coded right and 
should not be used unless they're fixed.  I'm not sure what was going wrong but I suspect it was the intersitial sites 
that were the problems.
"""
#
# #Body Centered Cubic Lattice BCC
# positionsBCCMain = np.array([[0,0,0],[0,1,0],[0,1,1],[0,0,1],[1,0,1],[1,1,1],[1,1,0],[1,0,0]])
# positionsBCCMainInit = positionsBCCMain
# positionsBCCInter = np.array([[.5,.5,.5]])
# positionsBCCInterInit = positionsBCCInter
# radiiBCC = np.sqrt(3)/4
# boxSizeBCC = np.array([2,2,2])
# #Expand the lattice so that particles don't interact with a single particle twice in periodic boundary conditions
# shifts = np.array([[0,1,0],[1,1,0],[1,0,0],[0,1,1],[1,1,1],[1,0,1],[0,0,1]])
# #Shift main sites
# for shift in shifts:
#     positionsBCC = np.vstack((positionsBCC,positionsBCCMainInit+shift*np.max(boxSizeBCC)))
# #Shift Inter sites
# for shift in shifts:
#     positionsBCCInter = np.vstack((positionsBCCInter,positionsBCCInterInit+shift*np.max(boxSizeBCC)))
#     positionsBCCInter = np.vstack((positionsBCCInter, positionsBCCInterInit + shift*.5*np.max(boxSizeBCC)))
# #Collate main sites and intersitial sites
# positionsBCC = np.vstack((positionsBCC,positionsBCCInter))
# #This naive way of shifting the lattice gives doubles of interior boundary particles
# positionsBCC = np.unique(positionsBCC, axis=0)
# #Expand the boxsize
# boxSizeBCC=2*boxSizeBCC
# radiiBCC=radiiBCC*np.ones(len(positionsBCC))
# contactDataBCC = hessianDataGenerator(positionsBCC,radiiBCC,boxSizeBCC,0)
# contactListBCC = contactDataBCC[:,:3]
# #Check number of neighbors of particle 0
# for particle in range(0,int(np.max(contactListBCC[:,:2]))):
#     print("Particle " +str(particle)+ " has "+str(np.sum(contactListBCC[:,:2]==particle))+" neighbors")
#
#
# #Face Centered Cubic Lattice FCC
# positionsFCCMain = np.array([[0,0,0],[0,1,0],[0,1,1],[0,0,1],[1,0,1],[1,1,1],[1,1,0],[1,0,0]])
# positionsFCCMainInit = positionsFCCMain
# positionsFCCInter = np.array([[1,.5,.5],[.5,0,.5],[.5,.5,0],[.5,1,.5],[.5,.5,1]])
# positionsFCCInterInit = positionsFCCInter
# radiiFCC = 1/np.sqrt(8)
# boxSizeFCC = np.array([2,2,2])
# #Expand the lattice so that particles don't interact with a single particle twice in periodic boundary conditions
# shifts = np.array([[0,1,0],[1,1,0],[1,0,0],[0,1,1],[1,1,1],[1,0,1],[0,0,1]])
# #Shift main sites
# for shift in shifts:
#     positionsFCC = np.vstack((positionsFCC,positionsFCCMainInit+shift*np.max(boxSizeFCC)))
# #Shift intersitial sites
# for shift in shifts:
#     positionsFCCInter = np.vstack((positionsFCCInter,positionsFCCInterInit+shift*np.max(boxSizeFCC)))
#     positionsFCCInter = np.vstack((positionsFCCInter, positionsFCCInterInit + shift*.5*np.max(boxSizeFCC)))
# #Collate main sites and intersitial sites
# positionsFCC = np.vstack((positionsFCC,positionsFCCInter))
# #This naive way of shifting the lattice gives doubles of interior boundary particles
# positionsFCC = np.unique(positionsFCC, axis=0)
# #Expand the boxsize
# boxSizeFCC=2*boxSizeFCC
# radiiFCC=radiiFCC*np.ones(len(positionsFCC))
# contactDataFCC = hessianDataGenerator(positionsFCC,radiiFCC,boxSizeFCC,0)
# contactListFCC = contactDataFCC[:,:3]
#
# #Plot all the lattices for sanity check
# fig = pyplot.figure()
# ax = fig.add_subplot(projection='3d')
# #ax.scatter(positionsSC[:,0], positionsSC[:,1], positionsSC[:,2], marker='v',color="green",label="SC")
# ax.scatter(positionsBCC[:,0], positionsBCC[:,1], positionsBCC[:,2], marker='o',color="red",label="BCC")
# #ax.scatter(positionsFCC[:,0], positionsFCC[:,1], positionsFCC[:,2], marker='v',color="black",label="FCC")
# pyplot.legend()