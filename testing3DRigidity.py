import numpy as np
import sys
import Configuration_altered as CF
import Pebbles as PB
import itertools
import Hessian_3D

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

def rigidClusterGenerator(contactList,k,l):
    (numParticles,_) = np.shape(contactList)
    radii=np.ones(numParticles)
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


"""
Below is testing the 3D Hessian and various pebble games with simple cubic, body centered cubic, and face centered
cubic lattices.
"""

#For all lattices we can set normal and tangential spring stiffness' as well as whether each contact constrains one or
#two degrees of freedom:
kn=1
k_t=kn
slidingOrNot = 0 #Check how dof 0 is vs 1 (MIKE)


#Simple Cubic Lattice SC
positionsSCInit = np.array([[0,0,0],[0,1,0],[0,1,1],[0,0,1],[1,0,1],[1,1,1],[1,1,0],[1,0,0]])
positionsSC = positionsSCInit
radiiSC = 1/2
boxSizeSC = np.array([2,2,2])+2*radiiSC
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
hessianSC = Hessian_3D.hessianMatrixGenerator(contactDataSC,kn,k_t,slidingOrNot)


#Body Centered Cubic Lattice BCC
positionsBCCInit = np.array([[0,0,0],[0,1,0],[0,1,1],[0,0,1],[1,0,1],[1,1,1],[1,1,0],[1,0,0],[.5,.5,.5]])
positionsBCC = positionsBCCInit
radiiBCC = np.sqrt(3)/4
boxSizeBCC = np.array([1,1,1])+2*radiiBCC
#Expand the lattice so that particles don't interact with a single particle twice in periodic boundary conditions
shifts = np.array([[0,1,0],[1,1,0],[1,0,0],[0,1,1],[1,1,1],[1,0,1],[0,0,1]])
for shift in shifts:
    positionsBCC = np.vstack((positionsBCC,positionsBCCInit+shift*np.max(boxSizeBCC)))
#This naive way of shifting the lattice gives doubles of interior boundary particles
positionsBCC = np.unique(positionsBCC, axis=0)
#Expand the boxsize
boxSizeBCC=2*boxSizeBCC
radiiBCC=radiiBCC*np.ones(len(positionsBCC))
contactDataBCC = hessianDataGenerator(positionsBCC,radiiBCC,boxSizeBCC,0)
hessianBCC = Hessian_3D.hessianMatrixGenerator(contactDataBCC,kn,k_t,slidingOrNot)


#Face Centered Cubic lattice FCC
a=2*np.sqrt(2)
positionsFCC = np.array([[0,0,0],[0,a,0],[0,a,a],[0,0,a],[a,0,a],[a,a,a],[a,a,0],[a,0,0],[a/2,a/2,0],[0,a/2,a/2],[a/2,0,a/2],[a/2,a,a/2],[a/2,a/2,a],[a,a/2,a/2]])
positionsFCCInit = positionsFCC
radiiFCC = 1
boxSizeFCC = np.sqrt(8)*np.array([1,1,1])+2*radiiFCC
#Expand the lattice so that particles don't interact with a single particle twice in periodic boundary conditions
shifts = np.array([[0,1,0],[1,1,0],[1,0,0],[0,1,1],[1,1,1],[1,0,1],[0,0,1]])
for shift in shifts:
    positionsFCC = np.vstack((positionsFCC,positionsFCCInit+shift*np.max(boxSizeFCC)))
#This naive way of shifting the lattice gives doubles of interior boundary particles
positionsFCC = np.unique(positionsFCC, axis=0)
#Expand the boxsize
boxSizeFCC=2*boxSizeFCC
#
radiiFCC=radiiFCC*np.ones(len(positionsFCC))
contactDataFCC = hessianDataGenerator(positionsFCC,radiiFCC,boxSizeFCC,0)
hessianFCC = Hessian_3D.hessianMatrixGenerator(contactDataFCC,kn,k_t,slidingOrNot)






