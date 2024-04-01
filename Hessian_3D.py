# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 09:45:34 2022
@author: Mike van der Naald

This code is based on Silke Henke's code called "Hessian.py" found in the following Github repository:
    
    
    
That code takes experimental data and also frictional packing simulations as inputs
and generates their frictional Hessian, more information can be found in the following publication:
    
    

This code takes in LF_DEM simulation data as inputs and generates their frictional Hessian.


"""
import numpy as np
import sys
import os
from matplotlib import pyplot
import itertools
import re
from scipy.spatial.distance import pdist
from os import listdir
from os.path import isfile, join
import time
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
from numpy import linalg as LA
from scipy import sparse

def hessianMatrixGenerator(numParticles,contactData,kn,k_t,fullMatrix=True,cylinderHeight=1,particleDensity=1):
    """
    This function generates the 3D frictional hessian for a single packing.  The Hessian can be returned as either

    :param contactData: This is a numpy array with a row for each contact in the packing and seven columns.  The first
    column is the particle ID of the first particle i, the second column  is the particle ID of the second particle m
    the third column is a 0 if the contact is at the Coloumb threshold (sliding) and 1 if it is below. The fourth through
    sixth columns encode the three normal components of the contact n_x, n_y, and n_z, respectively.  Finally, the
    seventh column encodes the center to center distance of particles i and j.
    :param kn: This is the normal spring stiffness
    :param k_t:  This is tangential spring stiffness.
    :param fullMatrix:  If set to true the function returns the hessian matrix as a full 2D numpy array with all zeros.
    If set to false it returns four

    :return:
    """

    #Initialize the correct container
    if fullMatrix == True:
        hessian = np.zeros((6*numParticles,6*numParticles))
    else:
        hessianRowHolder = np.array([])
        hessianColHolder = np.array([])
        hessianDataHolder = np.array([])

    
    for contact in contactData:
        i = int(contact[0])
        j = int(contact[1])
        slidingOrNot = int(contact[2])
        # Finn: Also read contact nz0 from currentSnapShot in contactInfo (from currentContct from prior)
        nx0 = contact[3]
        ny0 = contact[4]
        nz0 = contact[5]
        rval = contact[6]


        R_i = radii[i]
        R_j = radii[j]

        Ai = (2.0 / 5.) ** 0.5
        Aj = (2.0 / 5) ** 0.5

        mi = particleDensity * cylinderHeight * np.pi * R_i ** 2
        mj = particleDensity * cylinderHeight * np.pi * R_j ** 2

        fn = kn * (R_i + R_j - dimGap)

        if slidingOrNot == 2:
            kt = k_t
        else:
            kt = 0

        # Finn: This is the square in global coordinates
        Hij = np.zeros((6, 6))
        Hij[0, 0] = (((fn / rval) - kt) * (1 - nx0 ** 2)) - kn * nx0 ** 2
        Hij[1, 0] = -(kn + (fn / rval) - kt) * nx0 * ny0
        Hij[2, 0] = -(kn + (fn / rval) - kt) * nx0 * nz0
        Hij[4, 0] = -Ai * kt * nz0
        Hij[5, 0] = Ai * kt * ny0
        Hij[0, 1] = -(kn + (fn / rval) - kt) * nx0 * ny0
        Hij[1, 1] = (((fn / rval) - kt) * (1 - ny0 ** 2)) - kn * ny0 ** 2
        Hij[2, 1] = -(kn + (fn / rval) - kt) * ny0 * nz0
        Hij[3, 1] = Ai * kt * nz0
        Hij[5, 1] = -Ai * kt * nx0
        Hij[0, 2] = -(kn + (fn / rval) - kt) * nx0 * nz0
        Hij[1, 2] = -(kn + (fn / rval) - kt) * ny0 * nz0
        Hij[2, 2] = (((fn / rval) - kt) * (1 - nz0 ** 2)) - kn * nz0 ** 2
        Hij[3, 2] = -Ai * kt * ny0
        Hij[4, 2] = Ai * kt * nx0
        Hij[1, 3] = -Ai * kt * nz0
        Hij[2, 3] = Ai * kt * ny0
        Hij[3, 3] = Ai * Aj * kt * (1 - (nx0 ** 2))
        Hij[4, 3] = -Ai * Aj * kt * nx0 * ny0
        Hij[5, 3] = -Ai * Aj * kt * nx0 * nz0
        Hij[0, 4] = Ai * kt * nz0
        Hij[2, 4] = -Ai * kt * nx0
        Hij[3, 4] = -Ai * Aj * kt * nx0 * ny0
        Hij[4, 4] = Ai * Aj * kt * (1 - (ny0 ** 2))
        Hij[5, 4] = -Ai * Aj * kt * ny0 * nz0
        Hij[0, 5] = -Ai * kt * ny0
        Hij[1, 5] = Ai * kt * nx0
        Hij[3, 5] = -Ai * Aj * kt * nx0 * nz0
        Hij[4, 5] = -Ai * Aj * kt * ny0 * nz0
        Hij[5, 5] = Ai * Aj * kt * (1 - (nz0 ** 2))

        if fullMatrix == True:
            #put sub matrix into big one
            hessian[6*i:(6*i+6),6*j:(6*j+6)] = Hij/(mi*mj)**0.5
        else:

            hessianRowHolder = np.append(
                [6 * i, 6 * i, 6 * i, 6 * i, 6 * i, 6 * i, 6 * i + 1, 6 * i + 1, 6 * i + 1, 6 * i + 1, 6 * i + 1,
                 6 * i + 1, 6 * i + 2, 6 * i + 2, 6 * i + 2, 6 * i + 2, 6 * i + 2, 6 * i + 2, 6 * i + 3, 6 * i + 3,
                 6 * i + 3, 6 * i + 3, 6 * i + 3, 6 * i + 3, 6 * i + 4, 6 * i + 4, 6 * i + 4, 6 * i + 4, 6 * i + 4,
                 6 * i + 4, 6 * i + 5, 6 * i + 5, 6 * i + 5, 6 * i + 5, 6 * i + 5, 6 * i + 5], hessianRowHolder)
            hessianColHolder = np.append(
                [6 * j, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5, 6 * j, 6 * j + 1, 6 * j + 2, 6 * j + 3,
                 6 * j + 4, 6 * j + 5, 6 * j, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5, 6 * j, 6 * j + 1,
                 6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5, 6 * j, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4,
                 6 * j + 5, 6 * j, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5], hessianColHolder)
            currentMatCValues = [Hij[0, 0], Hij[0, 1], Hij[0, 2], Hij[0, 3], Hij[0, 4], Hij[0, 5], Hij[1, 0], Hij[1, 2],
                                 Hij[1, 3], Hij[1, 4], Hij[1, 5], Hij[2, 0], Hij[2, 1], Hij[2, 3], Hij[2, 4], Hij[2, 5],
                                 Hij[3, 0], Hij[3, 1], Hij[3, 2], Hij[3, 3], Hij[3, 4], Hij[3, 5], Hij[4, 0], Hij[4, 1],
                                 Hij[4, 2], Hij[4, 3], Hij[4, 4], Hij[4, 5], Hij[5, 0], Hij[5, 1], Hij[5, 2], Hij[5, 3],
                                 Hij[5, 4], Hij[5, 5]]
            hessianDataHolder = np.append(currentMatCValues / (mi * mj) ** 0.5, hessianDataHolder)


        # Finn: Construct ji matrix via transpose
        Hji = np.transpose(Hij)

        # Mike:  Here is the sparse matrix format of what is above.
        # hessianRowHolder = np.append([3*j,3*j,3*j+1,3*j+1,3*j,3*j+1,3*j+2,3*j+2,3*j+2],hessianRowHolder)
        # hessianColHolder = np.append([3*i,3*i+1,3*i,3*i+1,3*i+2,3*i+2,3*i,3*i+1,3*i+2],hessianColHolder)
        # currentMatCValues = [subsquare[0, 0] * (-nx0)**2 + subsquare[1, 1] * (-tx0)**2,subsquare[0, 0] * (-nx0) * (-ny0) + subsquare[1, 1] * (-tx0) * (-ty0),subsquare[0, 0] * (-ny0) * (-nx0) + subsquare[1, 1] * (-ty0) * (-tx0),subsquare[0, 0] * (-ny0)**2 + subsquare[1, 1] * (-ty0)**2,subsquare[1, 2] * (-tx0),subsquare[1, 2] * (-ty0),subsquare[2, 1] * (-tx0),subsquare[2, 1] * (-ty0),subsquare[2, 2]]
        # hessianDataHolder = np.append(currentMatCValues /(mi * mj)**0.5,hessianDataHolder)
        # And put it into the Hessian
        # now for contact ji
        # hessianHolder[3 * j:(3 * j + 3),3 * i:(3 * i + 3),counter] = Hji / (mi * mj)**0.5
        if fullMatrix ==True:
            hessian[6*j:(6*j+6),6*i:(6*i+6)]=Hji/(mi*mj)**0.5
        else:
            hessianRowHolder = np.append(
                [6 * i, 6 * i, 6 * i, 6 * i, 6 * i, 6 * i, 6 * i + 1, 6 * i + 1, 6 * i + 1, 6 * i + 1, 6 * i + 1,
                 6 * i + 1, 6 * i + 2, 6 * i + 2, 6 * i + 2, 6 * i + 2, 6 * i + 2, 6 * i + 2, 6 * i + 3, 6 * i + 3,
                 6 * i + 3, 6 * i + 3, 6 * i + 3, 6 * i + 3, 6 * i + 4, 6 * i + 4, 6 * i + 4, 6 * i + 4, 6 * i + 4,
                 6 * i + 4, 6 * i + 5, 6 * i + 5, 6 * i + 5, 6 * i + 5, 6 * i + 5, 6 * i + 5], hessianRowHolder)
            hessianColHolder = np.append(
                [6 * j, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5, 6 * j, 6 * j + 1, 6 * j + 2, 6 * j + 3,
                 6 * j + 4, 6 * j + 5, 6 * j, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5, 6 * j, 6 * j + 1,
                 6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5, 6 * j, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4,
                 6 * j + 5, 6 * j, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5], hessianColHolder)
            currentMatCValues = [Hji[0, 0], Hji[0, 1], Hji[0, 2], Hji[0, 3], Hji[0, 4], Hji[0, 5], Hji[1, 0], Hji[1, 2],
                                 Hji[1, 3], Hji[1, 4], Hji[1, 5], Hji[2, 0], Hji[2, 1], Hji[2, 3], Hji[2, 4], Hji[2, 5],
                                 Hji[3, 0], Hji[3, 1], Hji[3, 2], Hji[3, 3], Hji[3, 4], Hji[3, 5], Hji[4, 0], Hji[4, 1],
                                 Hji[4, 2], Hji[4, 3], Hji[4, 4], Hji[4, 5], Hji[5, 0], Hji[5, 1], Hji[5, 2], Hji[5, 3],
                                 Hji[5, 4], Hji[5, 5]]
            hessianDataHolder = np.append(currentMatCValues / (mi * mj) ** 0.5, hessianDataHolder)

        # Careful, the diagonal bits are not just minus because of the rotations
        # diagsquare = np.zeros((3, 3))
        # diagsquare[0, 0] = kn
        # diagsquare[1, 1] = -fn / rval + kt
        # diagsquare[1, 2] = kt * Ai
        # diagsquare[2, 1] = kt * Ai
        # diagsquare[2, 2] = kt * Ai**2

        Hii = np.zeros((6, 6))
        Hii[0, 0] = (((fn / rval) - kt) * (1 - nx0 ** 2)) + kn * nx0 ** 2
        Hii[1, 0] = (kn + (fn / rval) - kt) * nx0 * ny0
        Hii[2, 0] = (kn + (fn / rval) - kt) * nx0 * nz0
        Hii[4, 0] = Ai * kt * nz0
        Hii[5, 0] = -Ai * kt * ny0
        Hii[0, 1] = (kn + (fn / rval) - kt) * nx0 * ny0
        Hii[1, 1] = (((fn / rval) - kt) * (1 - ny0 ** 2)) + kn * ny0 ** 2
        Hii[2, 1] = (kn + (fn / rval) - kt) * ny0 * nz0
        Hii[3, 1] = -Ai * kt * nz0
        Hii[5, 1] = Ai * kt * nx0
        Hii[0, 2] = (kn + (fn / rval) - kt) * nx0 * nz0
        Hii[1, 2] = (kn + (fn / rval) - kt) * ny0 * nz0
        Hii[2, 2] = (((fn / rval) - kt) * (1 - nz0 ** 2)) + kn * nz0 ** 2
        Hii[3, 2] = Ai * kt * ny0
        Hii[4, 2] = -Ai * kt * nx0
        Hii[1, 3] = -Ai * kt * nz0
        Hii[2, 3] = Ai * kt * ny0
        Hii[3, 3] = Ai * Aj * kt * (1 - (nx0 ** 2))
        Hii[4, 3] = -Ai * Aj * kt * nx0 * ny0
        Hii[5, 3] = -Ai * Aj * kt * nx0 * nz0
        Hii[0, 4] = Ai * kt * nz0
        Hii[2, 4] = -Ai * kt * nx0
        Hii[3, 4] = -Ai * Aj * kt * nx0 * ny0
        Hii[4, 4] = Ai * Aj * kt * (1 - (ny0 ** 2))
        Hii[5, 4] = -Ai * Aj * kt * ny0 * nz0
        Hii[0, 5] = -Ai * kt * ny0
        Hii[1, 5] = Ai * kt * nx0
        Hii[3, 5] = -Ai * Aj * kt * nx0 * nz0
        Hii[4, 5] = -Ai * Aj * kt * ny0 * nz0
        Hii[5, 5] = Ai * Aj * kt * (1 - (nz0 ** 2))

        Hjj = np.zeros((6, 6))
        Hjj[0, 0] = (((fn / rval) - kt) * (1 - nx0 ** 2)) + kn * nx0 ** 2
        Hjj[1, 0] = (kn + (fn / rval) - kt) * nx0 * ny0
        Hjj[2, 0] = (kn + (fn / rval) - kt) * nx0 * nz0
        Hjj[4, 0] = Aj * kt * nz0
        Hjj[5, 0] = -Aj * kt * ny0
        Hjj[0, 1] = (kn + (fn / rval) - kt) * nx0 * ny0
        Hjj[1, 1] = (((fn / rval) - kt) * (1 - ny0 ** 2)) + kn * ny0 ** 2
        Hjj[2, 1] = (kn + (fn / rval) - kt) * ny0 * nz0
        Hjj[3, 1] = -Aj * kt * nz0
        Hjj[5, 1] = Aj * kt * nx0
        Hjj[0, 2] = (kn + (fn / rval) - kt) * nx0 * nz0
        Hjj[1, 2] = (kn + (fn / rval) - kt) * ny0 * nz0
        Hjj[2, 2] = (((fn / rval) - kt) * (1 - nz0 ** 2)) + kn * nz0 ** 2
        Hjj[3, 2] = Aj * kt * ny0
        Hjj[4, 2] = -Aj * kt * nx0
        Hjj[1, 3] = -Aj * kt * nz0
        Hjj[2, 3] = Aj * kt * ny0
        Hjj[3, 3] = Aj * Aj * kt * (1 - (nx0 ** 2))
        Hjj[4, 3] = -Aj * Aj * kt * nx0 * ny0
        Hjj[5, 3] = -Aj * Aj * kt * nx0 * nz0
        Hjj[0, 4] = Aj * kt * nz0
        Hjj[2, 4] = -Aj * kt * nx0
        Hjj[3, 4] = -Aj * Aj * kt * nx0 * ny0
        Hjj[4, 4] = Aj * Aj * kt * (1 - (ny0 ** 2))
        Hjj[5, 4] = -Aj * Aj * kt * ny0 * nz0
        Hjj[0, 5] = -Aj * kt * ny0
        Hjj[1, 5] = Aj * kt * nx0
        Hjj[3, 5] = -Aj * Aj * kt * nx0 * nz0
        Hjj[4, 5] = -Aj * Aj * kt * ny0 * nz0
        Hjj[5, 5] = Aj * Aj * kt * (1 - (nz0 ** 2))

        # Stick this into the appropriate places:
        # Hijdiag = np.zeros((3, 3))
        # Hijdiag[0, 0] = diagsquare[0, 0] * nx0**2 + diagsquare[1, 1] * tx0**2
        # Hijdiag[0, 1] = diagsquare[0, 0] * nx0 * ny0 + diagsquare[1, 1] * tx0 * ty0
        # Hijdiag[1, 0] = diagsquare[0, 0] * ny0 * nx0 + diagsquare[1, 1] * ty0 * tx0
        # Hijdiag[1, 1] = diagsquare[0, 0] * ny0**2 + diagsquare[1, 1] * ty0**2
        # Hijdiag[0, 2] = diagsquare[1, 2] * tx0
        # Hijdiag[1, 2] = diagsquare[1, 2] * ty0
        # Hijdiag[2, 0] = diagsquare[2, 1] * tx0
        # Hijdiag[2, 1] = diagsquare[2, 1] * ty0
        # Hijdiag[2, 2] = diagsquare[2, 2]

        # And then *add* it to the diagnual
        # hessianRowHolder = np.append([3*i,3*i,3*i+1,3*i+1,3*i,3*i+1,3*i+2,3*i+2,3*i+2],hessianRowHolder)
        # hessianColHolder = np.append([3*i,3*i+1,3*i,3*i+1,3*i+2,3*i+2,3*i,3*i+1,3*i+2],hessianColHolder)
        # currentMatCValues = [diagsquare[0, 0] * nx0**2 + diagsquare[1, 1] * tx0**2,diagsquare[0, 0] * nx0 * ny0 + diagsquare[1, 1] * tx0 * ty0,diagsquare[0, 0] * ny0 * nx0 + diagsquare[1, 1] * ty0 * tx0,diagsquare[0, 0] * ny0**2 + diagsquare[1, 1] * ty0**2,diagsquare[1, 2] * tx0,diagsquare[1, 2] * ty0,diagsquare[2, 1] * tx0,diagsquare[2, 1] * ty0,diagsquare[2, 2]]
        # hessianDataHolder = np.append(currentMatCValues / mi,hessianDataHolder)
        if fullMatrix ==True:
            #Finn, I need your help here
            hessian[6*i:(6*i+6),6*i:(6*i+6)]=Hii/mi

            hessian[6*j:(6*j+6),6*j:(6*j+6)]=Hjj/mj



        else:

            hessianRowHolder = np.append(
                [6 * i, 6 * i, 6 * i, 6 * i, 6 * i, 6 * i, 6 * i + 1, 6 * i + 1, 6 * i + 1, 6 * i + 1, 6 * i + 1,
                 6 * i + 1, 6 * i + 2, 6 * i + 2, 6 * i + 2, 6 * i + 2, 6 * i + 2, 6 * i + 2, 6 * i + 3, 6 * i + 3,
                 6 * i + 3, 6 * i + 3, 6 * i + 3, 6 * i + 3, 6 * i + 4, 6 * i + 4, 6 * i + 4, 6 * i + 4, 6 * i + 4,
                 6 * i + 4, 6 * i + 5, 6 * i + 5, 6 * i + 5, 6 * i + 5, 6 * i + 5, 6 * i + 5], hessianRowHolder)
            hessianColHolder = np.append(
                [6 * i, 6 * i + 1, 6 * i + 2, 6 * i + 3, 6 * i + 4, 6 * i + 5, 6 * i, 6 * i + 1, 6 * i + 2, 6 * i + 3,
                 6 * i + 4, 6 * i + 5, 6 * i, 6 * i + 1, 6 * i + 2, 6 * i + 3, 6 * i + 4, 6 * i + 5, 6 * i, 6 * i + 1,
                 6 * i + 2, 6 * i + 3, 6 * i + 4, 6 * i + 5, 6 * i, 6 * i + 1, 6 * i + 2, 6 * i + 3, 6 * i + 4,
                 6 * i + 5, 6 * i, 6 * i + 1, 6 * i + 2, 6 * i + 3, 6 * i + 4, 6 * i + 5], hessianColHolder)
            currentMatCValues = [Hii[0, 0], Hii[0, 1], Hii[0, 2], Hii[0, 3], Hii[0, 4], Hii[0, 5], Hii[1, 0], Hii[1, 2],
                                 Hii[1, 3], Hii[1, 4], Hii[1, 5], Hii[2, 0], Hii[2, 1], Hii[2, 3], Hii[2, 4], Hii[2, 5],
                                 Hii[3, 0], Hii[3, 1], Hii[3, 2], Hii[3, 3], Hii[3, 4], Hii[3, 5], Hii[4, 0], Hii[4, 1],
                                 Hii[4, 2], Hii[4, 3], Hii[4, 4], Hii[4, 5], Hii[5, 0], Hii[5, 1], Hii[5, 2], Hii[5, 3],
                                 Hii[5, 4], Hii[5, 5]]
            hessianDataHolder = np.append(currentMatCValues / mi, hessianDataHolder)

        # hessianHolder[3 * i:(3 * i + 3), 3 * i:(3 * i + 3),counter] += Hijdiag / mi

        # And once more for the jj contribution, which is the same whizz with the flipped sign of n and t
        # and adjusted A's
        # diagsquare = np.zeros((3, 3))
        # diagsquare[0, 0] = kn
        # diagsquare[1, 1] = -fn / rval + kt
        # diagsquare[1, 2] = kt * Aj
        # diagsquare[2, 1] = kt * Aj
        # diagsquare[2, 2] = kt * Aj**2


        # Hjidiag = np.zeros((3, 3))
        # Hjidiag[0, 0] = diagsquare[0, 0] * (-nx0)**2 + diagsquare[1, 1] * (-tx0)**2
        # jidiag[0, 1] = diagsquare[0, 0] * (-nx0) * (-ny0) + diagsquare[1, 1] * (-tx0) * (-ty0)
        # Hjidiag[1, 0] = diagsquare[0, 0] * (-ny0) * (-nx0) + diagsquare[1, 1] * (-ty0) * (-tx0)
        # Hjidiag[1, 1] = diagsquare[0, 0] * (-ny0)**2 + diagsquare[1, 1] * (-ty0)**2
        # Hjidiag[0, 2] = diagsquare[1, 2] * (-tx0)
        # Hjidiag[1, 2] = diagsquare[1, 2] * (-ty0)
        # Hjidiag[2, 0] = diagsquare[2, 1] * (-tx0)
        # Hjidiag[2, 1] = diagsquare[2, 1] * (-ty0)
        # Hjidiag[2, 2] = diagsquare[2, 2]

        # And then *add* it to the diagnual
        # hessianHolder[3 * j:(3 * j + 3), 3 * j:(3 * j + 3),counter] += Hjidiag / mj
        # hessianRowHolder = np.append([3*j,3*j,3*j+1,3*j+1,3*j,3*j+1,3*j+2,3*j+2,3*j+2],hessianRowHolder)
        # hessianColHolder = np.append([3*j,3*j+1,3*j,3*j+1,3*j+2,3*j+2,3*j,3*j+1,3*j+2],hessianColHolder)
        # currentMatCValues = [diagsquare[0, 0] * (-nx0)**2 + diagsquare[1, 1] * (-tx0)**2,diagsquare[0, 0] * (-nx0) * (-ny0) + diagsquare[1, 1] * (-tx0) * (-ty0),diagsquare[0, 0] * (-ny0) * (-nx0) + diagsquare[1, 1] * (-ty0) * (-tx0),diagsquare[0, 0] * (-ny0)**2 + diagsquare[1, 1] * (-ty0)**2,diagsquare[1, 2] * (-tx0),diagsquare[1, 2] * (-ty0),diagsquare[2, 1] * (-tx0),diagsquare[2, 1] * (-ty0),diagsquare[2, 2]]
        # hessianDataHolder = np.append(currentMatCValues / mj,hessianDataHolder)

        hessianRowHolder = np.append(
            [6 * j, 6 * j, 6 * j, 6 * j, 6 * j, 6 * j, 6 * j + 1, 6 * j + 1, 6 * j + 1, 6 * j + 1, 6 * j + 1,
             6 * j + 1, 6 * j + 2, 6 * j + 2, 6 * j + 2, 6 * j + 2, 6 * j + 2, 6 * j + 2, 6 * j + 3, 6 * j + 3,
             6 * j + 3, 6 * j + 3, 6 * j + 3, 6 * j + 3, 6 * j + 4, 6 * j + 4, 6 * j + 4, 6 * j + 4, 6 * j + 4,
             6 * j + 4, 6 * j + 5, 6 * j + 5, 6 * j + 5, 6 * j + 5, 6 * j + 5, 6 * j + 5], hessianRowHolder)
        hessianColHolder = np.append(
            [6 * j, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5, 6 * j, 6 * j + 1, 6 * j + 2, 6 * j + 3,
             6 * j + 4, 6 * j + 5, 6 * j, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5, 6 * j, 6 * j + 1,
             6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5, 6 * j, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4,
             6 * j + 5, 6 * j, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5], hessianColHolder)
        currentMatCValues = [Hjj[0, 0], Hjj[0, 1], Hjj[0, 2], Hjj[0, 3], Hjj[0, 4], Hjj[0, 5], Hjj[1, 0], Hjj[1, 2],
                             Hjj[1, 3], Hjj[1, 4], Hjj[1, 5], Hjj[2, 0], Hjj[2, 1], Hjj[2, 3], Hjj[2, 4], Hjj[2, 5],
                             Hjj[3, 0], Hjj[3, 1], Hjj[3, 2], Hjj[3, 3], Hjj[3, 4], Hjj[3, 5], Hjj[4, 0], Hjj[4, 1],
                             Hjj[4, 2], Hjj[4, 3], Hjj[4, 4], Hjj[4, 5], Hjj[5, 0], Hjj[5, 1], Hjj[5, 2], Hjj[5, 3],
                             Hjj[5, 4], Hjj[5, 5]]
        hessianDataHolder = np.append(currentMatCValues / mj, hessianDataHolder)


    if fullMatrix==True:
        return hessian
    else:
        return hessianRowHolder,hessianColHolder,hessianDataHolder,hessian3rdDimHolder



def hessianGenerator(radii,springConstants,contactInfo,outputDir,nameOfFile,snapShotRange=False,partialSave=False,cylinderHeight=1,particleDensity=1):

    """
    This function generates a 2D frictional Hessian using the formulation and some original code from the following publication:

    The three input variables are:
        1.  The radii, this is a numpy array that has an entry for each particle and denotes what their radii is.
        2.  springConstants is a numpy array that has a row for each simulation snap shot.  The first column is the
        strain step of the snap shot, the second column is the normal spring constant coeff, and the third column is the
        tangential spring constant coeff.
        3.  contactInfo is a list with each entry corresponding to a numpy array for each simulation snapshot.  Each numpy array
        has a row for each frictional contact, the first column is the ID of the particle i, the second column is the ID of particle j.
        The third and fourth columns are the x and z components of the normal vector (y is zero in 2D).  Finally the fifth entry is the
        "dimensionless gap which we'll convert into the center to center distance.
        4.  outputDir is the directory to output the Hessian to.
        5.  nameOfFile:  This is the name of the file that will be outputted into outputDir
        6 snapShotRange:  This is an optional argument that you can use to extract the data needed for the hessian for only
        specific simulation snapshots as the par_ and int_ files will usually contain >200 snapshots and that can take awhile.
        If snapShotRange=False then this program will output the data for every snapshot.  If snapShotRange=[0,5] that will
        process the first 5 snapshots and if snapShotRange=[50,-1] then this function will begin at the 51st snapshot and
        keep processing until it reaches the last snapshot.
        7 partialSave:  This outputs hessianFiles more so that the RAM doesn't get bogged down by holding all the data at once.
    """
    snapShotArray = np.linspace(snapShotRange[0],snapShotRange[1],snapShotRange[1]-snapShotRange[0]+1)
    lastSaveSnapShot = snapShotRange[0]
    #Each Hessian file takes up a lot of spac and
    if os.path.exists(os.path.join(outputDir,"hessianFiles"))==False:
        os.mkdir(os.path.join(outputDir,"hessianFiles"))




    #Most entries in the Hessian will be zero.  This is great for us because it means
    #it will be faster to just store the value of the entry and the three (i,j,k) values
    #that correspond to the indices of the entries.  Indices will start at 0.

    hessianRowHolder = np.array([])
    hessianColHolder = np.array([])
    hessian3rdDimHolder = np.array([])
    hessianDataHolder = np.array([])

    globalTimer = time.time()
    counter=0
    springConstantCounter=0
    for currentSnapShot in contactInfo:
        print("Starting new snapshot, currently this calculation has been running for:" +str((time.time()-globalTimer)/60) +" minutes")
        #Let's get the spring constants for this snapshot
        kn = springConstants[springConstantCounter,1]
        k_t = springConstants[springConstantCounter,2]
        snapShotIndex = snapShotArray[counter]


        #Now let's loop through every contact and construct that portion of the Hessian.
        for contact in currentSnapShot:
            if np.array_equal(currentSnapShot,np.array([0]))!=True:

                i = int(contact[0])
                j = int(contact[1])
                slidingOrNot = int(contact[2])
        #Finn: Also read contact nz0 from currentSnapShot in contactInfo (from currentContct from prior)
                nx0 = contact[3]
                ny0 = contact[4]
                nz0 = contact[5]
                dimGap = contact[6]

                #tx0 = -ny0
                #ty0 = nx0
        #Change orientation generation

                R_i = radii[i]
                R_j = radii[j]
                rval = (1/2)*(dimGap+2)*(R_i+R_j)

                Ai = (2.0 / 5.)**0.5
                Aj = (2.0 / 5)**0.5

                mi = particleDensity * cylinderHeight * np.pi * R_i**2
                mj = particleDensity * cylinderHeight * np.pi * R_j**2

                fn = kn * (R_i + R_j - dimGap)

                if slidingOrNot==2:
                    kt = k_t
                else:
                    kt = 0


                #Mike:  From here on out we're trusting Silke's original construction
                #and just relabeling things.  The other thing is that we have to
                #reconfigure everything so that it is in a coo_array format


                # This is our litte square in local coordinates (where nonzero)
                #subsquare = np.zeros((3, 3))
                #subsquare[0, 0] = -kn
                #subsquare[1, 1] = fn / rval - kt
                #subsquare[1, 2] = kt * Aj
                # note asymmetric cross-term
                #subsquare[2, 1] = -kt * Ai
                #subsquare[2, 2] = kt * Ai * Aj

                #Finn: This is the sqare in global coordinates
                Hij = np.zeros((6, 6))
                Hij[0, 0] = (((fn/rval)-kt)*(1 - nx0**2)) - kn * nx0**2
                Hij[1, 0] = -(kn + (fn/rval) - kt) * nx0 * ny0
                Hij[2, 0] = -(kn + (fn/rval) - kt) * nx0 * nz0
                Hij[4, 0] = -Ai * kt * nz0
                Hij[5, 0] = Ai * kt * ny0
                Hij[0, 1] = -(kn + (fn/rval) - kt)* nx0 * ny0
                Hij[1, 1] = (((fn/rval)-kt)*(1 - ny0**2)) - kn * ny0**2
                Hij[2, 1] = -(kn + (fn/rval) - kt)* ny0 * nz0
                Hij[3, 1] = Ai * kt * nz0
                Hij[5, 1] = -Ai * kt * nx0
                Hij[0, 2] = -(kn + (fn/rval) - kt)* nx0 * nz0
                Hij[1, 2] = -(kn + (fn/rval) - kt)* ny0 * nz0
                Hij[2, 2] = (((fn/rval)-kt)*(1 - nz0**2)) - kn * nz0**2
                Hij[3, 2] = -Ai * kt * ny0
                Hij[4, 2] = Ai * kt * nx0
                Hij[1, 3] = -Ai * kt * nz0
                Hij[2, 3] = Ai * kt * ny0
                Hij[3, 3] = Ai * Aj * kt * (1 - (nx0**2))
                Hij[4, 3] = -Ai * Aj * kt * nx0 * ny0
                Hij[5, 3] = -Ai * Aj * kt * nx0 * nz0
                Hij[0, 4] = Ai * kt * nz0
                Hij[2, 4] = -Ai * kt * nx0
                Hij[3, 4] = -Ai * Aj * kt * nx0 * ny0
                Hij[4, 4] = Ai * Aj * kt * (1 - (ny0**2))
                Hij[5, 4] = -Ai * Aj * kt * ny0 * nz0
                Hij[0, 5] = -Ai * kt * ny0
                Hij[1, 5] = Ai * kt * nx0
                Hij[3, 5] = -Ai * Aj * kt * nx0 * nz0
                Hij[4, 5] = -Ai * Aj * kt * ny0 * nz0
                Hij[5, 5] = Ai * Aj * kt * (1 - (nz0**2))




                #Mike:  Since we're no longer sticking it in one huge matrix I'm instead encoding these values into
                #the sparse array format
                # Stick this into the appropriate places after rotating it away from the (n,t) frame
                #Hij = np.zeros((3, 3))
                #Hij[0, 0] = subsquare[0, 0] * nx0**2 + subsquare[1, 1] * tx0**2
                #Hij[0,1] = subsquare[0, 0] * nx0 * ny0 + subsquare[1, 1] * tx0 * ty0
                #Hij[1,0] = subsquare[0, 0] * ny0 * nx0 + subsquare[1, 1] * ty0 * tx0
                #Hij[1, 1] = subsquare[0, 0] * ny0**2 + subsquare[1, 1] * ty0**2
                #Hij[0, 2] = subsquare[1, 2] * tx0
                #Hij[1, 2] = subsquare[1, 2] * ty0
                #Hij[2, 0] = subsquare[2, 1] * tx0
                #Hij[2, 1] = subsquare[2, 1] * ty0
                #Hij[2, 2] = subsquare[2, 2]

                #Mike:  Here is the sparse matrix format of what is above.
                #hessianRowHolder = np.append([3*i,3*i,3*i+1,3*i+1,3*i,3*i+1,3*i+2,3*i+2,3*i+2],hessianRowHolder)
                #hessianColHolder = np.append([3*j,3*j+1,3*j,3*j+1,3*j+2,3*j+2,3*j,3*j+1,3*j+2],hessianColHolder)
                #currentMatCValues = [subsquare[0, 0] * nx0**2 + subsquare[1, 1] * tx0**2,subsquare[0, 0] * nx0 * ny0 + subsquare[1, 1] * tx0 * ty0,subsquare[0, 0] * ny0 * nx0 + subsquare[1, 1] * ty0 * tx0,subsquare[0, 0] * ny0**2 + subsquare[1, 1] * ty0**2,subsquare[1, 2] * tx0,subsquare[1, 2] * ty0,subsquare[2, 1] * tx0,subsquare[2, 1] * ty0,subsquare[2, 2]]
                #hessianDataHolder = np.append(currentMatCValues/ (mi * mj)**0.5,hessianDataHolder)

                hessianRowHolder = np.append([6*i,6*i,6*i,6*i,6*i,6*i,6*i+1,6*i+1,6*i+1,6*i+1,6*i+1,6*i+1,6*i+2,6*i+2,6*i+2,6*i+2,6*i+2,6*i+2,6*i+3,6*i+3,6*i+3,6*i+3,6*i+3,6*i+3,6*i+4,6*i+4,6*i+4,6*i+4,6*i+4,6*i+4,6*i+5,6*i+5,6*i+5,6*i+5,6*i+5,6*i+5],hessianRowHolder)
                hessianColHolder = np.append([6*j,6*j+1,6*j+2,6*j+3,6*j+4,6*j+5,6*j,6*j+1,6*j+2,6*j+3,6*j+4,6*j+5,6*j,6*j+1,6*j+2,6*j+3,6*j+4,6*j+5,6*j,6*j+1,6*j+2,6*j+3,6*j+4,6*j+5,6*j,6*j+1,6*j+2,6*j+3,6*j+4,6*j+5,6*j,6*j+1,6*j+2,6*j+3,6*j+4,6*j+5],hessianColHolder)
                currentMatCValues = [Hij[0,0],Hij[0,1],Hij[0,2],Hij[0,3],Hij[0,4],Hij[0,5],Hij[1,0],Hij[1,2],Hij[1,3],Hij[1,4],Hij[1,5],Hij[2,0],Hij[2,1],Hij[2,3],Hij[2,4],Hij[2,5],Hij[3,0],Hij[3,1],Hij[3,2],Hij[3,3],Hij[3,4],Hij[3,5],Hij[4,0],Hij[4,1],Hij[4,2],Hij[4,3],Hij[4,4],Hij[4,5],Hij[5,0],Hij[5,1],Hij[5,2],Hij[5,3],Hij[5,4],Hij[5,5]]
                hessianDataHolder = np.append(currentMatCValues/ (mi * mj)**0.5,hessianDataHolder)
                # And put it into the Hessian, with correct elasticity prefactor
                # once for contact ij
                #hessianHolder[3 * i:(3 * i + 3),3 * j:(3 * j + 3),counter] = Hij / (mi * mj)**0.5

                # see notes for the flip one corresponding to contact ji
                # both n and t flip signs. Put in here explicitly. Essentially, angle cross-terms flip sign
                # Yes, this is not fully efficient, but it's clearer. Diagonalisation is rate-limiting step, not this.
                # careful with the A's
                subsquare[1, 2] = subsquare[1, 2] * Ai / Aj
                subsquare[2, 1] = subsquare[2, 1] * Aj / Ai
                #Hji = np.zeros((3, 3))
                #Hji[0,0] = subsquare[0, 0] * (-nx0)**2 + subsquare[1, 1] * (-tx0)**2
                #Hji[0, 1] = subsquare[0, 0] * (-nx0) * (-ny0) + subsquare[1, 1] * (-tx0) * (-ty0)
                #Hji[1, 0] = subsquare[0, 0] * (-ny0) * (-nx0) + subsquare[1, 1] * (-ty0) * (-tx0)
                #Hji[1,1] = subsquare[0, 0] * (-ny0)**2 + subsquare[1, 1] * (-ty0)**2
                #Hji[0, 2] = subsquare[1, 2] * (-tx0)
                #Hji[1, 2] = subsquare[1, 2] * (-ty0)
                #Hji[2, 0] = subsquare[2, 1] * (-tx0)
                #Hji[2, 1] = subsquare[2, 1] * (-ty0)
                #Hji[2, 2] = subsquare[2, 2]

                #Finn: Construct ji matrix via transpose
                Hji = np.transpose(Hij)

                # Mike:  Here is the sparse matrix format of what is above.
                #hessianRowHolder = np.append([3*j,3*j,3*j+1,3*j+1,3*j,3*j+1,3*j+2,3*j+2,3*j+2],hessianRowHolder)
                #hessianColHolder = np.append([3*i,3*i+1,3*i,3*i+1,3*i+2,3*i+2,3*i,3*i+1,3*i+2],hessianColHolder)
                #currentMatCValues = [subsquare[0, 0] * (-nx0)**2 + subsquare[1, 1] * (-tx0)**2,subsquare[0, 0] * (-nx0) * (-ny0) + subsquare[1, 1] * (-tx0) * (-ty0),subsquare[0, 0] * (-ny0) * (-nx0) + subsquare[1, 1] * (-ty0) * (-tx0),subsquare[0, 0] * (-ny0)**2 + subsquare[1, 1] * (-ty0)**2,subsquare[1, 2] * (-tx0),subsquare[1, 2] * (-ty0),subsquare[2, 1] * (-tx0),subsquare[2, 1] * (-ty0),subsquare[2, 2]]
                #hessianDataHolder = np.append(currentMatCValues /(mi * mj)**0.5,hessianDataHolder)
                # And put it into the Hessian
                # now for contact ji
                #hessianHolder[3 * j:(3 * j + 3),3 * i:(3 * i + 3),counter] = Hji / (mi * mj)**0.5

                hessianRowHolder = np.append([6*i,6*i,6*i,6*i,6*i,6*i,6*i+1,6*i+1,6*i+1,6*i+1,6*i+1,6*i+1,6*i+2,6*i+2,6*i+2,6*i+2,6*i+2,6*i+2,6*i+3,6*i+3,6*i+3,6*i+3,6*i+3,6*i+3,6*i+4,6*i+4,6*i+4,6*i+4,6*i+4,6*i+4,6*i+5,6*i+5,6*i+5,6*i+5,6*i+5,6*i+5],hessianRowHolder)
                hessianColHolder = np.append([6*j,6*j+1,6*j+2,6*j+3,6*j+4,6*j+5,6*j,6*j+1,6*j+2,6*j+3,6*j+4,6*j+5,6*j,6*j+1,6*j+2,6*j+3,6*j+4,6*j+5,6*j,6*j+1,6*j+2,6*j+3,6*j+4,6*j+5,6*j,6*j+1,6*j+2,6*j+3,6*j+4,6*j+5,6*j,6*j+1,6*j+2,6*j+3,6*j+4,6*j+5],hessianColHolder)
                currentMatCValues = [Hji[0,0],Hji[0,1],Hji[0,2],Hji[0,3],Hji[0,4],Hji[0,5],Hji[1,0],Hji[1,2],Hji[1,3],Hji[1,4],Hji[1,5],Hji[2,0],Hji[2,1],Hji[2,3],Hji[2,4],Hji[2,5],Hji[3,0],Hji[3,1],Hji[3,2],Hji[3,3],Hji[3,4],Hji[3,5],Hji[4,0],Hji[4,1],Hji[4,2],Hji[4,3],Hji[4,4],Hji[4,5],Hji[5,0],Hji[5,1],Hji[5,2],Hji[5,3],Hji[5,4],Hji[5,5]]
                hessianDataHolder = np.append(currentMatCValues/ (mi * mj)**0.5,hessianDataHolder)

                # Careful, the diagonal bits are not just minus because of the rotations
                #diagsquare = np.zeros((3, 3))
                #diagsquare[0, 0] = kn
                #diagsquare[1, 1] = -fn / rval + kt
                #diagsquare[1, 2] = kt * Ai
                #diagsquare[2, 1] = kt * Ai
                #diagsquare[2, 2] = kt * Ai**2

                Hii = np.zeros((6, 6))
                Hii[0, 0] = (((fn/rval)-kt)*(1 - nx0**2)) + kn * nx0**2
                Hii[1, 0] = (kn + (fn/rval) - kt) * nx0 * ny0
                Hii[2, 0] = (kn + (fn/rval) - kt) * nx0 * nz0
                Hii[4, 0] = Ai * kt * nz0
                Hii[5, 0] = -Ai * kt * ny0
                Hii[0, 1] = (kn + (fn/rval) - kt)* nx0 * ny0
                Hii[1, 1] = (((fn/rval)-kt)*(1 - ny0**2)) + kn * ny0**2
                Hii[2, 1] = (kn + (fn/rval) - kt)* ny0 * nz0
                Hii[3, 1] = -Ai * kt * nz0
                Hii[5, 1] = Ai * kt * nx0
                Hii[0, 2] = (kn + (fn/rval) - kt)* nx0 * nz0
                Hii[1, 2] = (kn + (fn/rval) - kt)* ny0 * nz0
                Hii[2, 2] = (((fn/rval)-kt)*(1 - nz0**2)) + kn * nz0**2
                Hii[3, 2] = Ai * kt * ny0
                Hii[4, 2] = -Ai * kt * nx0
                Hii[1, 3] = -Ai * kt * nz0
                Hii[2, 3] = Ai * kt * ny0
                Hii[3, 3] = Ai * Aj * kt * (1 - (nx0**2))
                Hii[4, 3] = -Ai * Aj * kt * nx0 * ny0
                Hii[5, 3] = -Ai * Aj * kt * nx0 * nz0
                Hii[0, 4] = Ai * kt * nz0
                Hii[2, 4] = -Ai * kt * nx0
                Hii[3, 4] = -Ai * Aj * kt * nx0 * ny0
                Hii[4, 4] = Ai * Aj * kt * (1 - (ny0**2))
                Hii[5, 4] = -Ai * Aj * kt * ny0 * nz0
                Hii[0, 5] = -Ai * kt * ny0
                Hii[1, 5] = Ai * kt * nx0
                Hii[3, 5] = -Ai * Aj * kt * nx0 * nz0
                Hii[4, 5] = -Ai * Aj * kt * ny0 * nz0
                Hii[5, 5] = Ai * Aj * kt * (1 - (nz0**2))

                # Stick this into the appropriate places:
                #Hijdiag = np.zeros((3, 3))
                #Hijdiag[0, 0] = diagsquare[0, 0] * nx0**2 + diagsquare[1, 1] * tx0**2
                #Hijdiag[0, 1] = diagsquare[0, 0] * nx0 * ny0 + diagsquare[1, 1] * tx0 * ty0
                #Hijdiag[1, 0] = diagsquare[0, 0] * ny0 * nx0 + diagsquare[1, 1] * ty0 * tx0
                #Hijdiag[1, 1] = diagsquare[0, 0] * ny0**2 + diagsquare[1, 1] * ty0**2
                #Hijdiag[0, 2] = diagsquare[1, 2] * tx0
                #Hijdiag[1, 2] = diagsquare[1, 2] * ty0
                #Hijdiag[2, 0] = diagsquare[2, 1] * tx0
                #Hijdiag[2, 1] = diagsquare[2, 1] * ty0
                #Hijdiag[2, 2] = diagsquare[2, 2]

                # And then *add* it to the diagnual
                #hessianRowHolder = np.append([3*i,3*i,3*i+1,3*i+1,3*i,3*i+1,3*i+2,3*i+2,3*i+2],hessianRowHolder)
                #hessianColHolder = np.append([3*i,3*i+1,3*i,3*i+1,3*i+2,3*i+2,3*i,3*i+1,3*i+2],hessianColHolder)
                #currentMatCValues = [diagsquare[0, 0] * nx0**2 + diagsquare[1, 1] * tx0**2,diagsquare[0, 0] * nx0 * ny0 + diagsquare[1, 1] * tx0 * ty0,diagsquare[0, 0] * ny0 * nx0 + diagsquare[1, 1] * ty0 * tx0,diagsquare[0, 0] * ny0**2 + diagsquare[1, 1] * ty0**2,diagsquare[1, 2] * tx0,diagsquare[1, 2] * ty0,diagsquare[2, 1] * tx0,diagsquare[2, 1] * ty0,diagsquare[2, 2]]
                #hessianDataHolder = np.append(currentMatCValues / mi,hessianDataHolder)

                hessianRowHolder = np.append([6*i,6*i,6*i,6*i,6*i,6*i,6*i+1,6*i+1,6*i+1,6*i+1,6*i+1,6*i+1,6*i+2,6*i+2,6*i+2,6*i+2,6*i+2,6*i+2,6*i+3,6*i+3,6*i+3,6*i+3,6*i+3,6*i+3,6*i+4,6*i+4,6*i+4,6*i+4,6*i+4,6*i+4,6*i+5,6*i+5,6*i+5,6*i+5,6*i+5,6*i+5],hessianRowHolder)
                hessianColHolder = np.append([6*i,6*i+1,6*i+2,6*i+3,6*i+4,6*i+5,6*i,6*i+1,6*i+2,6*i+3,6*i+4,6*i+5,6*i,6*i+1,6*i+2,6*i+3,6*i+4,6*i+5,6*i,6*i+1,6*i+2,6*i+3,6*i+4,6*i+5,6*i,6*i+1,6*i+2,6*i+3,6*i+4,6*i+5,6*i,6*i+1,6*i+2,6*i+3,6*i+4,6*i+5],hessianColHolder)
                currentMatCValues = [Hii[0,0],Hii[0,1],Hii[0,2],Hii[0,3],Hii[0,4],Hii[0,5],Hii[1,0],Hii[1,2],Hii[1,3],Hii[1,4],Hii[1,5],Hii[2,0],Hii[2,1],Hii[2,3],Hii[2,4],Hii[2,5],Hii[3,0],Hii[3,1],Hii[3,2],Hii[3,3],Hii[3,4],Hii[3,5],Hii[4,0],Hii[4,1],Hii[4,2],Hii[4,3],Hii[4,4],Hii[4,5],Hii[5,0],Hii[5,1],Hii[5,2],Hii[5,3],Hii[5,4],Hii[5,5]]
                hessianDataHolder = np.append(currentMatCValues/ mi ,hessianDataHolder)

                #hessianHolder[3 * i:(3 * i + 3), 3 * i:(3 * i + 3),counter] += Hijdiag / mi

                #And once more for the jj contribution, which is the same whizz with the flipped sign of n and t
                # and adjusted A's
                #diagsquare = np.zeros((3, 3))
                #diagsquare[0, 0] = kn
                #diagsquare[1, 1] = -fn / rval + kt
                #diagsquare[1, 2] = kt * Aj
                #diagsquare[2, 1] = kt * Aj
                #diagsquare[2, 2] = kt * Aj**2

                Hjj = np.zeros((6, 6))
                Hjj[0, 0] = (((fn/rval)-kt)*(1 - nx0**2)) + kn * nx0**2
                Hjj[1, 0] = (kn + (fn/rval) - kt) * nx0 * ny0
                Hjj[2, 0] = (kn + (fn/rval) - kt) * nx0 * nz0
                Hjj[4, 0] = Aj * kt * nz0
                Hjj[5, 0] = -Aj * kt * ny0
                Hjj[0, 1] = (kn + (fn/rval) - kt)* nx0 * ny0
                Hjj[1, 1] = (((fn/rval)-kt)*(1 - ny0**2)) + kn * ny0**2
                Hjj[2, 1] = (kn + (fn/rval) - kt)* ny0 * nz0
                Hjj[3, 1] = -Aj * kt * nz0
                Hjj[5, 1] = Aj * kt * nx0
                Hjj[0, 2] = (kn + (fn/rval) - kt)* nx0 * nz0
                Hjj[1, 2] = (kn + (fn/rval) - kt)* ny0 * nz0
                Hjj[2, 2] = (((fn/rval)-kt)*(1 - nz0**2)) + kn * nz0**2
                Hjj[3, 2] = Aj * kt * ny0
                Hjj[4, 2] = -Aj * kt * nx0
                Hjj[1, 3] = -Aj * kt * nz0
                Hjj[2, 3] = Aj * kt * ny0
                Hjj[3, 3] = Aj * Aj * kt * (1 - (nx0**2))
                Hjj[4, 3] = -Aj * Aj * kt * nx0 * ny0
                Hjj[5, 3] = -Aj * Aj * kt * nx0 * nz0
                Hjj[0, 4] = Aj * kt * nz0
                Hjj[2, 4] = -Aj * kt * nx0
                Hjj[3, 4] = -Aj * Aj * kt * nx0 * ny0
                Hjj[4, 4] = Aj * Aj * kt * (1 - (ny0**2))
                Hjj[5, 4] = -Aj * Aj * kt * ny0 * nz0
                Hjj[0, 5] = -Aj * kt * ny0
                Hjj[1, 5] = Aj * kt * nx0
                Hjj[3, 5] = -Aj * Aj * kt * nx0 * nz0
                Hjj[4, 5] = -Aj * Aj * kt * ny0 * nz0
                Hjj[5, 5] = Aj * Aj * kt * (1 - (nz0**2))
    
                #Hjidiag = np.zeros((3, 3))
                #Hjidiag[0, 0] = diagsquare[0, 0] * (-nx0)**2 + diagsquare[1, 1] * (-tx0)**2
                #jidiag[0, 1] = diagsquare[0, 0] * (-nx0) * (-ny0) + diagsquare[1, 1] * (-tx0) * (-ty0)
                #Hjidiag[1, 0] = diagsquare[0, 0] * (-ny0) * (-nx0) + diagsquare[1, 1] * (-ty0) * (-tx0)
                #Hjidiag[1, 1] = diagsquare[0, 0] * (-ny0)**2 + diagsquare[1, 1] * (-ty0)**2
                #Hjidiag[0, 2] = diagsquare[1, 2] * (-tx0)
                #Hjidiag[1, 2] = diagsquare[1, 2] * (-ty0)
                #Hjidiag[2, 0] = diagsquare[2, 1] * (-tx0)
                #Hjidiag[2, 1] = diagsquare[2, 1] * (-ty0)
                #Hjidiag[2, 2] = diagsquare[2, 2]
    
                # And then *add* it to the diagnual
                #hessianHolder[3 * j:(3 * j + 3), 3 * j:(3 * j + 3),counter] += Hjidiag / mj
                #hessianRowHolder = np.append([3*j,3*j,3*j+1,3*j+1,3*j,3*j+1,3*j+2,3*j+2,3*j+2],hessianRowHolder)
                #hessianColHolder = np.append([3*j,3*j+1,3*j,3*j+1,3*j+2,3*j+2,3*j,3*j+1,3*j+2],hessianColHolder)
                #currentMatCValues = [diagsquare[0, 0] * (-nx0)**2 + diagsquare[1, 1] * (-tx0)**2,diagsquare[0, 0] * (-nx0) * (-ny0) + diagsquare[1, 1] * (-tx0) * (-ty0),diagsquare[0, 0] * (-ny0) * (-nx0) + diagsquare[1, 1] * (-ty0) * (-tx0),diagsquare[0, 0] * (-ny0)**2 + diagsquare[1, 1] * (-ty0)**2,diagsquare[1, 2] * (-tx0),diagsquare[1, 2] * (-ty0),diagsquare[2, 1] * (-tx0),diagsquare[2, 1] * (-ty0),diagsquare[2, 2]]
                #hessianDataHolder = np.append(currentMatCValues / mj,hessianDataHolder)
                
                hessianRowHolder = np.append([6*j,6*j,6*j,6*j,6*j,6*j,6*j+1,6*j+1,6*j+1,6*j+1,6*j+1,6*j+1,6*j+2,6*j+2,6*j+2,6*j+2,6*j+2,6*j+2,6*j+3,6*j+3,6*j+3,6*j+3,6*j+3,6*j+3,6*j+4,6*j+4,6*j+4,6*j+4,6*j+4,6*j+4,6*j+5,6*j+5,6*j+5,6*j+5,6*j+5,6*j+5],hessianRowHolder)
                hessianColHolder = np.append([6*j,6*j+1,6*j+2,6*j+3,6*j+4,6*j+5,6*j,6*j+1,6*j+2,6*j+3,6*j+4,6*j+5,6*j,6*j+1,6*j+2,6*j+3,6*j+4,6*j+5,6*j,6*j+1,6*j+2,6*j+3,6*j+4,6*j+5,6*j,6*j+1,6*j+2,6*j+3,6*j+4,6*j+5,6*j,6*j+1,6*j+2,6*j+3,6*j+4,6*j+5],hessianColHolder)
                currentMatCValues = [Hjj[0,0],Hjj[0,1],Hjj[0,2],Hjj[0,3],Hjj[0,4],Hjj[0,5],Hjj[1,0],Hjj[1,2],Hjj[1,3],Hjj[1,4],Hjj[1,5],Hjj[2,0],Hjj[2,1],Hjj[2,3],Hjj[2,4],Hjj[2,5],Hjj[3,0],Hjj[3,1],Hjj[3,2],Hjj[3,3],Hjj[3,4],Hjj[3,5],Hjj[4,0],Hjj[4,1],Hjj[4,2],Hjj[4,3],Hjj[4,4],Hjj[4,5],Hjj[5,0],Hjj[5,1],Hjj[5,2],Hjj[5,3],Hjj[5,4],Hjj[5,5]]
                hessianDataHolder = np.append(currentMatCValues/ mj,hessianDataHolder)

                hessian3rdDimHolder = np.append(counter*np.ones(36),hessian3rdDimHolder)

        #This portion of the code checks to see what the counter is and if it's a multiple of 10 it saves
        if partialSave!=False:
            if counter%100==0 and counter!=0:

                nameOfFileNew=nameOfFile.replace("hessian_","hessian_"+str(lastSaveSnapShot)+"_"+str(snapShotIndex)+"_")
                np.save(nameOfFileNew,np.vstack((hessianRowHolder, hessianColHolder, hessian3rdDimHolder, hessianDataHolder)))
                hessianRowHolder = np.array([])
                hessianColHolder = np.array([])
                hessian3rdDimHolder = np.array([])
                hessianDataHolder = np.array([])
                lastSaveSnapShot = snapShotIndex

        counter=counter+1
        springConstantCounter = springConstantCounter+1

    nameOfFileNew = nameOfFile.replace("hessian_", "hessian_" + str(lastSaveSnapShot) + "_" + str(snapShotIndex) + "_")
    np.save(nameOfFileNew,np.vstack((hessianRowHolder, hessianColHolder,hessian3rdDimHolder,hessianDataHolder)))

def generateHessianFiles(topDir,numParticles,snapShotRange=False):

    
    #hessianDataExtractor(intFile,parFile,dataFile,numParticles,snapShotRange=False)
    #First let's collect all the par, int, and data  files needed from topDir
    intFileHolder = []
    parFileHolder = []
    dataFileHolder = []
    for file in os.listdir(topDir):
        if file.startswith("int_"):
            intFileHolder.append(file)
        if file.startswith("par_"):
            parFileHolder.append(file)
        if file.startswith("data_"):
            dataFileHolder.append(file)
            
    intFileHolder = [os.path.join(topDir,file) for file in intFileHolder]
    parFileHolder = [os.path.join(topDir,file) for file in parFileHolder]
    dataFileHolder = [os.path.join(topDir,file) for file in dataFileHolder]


    if len(intFileHolder)!=len(parFileHolder):
        raise Exception("There are an unequal number of par_ and int_ files in the topDir given.")
    if len(intFileHolder)!=len(dataFileHolder):
        raise Exception("There are an unequal number of data_ and int_ files in the topDir given.")
    if len(parFileHolder)!=len(dataFileHolder):
        raise Exception("There are an unequal number of data_ and par_ files in the topDir given.")
    
    intStressHolder = np.zeros(len(intFileHolder))
    parStressHolder = np.zeros(len(parFileHolder))
    dataStressHolder = np.zeros(len(dataFileHolder))
    
    for i in range(0,len(intFileHolder)):     
        result = re.search('_stress(.*)cl', intFileHolder[i])
        intStressHolder[i] = float(result.group(1))
        result = re.search('_stress(.*)cl', parFileHolder[i])
        parStressHolder[i] = float(result.group(1))
        result = re.search('_stress(.*)cl', dataFileHolder[i])
        dataStressHolder[i] = float(result.group(1))
        
        
    #Now that we have all the int and par files we can begin iterating through them to generate the hessian files
    for i in range(0,len(intFileHolder)):
        currentIntFile = intFileHolder[i]
        currentStress = intStressHolder[i]
        currentDataFile = dataFileHolder[int(np.where(dataStressHolder==currentStress)[0][0])]
        currentParFile = parFileHolder[int(np.where(parStressHolder==currentStress)[0][0])]

  
        radii, springConstants, currentContacts = hessianDataExtractor(currentIntFile,currentParFile,currentDataFile,numParticles,snapShotRange)

        nameOfHessianFile = currentParFile.replace("par_","hessian_").replace(".dat","")
        hessianGenerator(radii,springConstants,currentContacts,topDir,nameOfHessianFile)

def generateSingleHessianFiles(topDir,stressValue,numParticles,snapShotRange=False,partialSave=False):

    
    #hessianDataExtractor(intFile,parFile,dataFile,numParticles,snapShotRange=False)
    #First let's collect all the par, int, and data  files needed from topDir
    intFileHolder = []
    parFileHolder = []
    dataFileHolder = []
    for file in os.listdir(topDir):
        if file.startswith("int_"):
            intFileHolder.append(file)
        if file.startswith("par_"):
            parFileHolder.append(file)
        if file.startswith("data_"):
            dataFileHolder.append(file)
            
    intFileHolder = [os.path.join(topDir,file) for file in intFileHolder]
    parFileHolder = [os.path.join(topDir,file) for file in parFileHolder]
    dataFileHolder = [os.path.join(topDir,file) for file in dataFileHolder]


    if len(intFileHolder)!=len(parFileHolder):
        raise Exception("There are an unequal number of par_ and int_ files in the topDir given.")
    if len(intFileHolder)!=len(dataFileHolder):
        raise Exception("There are an unequal number of data_ and int_ files in the topDir given.")
    if len(parFileHolder)!=len(dataFileHolder):
        raise Exception("There are an unequal number of data_ and par_ files in the topDir given.")
    
    intStressHolder = np.zeros(len(intFileHolder))
    parStressHolder = np.zeros(len(parFileHolder))
    dataStressHolder = np.zeros(len(dataFileHolder))
    
    for i in range(0,len(intFileHolder)):     
        result = re.search('_stress(.*)cl', intFileHolder[i])
        intStressHolder[i] = float(result.group(1))
        result = re.search('_stress(.*)cl', parFileHolder[i])
        parStressHolder[i] = float(result.group(1))
        result = re.search('_stress(.*)cl', dataFileHolder[i])
        dataStressHolder[i] = float(result.group(1))
    
    dataFileIndex = np.where(dataStressHolder==stressValue)[0][0]
    intFileIndex = np.where(intStressHolder==stressValue)[0][0]
    parFileIndex = np.where(parStressHolder==stressValue)[0][0]


    intFile = intFileHolder[intFileIndex]
    dataFile = dataFileHolder[dataFileIndex]
    parFile = parFileHolder[parFileIndex]
    nameOfHessianFile = parFile.replace("par_","hessian_").replace(".dat","")

    radii, springConstants, currentContacts = hessianDataExtractor(intFile,parFile,dataFile,numParticles,snapShotRange)

    firstSnapShot = snapShotRange[0]
    lastSnapShot = snapShotRange[0]+len(currentContacts)
    hessianGenerator(radii,springConstants,currentContacts,topDir,nameOfHessianFile,[firstSnapShot,lastSnapShot],partialSave)

def eigenValueVectorExtraction(topDir,numParticles,numSnapShotsToProcess=-1):
    
    #Find all the files in topDir that start with "hessian_"
    hessianFileHolder = []
    for file in os.listdir(topDir):
        if file.startswith("hessian_"):
            hessianFileHolder.append(file)
    #Add topDir to the beginning of each hessian file found in topDir
    hessianFileHolderNoTopDir = hessianFileHolder
    hessianFileHolder = [os.path.join(topDir,file) for file in hessianFileHolder]
    
    
    #Loop over every hessian file found
    hessianCounter=0
    for currentHessianFile in hessianFileHolder:
        print("currently running" + currentHessianFile)
        
        #Load in the current Hessian and transpose it because otherwise I get
        #confused :(
        currentHessian = np.load(currentHessianFile)
        currentHessian = np.transpose(currentHessian)
        eigenValueName = hessianFileHolderNoTopDir[hessianCounter].replace("hessian_","eigenValues_").replace(".npy","").replace(topDir,"")
        eigenVectorName = hessianFileHolderNoTopDir[hessianCounter].replace("hessian_","eigenVector_").replace(".npy","").replace(topDir,"")
        
        #The format of each hessian file is the the first two rows 
        
    
        firstSnapShot = int(min(currentHessian[:,2]))
        if numSnapShotsToProcess==-1:
            lastSnapShot = int(max(currentHessian[:,2]))
        else:
            lastSnapShot = firstSnapShot+numSnapShotsToProcess
        
        #These files are so large we need to save them every iteration of the upcoming loop.
        #This way we aren't carrying around huge amounts of stuff in the RAM, the tradeoff is that we are going to be
        #saving files very often and lots of them so we should make a directory to hold them
        eigenDirName = currentHessianFile.replace("hessian_","eigenDirectory_").replace(".npy","")
        os.mkdir(os.path.join(eigenDirName))
        
        eigenValueHolder = np.zeros((lastSnapShot+1-firstSnapShot,3*numParticles))
        counter=0        
        for i in range(firstSnapShot,lastSnapShot+1):
            print("counter = " + str(counter))
            
            indicesOfInterest = np.where(currentHessian[:,2]==i)[0]
            hessianOfInterest = currentHessian[indicesOfInterest,:]
            hessianOfInterest = np.delete(hessianOfInterest, 2, 1)
            sparseMatrix = csr_matrix((hessianOfInterest[:,2], (hessianOfInterest[:,0], hessianOfInterest[:,1])), shape = (3*numParticles, 3*numParticles)).toarray()
            sparseMatrix = (0.5)*(sparseMatrix+np.transpose(sparseMatrix))
            
            eigenValueHolder, eigenVectorHolder = LA.eigh(sparseMatrix)
            sparseEigenValues = sparse.csr_matrix(eigenValueHolder)
            sparseEigenVectors = sparse.csr_matrix(eigenVectorHolder)
            
            sparse.save_npz(os.path.join(eigenDirName,eigenValueName+str(counter)), sparseEigenValues)
            sparse.save_npz(os.path.join(eigenDirName,eigenVectorName+str(counter)), sparseEigenVectors)

            counter=counter+1
    
        hessianCounter=hessianCounter+1
        #eigenValueHolder = sparse.csr_matrix(eigenValueHolder)
        
        #sparse.save_npz(eigenValueName, eigenValueHolder)


