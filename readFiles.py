import numpy as np # type: ignore
import random
import ctypes

'''
Revision History:
Mar  9, 2025 RVP - Initial version of the script.
May 30, 2025 RVP - Major error correction. nteraction, rigid 
                   and particle list did not append the last strain
                   as a result the total snapshots were one less in 
                   dimensions.
Jul 21, 2025 RVP - Updated rigClusterList() to make it nested list
                   earlier it was a list of strings which included 
                   zero clusters too ('0').

NOTE: Script contains functions to read LF-DEM simulation files
'''

def rigClusterList(rigFile):
    '''
    This function reads the rig_*.dat and creates a list of particle index 
    in rigid cluster. Each element in a list is a list with rigid particle index
    from a cluster. 

    Inputs: rigFile - open(path/to/rig_* file)
    Output: len(output) = no. of clusters
            output sample: [[210,600,550,600], .....,[224,224,948,775,948]]
            
    NOTE: 1. The list elements may have repeated index numbers. Filter before processing. 
          2. The output list is nested list with each element being the IDs of a cluster.
          3. Snapshots with no clusters are excluded from the output, hence the length of a nested list
             is always greater than 2.
    '''
    hashCounter = -3
    clusterIDs  = []
    for line in rigFile:
        if line[0] == '#':
            hashCounter += 1
        elif hashCounter >= 0:
            IDs_string = line.split()[0]
            IDs_list   = [int(x.strip()) for x in IDs_string.split(',') if x.split()]
            if len(IDs_list) > 2:
                clusterIDs.append(IDs_list)
    return clusterIDs

def rigList(rigFile):
    '''
    This function reads the rig_*.dat and creates a nested-list of particle index 
    in rigid cluster. Each element in a list (for each timestep) is a nested 
    list (for each cluster) with rigid particle index. 

    Input : open(path/to/rig_*.dat file)
    Output: [[[0]], [[97, 235, 97], [174, 201, 488]], ...,[[381, 235, 381]]]
            len(output) = no. of timesteps

    NOTE: The list elements may have repeated index numbers. Filter before processing.
    '''
    hashCounter = -3
    clusterIDs  = []
    temp = []
    for line in rigFile:
        if line[0] == '#':
            hashCounter += 1
            if len(temp) > 0:
                clusterIDs.append(temp)
                temp = []
        elif hashCounter >= 0:
            temp.append(line.strip())
    if temp:
        clusterIDs.append(temp)

    rigClusterIDsList = []
    for sampleList in clusterIDs:
        tempList = []
        for kk in range(len(sampleList)):
            tempList.append([int(indx) for indx in sampleList[kk].split(',')])
        rigClusterIDsList.append(tempList)
    return rigClusterIDsList

def particleSizeList(randomSeedFile, sizeRatio=1.4, npp = 1000):
    '''
    This function reads the random seed file and creates
    a list of particle sizes. The list index is the particle index.

    Inputs:
    randomSeedFile - open(path/to/random_seed.dat file)
    sizeRatio      - delta or ar for the case
    npp            - system size
    '''

    if sizeRatio == 1:
        # Painting particles randomly in two colours for monodisperse case.
        particleSize = [1]*(int(npp/2)) + [2]*(int(npp/2))
        random.shuffle(particleSize)
    else:
        particleSize = np.loadtxt(randomSeedFile, usecols = 3) # reading only column 3 which has particle size
        randomSeedFile.close()
    return particleSize


def interactionsList(interactionFile):
    '''
    This function reads the interaction file and creates a list of arrays.
    Each array contains interaction parameters for a timestep

    Input : interactionFile - open(path/to/int_* file)
    Output: len(output) = no. of timesteps
            output sample: [arr(t0), arr(t1)...., arr(tn)]
            arr(tn).shape = (n, 17) where n is the no. of interactions in a timestep
    '''

    hashCounter = 0
    temp        = []
    contactList = [] # list with interaction parameters for each element at each timestep

    fileLines = interactionFile.readlines()[27:] # skipping the comment lines
    for line in fileLines:
        if not line.split()[0] == '#':
            lineList = [float(value) for value in line.split()]
            temp.append(lineList)
        else:
            hashCounter += 1 # checking if counter reaches 7 (7 lines of comments after every timestep data)
            if hashCounter == 7: 
                contactList.append(np.array(temp))
                temp        = []
                hashCounter = 0

    if temp:
        contactList.append(np.array(temp))

    interactionFile.close()
    return contactList

def readParFile(particleFile):
    '''
    This function reads the particle file (contains particle wise information) and creates a list of arrays,
    each array contains all individual particle parameters for that timestep.

    NOTE: this function is same as readParFile2() function. Structed differently
    to set move the internal file pointer back to the required file.

    Input : ParametersFile - open(path/to/particle file)
    Output: output = [arr(t0), arr(t1)...., arr(tn)]
            len(output) = total snapshots 
            arr(tn).shape = (npp, 11) where npp is the no. of particles in system
           
    '''
    particleFile.seek(0)
    hashCounter   = 0
    temp          = []
    particlesList = []

    fileLines = particleFile.readlines()[22:] # skipping the comment lines

    for line in fileLines:
        if not line.split()[0] == '#':
            lineList = [float(value) for value in line.split()]
            temp.append(lineList)
        else:
            # Checking if counter reaches 7 (7 lines of comments after every timestep data).
            hashCounter += 1 
            if hashCounter == 7: 
                particlesList.append(np.array(temp))
                temp        = []
                hashCounter = 0

    # to append last the strain data
    if temp:
        particlesList.append(np.array(temp))

    particleFile.close()
    return particlesList

def readParFile2(particleFile):
    ## formerly parametersList()
    '''
    This function reads the particle file (contains particle wise information) and creates a list of arrays,
    each array contains all individual particle parameters for that timestep.

    NOTE: This version of reading the particle file sets the pointer and opens the file automatically

    Input : ParametersFile - path/to/particle file
    Output: output = [arr(t0), arr(t1)...., arr(tn)]
            len(output) = total snapshots 
            arr(tn).shape = (npp, 11) where npp is the no. of particles in system
    '''

    parFile = open(particleFile, 'r')

    hashCounter = 0
    temp        = []
    parList     = [] # list with parameters parameters for each element at each timestep

    fileLines = parFile.readlines()[22:] # skipping the comment lines
    for line in fileLines:
        if not line.split()[0] == '#':
            lineList = [float(value) for value in line.split()]
            temp.append(lineList)
        else:
            hashCounter += 1 # checking if counter reaches 7 (7 lines of comments after every timestep data)
            if hashCounter == 7: 
                parList.append(np.array(temp))
                temp        = []
                hashCounter = 0

    if temp:
        parList.append(np.array(temp))

    parFile.close()
    return parList

def readParFile3(particleFile, final_strain = 20):
    '''
    - This function reads the particle file (contains particle wise information) and creates a list of arrays,
      each array contains all individual particle parameters for that timestep.
    - This function output is exactly same as readParFile() and readParFile2() function. But this is much faster
      as it does not read the file line by line. It reads the entire file at once. Hence, more convenient for large files 
      when the total strain is known.

    Input : open('path/to/particle_file') and final_strain (final strain value in the simulation)
    Output: output        = [arr(t0), arr(t1)...., arr(tn)]
            len(output)   = total snapshots 
            arr(tn).shape = (npp, 11) where npp is the no. of particles in system
    '''
    particleFile.seek(0)
    data    = np.loadtxt(particleFile, comments='#')    # reading the file excluding comments
    parList = np.split(data, int((final_strain)*100)+1) # splitting the data into the snapshots
    particleFile.close()

    return parList

def readParFile4(particleFile, npp = 1000):
    '''
    - This function reads the particle file (contains particle wise information) and creates a list of arrays,
      each array contains all individual particle parameters for that timestep.
    - This function output is exactly same as previous readParFile*() functions. But this is much faster
      as it does not read the file line by line. It reads the entire file at once. Hence, more convenient for large files 
      when the total number of particles is known is known.

    Input : open('path/to/particle_file') and npp (number of particles in the system)
    Output: output        = [arr(t0), arr(t1)...., arr(tn)]
            len(output)   = total snapshots 
            arr(tn).shape = (npp, 11) where npp is the no. of particles in system
    '''
    particleFile.seek(0)
    data      = np.loadtxt(particleFile, comments='#')    # reading the file excluding comments
    tot_lines = data.shape[0]
    parList   = np.split(data, int(tot_lines/npp)) # splitting the data into the snapshots
    particleFile.close()

    return parList


def free_mem(*arrays):
    '''
    To free up memory instead of just using del
    efficacy not tested yet
    '''
    for arr in arrays:
        if isinstance(arr, np.ndarray):
            ctypes.memset(arr.ctypes.data, 0, arr.nbytes)