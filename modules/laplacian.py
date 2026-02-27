import gc
import time

import nibabel as nib
import numpy as np
import scipy
from scipy.sparse.linalg import lgmres


def loadNiiImages(imageList, scale = False):
    """
    imageList can contaiin both paths to .nii images or loaded nii images
    loads nii images from the paths provided in imageList and returns a list of 3D numpy array representing image data. 
    If numpy data is present in imageList, the same will be returned 
    """
    if scale:
        if type(imageList[0])==str:
            fImage = nib.load(imageList[0])
        else:
            scale = False

    images =[]
    for image in imageList:
        if type(image) == str:
            niiImage = nib.load(image)
            imdata = niiImage.get_fdata()

            # Execution is faster on copied data
            if scale:
                scales = tuple(np.array(niiImage.header.get_zooms()) / np.array(fImage.header.get_zooms()))

                imdata =  scipy.ndimage.zoom(imdata.copy(), scales, order=1)
            images.append(imdata.copy())

        else:
            images.append(image.copy()) 

    if (len(imageList)==1):
        return images[0]
    return images


def get_laplacian_index(z: int, y: int, x: int, shape: tuple):
    """
    Get the flattened index of a voxel at (x,y,z) in a 3D volume of shape 'shape'
    """
    idx  = z * shape[1] * shape[2] + y * shape[2] + x
    # fIndices = fpoints[:,0] * ny*nz + fpoints[:,1] * nz + fpoints[:,2]
    return idx


def get_adjacent_indices(z: int, y: int, x: int, shape: tuple):
    adjacent_indices = [None, None, None, None] # Left, right, up, down
    if x > 0:  # Left
        adjacent_indices[0] = get_laplacian_index(z, y, x - 1, shape)
    if x < shape[2] - 1:  # Right
        adjacent_indices[1] = get_laplacian_index(z, y, x + 1, shape)
    if y > 0:  # Up
        adjacent_indices[2] = get_laplacian_index(z, y - 1, x, shape)
    if y < shape[1] - 1:  # Down
        adjacent_indices[3] = get_laplacian_index(z, y + 1, x, shape)
    return adjacent_indices


def laplacianA3D(shape, boundaryIndices):
    """
    Creates the matrix A that correspond to the linear system of ewuations used to perform laplacian Interpolation on 3D volume images

    Parameters
    --------------
    shape : tuple 3D shape
    boundaryIndices : indices of boundary points when flattened. If (x,y) is pixel x*shape[1]*shape[2]+y*shape[2]+z will be flattened index. 
    """
    k = len(shape)
    # Create X,Y,Z arrays that represent the x,y,z indices of each voxel
    X,Y,Z = np.meshgrid(range(shape[0]), range(shape[1]), range(shape[2]), indexing='ij')

    X=X.flatten().astype(int)
    Y=Y.flatten().astype(int)
    Z=Z.flatten().astype(int)

    # Calculate each voxel's index when flattened
    ids_0  = X* shape[1]*shape[2] + Y*shape[2]+ Z
    data  = np.ones(len(ids_0)) * 2*k
    boundaryIndices = boundaryIndices.astype(int)

    """
    rids: row indices value
    cids: column indices values
    data: Diagonal entries of sparse matrix A. A[rid, rid] = 6 at all non boundary locations. 
          A[rid, rid] = 1 at Dirchlet boundary. A[rid, rid] = number of valid neighbours at volume boundary


    """
    #print("Building data for Laplacian Sparse Matrix A")

    # Calculate the voxel index of (x-1,y,z) for each (x,y,z)
    cids_x1  = (X-1)* shape[1]*shape[2] + Y*shape[2]+ Z
    invalid_cx1 = np.concatenate([np.where(X==0)[0], boundaryIndices])  # invalid column indices and coorespondences indices. X==0 is invalid because X-1 will be negative
    rids_x1  = np.delete(ids_0, invalid_cx1) # remove invalid row indices that correspond to column
    cids_x1  = np.delete(cids_x1, invalid_cx1) # remove invalid column indices
    data[invalid_cx1] -=1   # decrease the value of A[rid, rid] by 1 at invalid indices

    # Calculate the voxel index of (x+1,y,z) for each (x,y,z)
    cids_x2  = (X+1)* shape[1]*shape[2] + Y*shape[2]+ Z
    invalid_cx2 = np.concatenate([np.where(X==shape[0] -1 )[0], boundaryIndices])
    rids_x2  = np.delete(ids_0, invalid_cx2)
    cids_x2  = np.delete(cids_x2, invalid_cx2)
    data[invalid_cx2] -=1

    # Calculate the voxel index of (x,y-1,z) for each (x,y,z)
    cids_y1  = X* shape[1]*shape[2] + (Y-1)*shape[2]+ Z
    invalid_cy1 = np.concatenate([np.where(Y==0)[0], boundaryIndices])
    rids_y1  = np.delete(ids_0, invalid_cy1)
    cids_y1  = np.delete(cids_y1, invalid_cy1)
    data[invalid_cy1] -=1

    # Calculate the voxel index of (x,y+1,z) for each (x,y,z)
    cids_y2  = X* shape[1]*shape[2] + (Y+1)*shape[2]+ Z
    invalid_cy2 = np.concatenate([np.where(Y==shape[1]-1)[0], boundaryIndices])
    rids_y2  = np.delete(ids_0, invalid_cy2)
    cids_y2  = np.delete(cids_y2, invalid_cy2)
    data[invalid_cy2] -=1

    # Calculate the voxel index of (x,y,z-1) for each (x,y,z)
    cids_z1  = X* shape[1]*shape[2] + Y*shape[2]+ Z-1
    invalid_cz1 = np.concatenate([np.where(Z==0)[0], boundaryIndices])
    rids_z1  = np.delete(ids_0, invalid_cz1)
    cids_z1  = np.delete(cids_z1, invalid_cz1)
    data[invalid_cz1] -=1

    # Calculate the voxel index of (x,y,z+1) for each (x,y,z)
    cids_z2  = X* shape[1]*shape[2] +Y*shape[2]+ Z+1
    invalid_cz2 = np.concatenate([np.where(Z==shape[2] - 1)[0], boundaryIndices])   
    rids_z2  = np.delete(ids_0, invalid_cz2)  
    cids_z2  = np.delete(cids_z2, invalid_cz2) 
    data[invalid_cz2] -=1 

    # Diagonal entries corresponding to dirichlet boundaries should be 1
    data[boundaryIndices] +=1

    rowx = np.hstack([ids_0, rids_x1, rids_x2,rids_y1,rids_y2, rids_z1, rids_z2])
    rowy = np.hstack([ids_0, cids_x1, cids_x2,cids_y1,cids_y2, cids_z1, cids_z2])
    rowv = np.hstack([data , -1*np.ones(rowx.shape[0] - ids_0.shape[0]) ] )

    #print("Creating Laplacian Sparse Matrix A")
    A = scipy.sparse.csr_matrix((rowv,(rowx,rowy)), shape =(X.shape[0],X.shape[0]))
    del rowx, rowy, rowv, X, Y, Z, data
    gc.collect()

    return A


def compute3DLaplacianFromShape(shape, mpoints, fpoints):
    """
    The same as sliceToSlice3DLaplacian but with different args
    """
    
    # shape is depth, rows, columns - z,y,x
    (nz, ny, nx)  = (shape[0], shape[1], shape[2])
    # print(f'nz ny nx {nz} {ny} {nx}')
    # number of dimensions of the displacement vectoe
    nd  = len(shape)
    #print("fdata.shape", fdata.shape)
    
    deformationField = np.zeros((nd, nz, ny, nx))
    
    flen  = nx*ny*nz

    # these counts are not used anywhere
    Xcount = np.zeros(flen)
    Ycount = np.zeros(flen)
    Zcount = np.zeros(flen)

    Zd =  np.zeros(flen)
    Yd =  np.zeros(flen)
    Xd =  np.zeros(flen)

    fIndices = fpoints[:,0] * ny*nz + fpoints[:,1] * nz + fpoints[:,2]
    fIndices = fIndices.astype(int)
    
    Xcount[fIndices] +=1 # Added (AT)
    Ycount[fIndices] +=1
    Zcount[fIndices] +=1
    Zd[fIndices] += mpoints[:,0] - fpoints[:,0]  # Added (AT)
    Yd[fIndices] += mpoints[:,1] - fpoints[:,1]
    Xd[fIndices] += mpoints[:,2] - fpoints[:,2]
    
    start = time.time()
    A = laplacianA3D(shape, Ycount.nonzero()[0])
    #print("Saving to npy")
    #np.save("Laplacian_A.npy", A.toarray())
    #np.save("Yd.npy", Yd)
    #np.save("Zd.npy", Zd)
    #print("Computing dz")
    # Define the matrix A and vector B

    # A_ is [[L,0], [0,L]]
    # b_ is [[x],[y]]
    A0 = np.zeros((A.shape[0], A.shape[1]))

    # Creating the big combined Laplacian matrix
    A_ = np.block([
        [A.todense(), A0],
        [A0, A.todense()]
    ])
    b_ = np.concatenate([Xd, Yd])
    phi_xy = lgmres(A_, b_, rtol = 1e-2)[0]
    
    return phi_xy, A_, b_

# def sliceToSlice3DLaplacian(fixedImage, mpoints, fpoints):
    """
    Assumes both the images are matched slice to slice according to sliceMatchList along axis- 'axis'
    Gets 2D correspondences between the slices and interpolates them smoothly across the volume
    """
    fdata = loadNiiImages([fixedImage])
    
    # shape is depth, rows, columns - z,y,x
    nz, ny, nx  = fdata.shape
    # number of dimensions of the displacement vectoe
    nd  = len(fdata.shape)
    #print("fdata.shape", fdata.shape)
    
    deformationField = np.zeros((nd, nz, ny, nx))
    
    flen  = nx*ny*nz

    # these counts are not used anywhere
    Xcount = np.zeros(flen)
    Ycount = np.zeros(flen)
    Zcount = np.zeros(flen)

    Zd =  np.zeros(flen)
    Yd =  np.zeros(flen)
    Xd =  np.zeros(flen)

    fIndices = fpoints[:,0] * ny*nz + fpoints[:,1] * nz + fpoints[:,2]
    fIndices = fIndices.astype(int)
    
    Xcount[fIndices] +=1 # Added (AT)
    Ycount[fIndices] +=1
    Zcount[fIndices] +=1
    Zd[fIndices] += mpoints[:,0] - fpoints[:,0]  # Added (AT)
    Yd[fIndices] += mpoints[:,1] - fpoints[:,1]
    Xd[fIndices] += mpoints[:,2] - fpoints[:,2]
    
    start = time.time()
    A = laplacianA3D(fdata.shape, Ycount.nonzero()[0])
    #print("Saving to npy")
    #np.save("Laplacian_A.npy", A.toarray())
    #np.save("Yd.npy", Yd)
    #np.save("Zd.npy", Zd)
    #print("Computing dz")
    dz = lgmres(A, Zd, rtol = 1e-2)[0]
    #print("dz calculated in {}s".format(time.time() - start))
    
    #print("Computing dy")
    dy = lgmres(A, Yd , rtol = 1e-2)[0]
    #print("dy calculated in {}s".format(time.time() - start))

    #print("Computing dx")
    dx = lgmres(A, Xd, rtol = 1e-2)[0]
    #print("dx calculated in {}s".format(time.time() - start))

    # changed deformationField[0] to get displacement of z-coordinates
    #deformationField[0] = np.zeros(fdata.shape)
    deformationField[0] = dz.reshape(fdata.shape)
    deformationField[1] = dy.reshape(fdata.shape)
    deformationField[2] = dx.reshape(fdata.shape)
    
    return deformationField, A, Zd, Yd, Xd

def sliceToSlice3DLaplacian(fixedImage, mpoints, fpoints):
    """
    Assumes both the images are matched slice to slice according to sliceMatchList along axis- 'axis'
    Gets 2D correspondences between the slices and interpolates them smoothly across the volume
    """
    fdata = loadNiiImages([fixedImage])
    
    nx, ny, nz  = fdata.shape
    nd  = len(fdata.shape)
    #print("fdata.shape", fdata.shape)
    
    deformationField = np.zeros((nd, nx, ny, nz))
    
    flen  = nx*ny*nz
    Xcount = np.zeros(flen)
    Ycount = np.zeros(flen)
    Zcount = np.zeros(flen)

    Xd =  np.zeros(flen)
    Yd =  np.zeros(flen)
    Zd =  np.zeros(flen)

    fIndices = fpoints[:,0] * ny*nz + fpoints[:,1] * nz + fpoints[:,2]
    fIndices = fIndices.astype(int)
    
    Xcount[fIndices] +=1 # Added (AT)
    Ycount[fIndices] +=1
    Zcount[fIndices] +=1
    Xd[fIndices] += mpoints[:,0] - fpoints[:,0]  # Added (AT)
    Yd[fIndices] += mpoints[:,1] - fpoints[:,1]
    Zd[fIndices] += mpoints[:,2] - fpoints[:,2]
    
    start = time.time()
    A = laplacianA3D(fdata.shape, Ycount.nonzero()[0])
    #print("Saving to npy")
    #np.save("Laplacian_A.npy", A.toarray())
    #np.save("Yd.npy", Yd)
    #np.save("Zd.npy", Zd)
    #print("Computing dz")
    dx = lgmres(A, Xd, rtol = 1e-2)[0]
    #print("dz calculated in {}s".format(time.time() - start))
    
    #print("Computing dy")
    dy = lgmres(A, Yd , rtol = 1e-2)[0]
    #print("dz calculated in {}s".format(time.time() - start))

    #print("Computing dy")
    dz = lgmres(A, Zd, rtol = 1e-2)[0]
    #print("dx calculated in {}s".format(time.time() - start))

    deformationField[0] = np.zeros(fdata.shape)
    #deformationField[0] = dx.reshape(fdata.shape)
    deformationField[1] = dy.reshape(fdata.shape)
    deformationField[2] = dz.reshape(fdata.shape)
    
    return deformationField, A, Xd, Yd, Zd

def createA(fixedImage, mpoints, fpoints):
    """
    Assumes both the images are matched slice to slice according to sliceMatchList along axis- 'axis'
    Gets 2D correspondences between the slices and interpolates them smoothly across the volume
    """
    
    nx, ny, nz  = fixedImage.shape
    nd  = len(fixedImage.shape)
    print("fdata.shape", fixedImage.shape)
    
    deformationField = np.zeros((nd, nx, ny, nz))
    
    flen  = nx*ny*nz
    Xcount = np.zeros(flen)
    Ycount = np.zeros(flen)
    Zcount = np.zeros(flen)

    Xd =  np.zeros(flen)
    Yd =  np.zeros(flen)
    Zd =  np.zeros(flen)

    fIndices = fpoints[:,0] * ny*nz + fpoints[:,1] * nz + fpoints[:,2]
    fIndices = fIndices.astype(int)
    
    Xcount[fIndices] +=1 # Added (AT)
    Ycount[fIndices] +=1
    Zcount[fIndices] +=1
    Xd[fIndices] += mpoints[:,0] - fpoints[:,0]  # Added (AT)
    Yd[fIndices] += mpoints[:,1] - fpoints[:,1]
    Zd[fIndices] += mpoints[:,2] - fpoints[:,2]
    
    start = time.time()
    A = laplacianA3D(fixedImage.shape, Ycount.nonzero()[0])
    end = time.time()
    print("Time taken to create A: ", end-start)
    return A


def createA_no_correspondences(fixedImage, mpoints, fpoints):
    """
    Assumes both the images are matched slice to slice according to sliceMatchList along axis- 'axis'
    Gets 2D correspondences between the slices and interpolates them smoothly across the volume
    """
    
    nx, ny, nz  = fixedImage.shape
    nd  = len(fixedImage.shape)
    print("fdata.shape", fixedImage.shape)
    
    deformationField = np.zeros((nd, nx, ny, nz))
    
    flen  = nx*ny*nz
    Xcount = np.zeros(flen)
    Ycount = np.zeros(flen)
    Zcount = np.zeros(flen)

    Xd =  np.zeros(flen)
    Yd =  np.zeros(flen)
    Zd =  np.zeros(flen)

    fIndices = fpoints[:,0] * ny*nz + fpoints[:,1] * nz + fpoints[:,2]
    fIndices = fIndices.astype(int)
    
    Xcount[fIndices] +=1 # Added (AT)
    Ycount[fIndices] +=1
    Zcount[fIndices] +=1
    Xd[fIndices] += mpoints[:,0] - fpoints[:,0]  # Added (AT)
    Yd[fIndices] += mpoints[:,1] - fpoints[:,1]
    Zd[fIndices] += mpoints[:,2] - fpoints[:,2]
    
    start = time.time()
    A = laplacianA3D_no_correspondences(fixedImage.shape, Ycount.nonzero()[0])
    end = time.time()
    print("Time taken to create A: ", end-start)
    return A


def laplacianA3D_no_correspondences(shape, boundaryIndices):
    """
    Creates the matrix A that correspond to the linear system of ewuations used to perform laplacian Interpolation on 3D volume images

    Parameters
    --------------
    shape : tuple 3D shape
    boundaryIndices : indices of boundary points when flattened. If (x,y) is pixel x*shape[1]*shape[2]+y*shape[2]+z will be flattened index. 
    """
    k = len(shape)
    # Create X,Y,Z arrays that represent the x,y,z indices of each voxel
    X,Y,Z = np.meshgrid(range(shape[0]), range(shape[1]), range(shape[2]), indexing='ij')

    X=X.flatten().astype(int)
    Y=Y.flatten().astype(int)
    Z=Z.flatten().astype(int)

    # Calculate each voxel's index when flattened
    ids_0  = X* shape[1]*shape[2] + Y*shape[2]+ Z
    data  = np.ones(len(ids_0)) * 2*k
    boundaryIndices = boundaryIndices.astype(int)
    boundaryIndices = [0]
    print("Boundary Indices")
    print(boundaryIndices)

    """
    rids: row indices value
    cids: column indices values
    data: Diagonal entries of sparse matrix A. A[rid, rid] = 6 at all non boundary locations. 
          A[rid, rid] = 1 at Dirchlet boundary. A[rid, rid] = number of valid neighbours at volume boundary


    """
    print("Building data for Laplacian Sparse Matrix A")

    # Calculate the voxel index of (x-1,y,z) for each (x,y,z)
    cids_x1  = (X-1)* shape[1]*shape[2] + Y*shape[2]+ Z
    invalid_cx1 = np.concatenate([np.where(X==0)[0], boundaryIndices])  # invalid column indices and coorespondences indices. X==0 is invalid because X-1 will be negative
    rids_x1  = np.delete(ids_0, invalid_cx1) # remove invalid row indices that correspond to column
    cids_x1  = np.delete(cids_x1, invalid_cx1) # remove invalid column indices
    print("Invalid Column Indices")
    print(invalid_cx1)
    print("rids_x1")
    print(rids_x1)
    print("cids_x1")
    print(cids_x1)
    data[invalid_cx1] -=1   # decrease the value of A[rid, rid] by 1 at invalid indices

    # Calculate the voxel index of (x+1,y,z) for each (x,y,z)
    cids_x2  = (X+1)* shape[1]*shape[2] + Y*shape[2]+ Z
    invalid_cx2 = np.concatenate([np.where(X==shape[0] -1 )[0], boundaryIndices])
    rids_x2  = np.delete(ids_0, invalid_cx2)
    cids_x2  = np.delete(cids_x2, invalid_cx2)
    data[invalid_cx2] -=1

    # Calculate the voxel index of (x,y-1,z) for each (x,y,z)
    cids_y1  = X* shape[1]*shape[2] + (Y-1)*shape[2]+ Z
    invalid_cy1 = np.concatenate([np.where(Y==0)[0], boundaryIndices])
    rids_y1  = np.delete(ids_0, invalid_cy1)
    cids_y1  = np.delete(cids_y1, invalid_cy1)
    data[invalid_cy1] -=1

    # Calculate the voxel index of (x,y+1,z) for each (x,y,z)
    cids_y2  = X* shape[1]*shape[2] + (Y+1)*shape[2]+ Z
    invalid_cy2 = np.concatenate([np.where(Y==shape[1]-1)[0], boundaryIndices])
    rids_y2  = np.delete(ids_0, invalid_cy2)
    cids_y2  = np.delete(cids_y2, invalid_cy2)
    data[invalid_cy2] -=1

    # Calculate the voxel index of (x,y,z-1) for each (x,y,z)
    cids_z1  = X* shape[1]*shape[2] + Y*shape[2]+ Z-1
    invalid_cz1 = np.concatenate([np.where(Z==0)[0], boundaryIndices])
    rids_z1  = np.delete(ids_0, invalid_cz1)
    cids_z1  = np.delete(cids_z1, invalid_cz1)
    data[invalid_cz1] -=1

    # Calculate the voxel index of (x,y,z+1) for each (x,y,z)
    cids_z2  = X* shape[1]*shape[2] +Y*shape[2]+ Z+1
    invalid_cz2 = np.concatenate([np.where(Z==shape[2] - 1)[0], boundaryIndices])   
    rids_z2  = np.delete(ids_0, invalid_cz2)  
    cids_z2  = np.delete(cids_z2, invalid_cz2) 
    data[invalid_cz2] -=1 

    # Diagonal entries corresponding to dirichlet boundaries should be 1
    #data[boundaryIndices] +=1

    rowx = np.hstack([ids_0, rids_x1, rids_x2,rids_y1,rids_y2, rids_z1, rids_z2])
    rowy = np.hstack([ids_0, cids_x1, cids_x2,cids_y1,cids_y2, cids_z1, cids_z2])
    rowv = np.hstack([data , -1*np.ones(rowx.shape[0] - ids_0.shape[0]) ] )

    print("Creating Laplacian Sparse Matrix A (no correspondences)")
    A = scipy.sparse.csr_matrix((rowv,(rowx,rowy)), shape =(X.shape[0],X.shape[0]), dtype=np.int8)
    del rowx, rowy, rowv, X, Y,Z, data
    gc.collect()

    return A
