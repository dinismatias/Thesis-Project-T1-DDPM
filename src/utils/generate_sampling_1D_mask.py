"""
generate_sampling_1D_mask.py
Generate 1D compressed sensing sampling masks
Translated from MATLAB code by Marcelo V. W. Zibetti
"""

import numpy as np


def generate_sampling_1D_mask(dim, accel, mask_type, center=True):
    """
    Generate 1D compressed sensing sampling mask
    
    Parameters:
    -----------
    dim : array-like
        Dimensions [Nx, Ny, Nt] of the sampling mask
    accel : float
        Acceleration factor = length(sample_mask) / sum(sample_mask)
    mask_type : str
        Type of sampling mask:
        - 'out_r_1D': Fully sampled inner area + random outer area
        - 'on_in_1D': Only inner area sampled
        - 'full_sam': Fully sampled
    center : bool, optional
        If True, use larger center region. Default is True.
    
    Returns:
    --------
    sample_mask : ndarray
        Binary sampling mask of shape (Nx, Ny, Nt)
    
    Examples:
    ---------
    >>> mask = generate_sampling_1D_mask([128, 64, 1], 4, 'out_r_1D', center=True)
    >>> print(mask.shape)
    (128, 64, 1)
    """
    
    # Ensure dim is a numpy array
    dim = np.array(dim, dtype=int)
    
    # Initialize mask
    sample_mask = np.zeros(dim)
    
    if accel > 1:
        if mask_type == 'out_r_1D':
            # Mask with fully sampled inner area and random outer area
            
            # Create inner area mask
            if center:
                # Larger center region
                ps = 1/2 - 1/(2*3*4)  # 0.5 - 0.0417 = 0.4583
                pf = 1/2 + 1/(2*3*4)  # 0.5 + 0.0417 = 0.5417
            else:
                # Smaller center region
                ps = 1/2 - 1/(6*3*4)  # 0.5 - 0.0139 = 0.4861
                pf = 1/2 + 1/(6*3*4)  # 0.5 + 0.0139 = 0.5139
            
            # Define center region indices
            p1 = np.arange(0, dim[0])  # All rows (0-indexed)
            p2_start = int(np.round(ps * dim[1]))
            p2_end = int(np.round(pf * dim[1]))
            p2 = np.arange(p2_start, p2_end)
            
            # Create inner mask
            inmask = np.zeros(dim[:2])
            inmask[np.ix_(p1, p2)] = 1
            
            # For each time frame
            for tim in range(dim[2]):
                inlength = np.sum(inmask[0, :])
                fulllength = dim[1]
                outlength = int(np.floor(fulllength / accel) - inlength)
                
                # Find positions not in inner mask
                pos = np.where(inmask[0, :] == 0)[0]
                
                # Random permutation
                pos = pos[np.random.permutation(len(pos))]
                
                # Select first outlength positions
                if outlength > 0:
                    pos = pos[:outlength]
                else:
                    pos = np.array([], dtype=int)
                
                # Create random outer mask
                rmask = np.zeros((1, dim[1]))
                if len(pos) > 0:
                    rmask[0, pos] = 1
                
                # Replicate to all rows
                rmask = np.tile(rmask, (dim[0], 1))
                
                # Combine inner and outer masks
                sample_mask[:, :, tim] = inmask + rmask
        
        elif mask_type == 'on_in_1D':
            # Mask with only inner area sampled
            
            # Create inner area mask
            if center:
                ps = 1/2 - 1/(2*3*4)
                pf = 1/2 + 1/(2*3*4)
            else:
                ps = 1/2 - 1/(6*3*4)
                pf = 1/2 + 1/(6*3*4)
            
            # Define center region
            p1 = np.arange(0, dim[0])
            p2_start = int(np.round(ps * dim[1]))
            p2_end = int(np.round(pf * dim[1]))
            p2 = np.arange(p2_start, p2_end)
            
            # Create inner mask
            inmask = np.zeros(dim[:2])
            inmask[np.ix_(p1, p2)] = 1
            
            # Apply to all time frames
            for tim in range(dim[2]):
                sample_mask[:, :, tim] = inmask
        
        elif mask_type == 'full_sam':
            # Fully sampled
            sample_mask = np.ones(dim)
        
        else:
            raise ValueError(f"Unknown mask_type: {mask_type}")
    
    else:
        # No acceleration
        sample_mask = np.ones(dim)
    
    return sample_mask


if __name__ == "__main__":
    # Test the function
    print("Testing generate_sampling_1D_mask...")
    
    # Test 1: out_r_1D with center
    mask = generate_sampling_1D_mask([128, 64, 1], 4, 'out_r_1D', center=True)
    print(f"\nTest 1: out_r_1D, accel=4, center=True")
    print(f"  Shape: {mask.shape}")
    print(f"  Sampled points: {np.sum(mask)}")
    print(f"  Expected points: ~{128*64/4:.0f}")
    print(f"  Actual accel: {128*64/np.sum(mask):.2f}")
    
    # Test 2: on_in_1D
    mask = generate_sampling_1D_mask([128, 64, 1], 4, 'on_in_1D', center=True)
    print(f"\nTest 2: on_in_1D, accel=4, center=True")
    print(f"  Shape: {mask.shape}")
    print(f"  Sampled points: {np.sum(mask)}")
    
    # Test 3: Multi-frame
    mask = generate_sampling_1D_mask([128, 64, 5], 4, 'out_r_1D', center=True)
    print(f"\nTest 3: out_r_1D, Nt=5")
    print(f"  Shape: {mask.shape}")
    print(f"  Sampled points per frame:")
    for t in range(5):
        print(f"    Frame {t}: {np.sum(mask[:, :, t])}")
    
    print("\n✓ All tests completed!")
