"""
    CRFix(nz)

SIRS cosmic ray fixer.

    Parameters: nz::Int
                  Number of up-the-ramp frames in integrations.
"""
mutable struct CRFix
    
    # Declarations
    nz    # Number of up-the-ramp frames
    𝒦     # Fourier transform of finder kernel
    A     # Apodizer function
    w     # Finder kernel width in frames
    j_min # Minimum size for jumpt to be considered a cosmic ray
    
    function CRFix(nz)
        
        # Defaults
        j_min = Float32(0.0)
        
        # Make a cosmic ray finder kernel
        σ = 4       # Kernel standard deviation in frames
        w = 2*4σ    # Kernel width. Goes to zero outside this. This should be even.
        l = nz + 2w # Kernel length sized to allow apodization
        μ = l÷2+0.5 # Kernel center
        
        # Make the basic Gaussian shape
        x = collect(1:l)
        K = (1/(σ*sqrt(2π))) * exp.(-0.5((x.-μ)./σ).^2) # Normal distribution
        
        # Zero out stuff far away from the peak
        μ = Int(ceil(μ)) 
        K[(x .< μ-w÷2) .| (x .> μ+w÷2-1)] .= 0
        
        # Make apodizer
        A = 1 .- ifftshift(K./maximum(K))

        # Flip bottom half of kernel
        K[1:μ-1] .*= -1

        # Renormalize
        K *= -2

        # Shift
        K = ifftshift(K)

        # We require the Fourier transform of K
        𝒦 = reshape(CuArray(ComplexF32.(rfft(K))), (1,1,:))

        # We require a 32-bit apodizer
        A = reshape(CuArray(Float32.(A)), (1,1,:))
        
        # Instantiation
        new(nz, 𝒦, A, w, j_min)
        
    end

end



"""
    crfix(F, D; init=true)

SIRS cosmic ray fixing function. This uses convolution up-the-ramp
to find cosmic ray hits. Once found, it subtracts the hit amplitude from all
subsequent samples.

It may be necessary to call this more than once for pixels that have been
hit more than once.

    Parameters: F
                  A SIRS CRFix struct.
                D
                  Data to be scrubbed for cosmic rays.
                init
                  Initialize the cosmic ray finder. Initialization
                  computes the minimum cosmic ray amplitude to consider.
       Returns: D
                
                  
"""
function crfix(F::CRFix, D::CuArray{Float32,3}; init::Bool=true)

    # Get dimensions
    nx,ny,nz = size(D)
    
    # Extend the ends of the datacube to allow apodization
    L = reshape(D[:,:,1], (nx,ny,1)) .* CuArray(ones(Float32, (nx,ny,F.w)))

    R = reshape(D[:,:,end], (nx,ny,1)) .* CuArray(ones(Float32, (nx,ny,F.w)))
    D = cat(L, D, R, dims=3)
    
    # Apodize
    D .*= F.A
    
    # Convolve with the finder. This gives the jump
    # size for every frame transition and every pixel.
    J = irfft(rfft(D, 3) .* F.𝒦, size(D,3), 3)
    
    # Crop to just real frames
    J = J[:,:,F.w+1:end-F.w] # Jumps
    D = D[:,:,F.w+1:end-F.w] # And data
    
    # If init is true, set j_min using median absolute deviation
    # Use a 4-sigma clip.
    if init == true
        μ = median(J)
        σ = 1.4826 * median(abs.(J.-μ))
        F.j_min = μ + 4σ
        # println("Minimum jump size (DN) = ", F.j_min)
    end
    
    #= ***** This section should be revisited some day. Julia's CUDA package *****
    does not provide a GPU-enabled partialpermsort. Do it in the CPUs.
    This slows things down.
    
    # CPU version
    # Find hit frame indices. Fix only biggest hits per pixel.
    _J = Array(J) # Download to CPU
    J_z = Matrix{Int32}(undef, (128,128)) # Frame indices of hits go here
    Threads.@threads for c in 1:ny
        for r in 1:nx
            # The finder returns the last undisturbed sample.
            # We want the disturbed one which is one more.
            J_z[r,c] = partialsortperm(_J[r,c,:],1; rev=true) + 1
        end
    end
    J_z = CuArray(J_z) # Upload to GPU

    # Find hit amplitudes
    J_amp = dropdims(maximum(J, dims=3), dims=3)
    
    # Fix
    for _z in 1:nz
        M = _z .== J_z                       # Mask of pixels to fix in this frame
        D[M,_z:nz] .= D[M,_z:nz] .- J_amp[M] # Fix
    end
    # ***** END CPU CODE ***** =#
    
    # ***** BEGIN PURE GPU VERSION *****  
    # Cosmic rays are the maximum jump in each pixel
    J_amp = dropdims(maximum(J, dims=3), dims=3)
    for _z in 1:nz
        M = (J[:,:,_z] .== J_amp) .& (J_amp .> F.j_min) # Mask of pixels to fix in next frame
        D[M,_z+1:nz] .= D[M,_z+1:nz] .- J_amp[M] # Fix
    end   
    # ***** END PURE GPU VERSION
    
    # Done
    return(D)
    
end
