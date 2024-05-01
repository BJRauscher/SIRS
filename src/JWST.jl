"""
    nircam_rowcor(D::CuArray{Float32,3})

Rows-only reference correction for NIRCam. This treats odd/even columns
independently. It treats outputs independently. The pipeline's rows-only
reference correction seems to have a problem.

    Parameters: D
                  A NIRCam datacube.
       Returns: D
                  The reference corrected datacube.
"""
function nircam_rowcor(D::CuArray{Float32,3})

    # Definitions
    count = 16   # Discard this many low/high reference pixels for robustness
    nx    = 2048 # Number of image columns
    ny    = 2048 # Number of image rows
    nout  = 4    # Number of H2RG outputs

    # Reformat
    D = reshape(D, (2,256,nout,ny,:))                 # size(D) = (2, 256, 4, 2048, 30)
    D = permutedims(D, (2,4,1,3,5))                   # size(D) = (256, 2048, 2, 4, 30)

    # Pick off bottom, B, and top, T, reference pixels treating odd/even columns separately.
    # treating outputs individually. Use the inner 2 rows of reference pixels on the top
    # and bottom. These tend to be the most stable.
    B = reshape(D[:,2:3,:,:,:], (1,512,2,nout,:))       # size(B) = (1, 512, 2, 4, 30)
    T = reshape(D[:,2046:2047,:,:,:], (1,512,2,nout,:)) # size(T) = (1, 512, 2, 4, 30)

    # Robustly compute mean values. Means are in the 2nd index.
    B = mean(sort(B, dims=2)[:,count+1:end-count,:,:,:], dims=2)
    T = mean(sort(T, dims=2)[:,count+1:end-count,:,:,:], dims=2)

    # Compute reference planes
    Y = Int32.(reshape(CuArray(collect(1:ny)), (1,:,1,1,1)))
    S = (T .- B) / Float32(2044) # Slope
    R = B .+ S.*(Y.-Float32(2.5))                              # size(R) = (1, 2048, 2, 4, 30)

    # Reference correct
    D = D .- R

    # Restore shape
    D = reshape(permutedims(D, (3,1,4,2,5)), (nx,ny,:))
    
    # Done!
    D
    
end




"""
    h5write(prog_id, f, γ, ζ, ps_𝓵, ps_𝓻, ps_𝓷, scss, output_filename)

Write JWST NIRCam SIRS calibration files.

    Parameters: prog_id
                  JWST Program ID
                f
                  Fourier frequencies. 
                γ
                  SIRS gamma coefficients
                ζ
                  SIRS zeta coefficients
                ps_𝓵
                  Power spectrum of left reference columns.
                  The units are DN^2/bin.
                ps_𝓻
                  Power spectrum of right reference columns.
                  The units are DN^2/bin.
                ps_𝓷
                  Power spectrum of normal pixels.
                  The units are DN^2/bin.
                scss
                  List of NIRCam Sensor Chip Systems
                output_filename
                  Write results to this file.      
       Returns: This function write a file and exits.
"""
function h5write(prog_id::String, f::Vector{Float32}, γ, ζ, 
        ps_𝓵, ps_𝓻, ps_𝓷, scss::Vector{String}, output_filename::String)
    
    # Open the file. It is closed upon exiting this block.
    h5open(output_filename, "w") do fid
    
        # The input files came from this Program ID
        write_attribute(fid, "Program_ID", prog_id)

        # Write frequencies as a file attribute since they
        # are the same for all SCSs
        write_attribute(fid, "rfftfreq", f)

        # Create groups for each scs
        for scs in scss
           create_group(fid, scs) 
        end

        # Save data
        for i in 1:length(scss)
            # if i !=1; continue; end # STUB
            write_dataset(fid[scss[i]], "gamma", γ[i])
            write_dataset(fid[scss[i]], "zeta", ζ[i])
            write_dataset(fid[scss[i]], "ps_l", ps_𝓵[i])
            write_dataset(fid[scss[i]], "ps_r", ps_𝓻[i])
            write_dataset(fid[scss[i]], "ps_n", ps_𝓷[i])
        end

    end
    
end




"""
    xitr(op::Int64; flip::Bool=true)

Construct a time-ordered iterator for the specified output's
fast scan axis.

    Parameters: op::Int64
                  Output number selected from 1,2,... 4
                flip::Bool
                  Optionally flip even numbered outputs so
                  that pixels are time ordered
                  from left to right in the result.
    Returns:    xitr
                  An iterator
"""
function xitr(op::Int64; flip::Bool=true)
    
    # Definitions
    wout = 512 # Width of a JWST output in columns
    
    # Find the first and last column of this output
    x0 = (op-1) * wout + 1
    x1 = x0 + wout - 1
    
    # Odd columns increment left to right. Even ones
    # are the reverse
    if isodd(op)
        result = x0:x1
    else
        if flip==true
            result = x1:-1:x0
        else
            result = x0:x1
        end
    end
    result # Return result
    
end




"""
    cumedian(D::CuArray; dims())

Fast medians using the NVIDIA GPU. At the moment, Julia's
median function does not work well when the dims keyword
is set. This fixes that. It runs with full GPU speed.

    Parameters: D::CuArray
                  The input data
                dims::Tuple
                  Dimensions to compute median over
"""
function cumedian(D::CuArray; dims=())
    
    # The usual Statistics median works fine if
    # dims is not specified
    if dims == ()
        return(Statistics.median(D))
    end
    
    # Otherwise, do it the hard way...
    
    # We'll be needing a few things
    _ndims = ndims(D)
    _size = size(D)
    
    # Keep these dimensions
    keep = symdiff(dims, collect(1:ndims(D)))
    
    # Collapse these dimensions
    kill = intersect(dims, collect(1:ndims(D)))
    
    # Permute to put collapsing dimensions first
    D = permutedims(D, vcat(kill, keep))
    
    # How big is it?
    _size1 = size(D)
    n1 = prod(_size1[1:length(dims)])
    n2 = prod(_size1[length(dims)+1:ndims(D)])
    
    # Reshape D for sorting on 1st dimension
    D = reshape(D, (n1,n2)) # Reshape
    
    # Sort on the dimensions to be collapsed
    D = sort(D; dims=1)
    
    # Keep the median value
    if isodd(n1)
        R = D[n1÷2+1,:]
    else
        R = (D[n1÷2,:] .+ D[n1÷2+1,:])/Float32(2)
    end
    
    # Restore final dimensions
    final_dims = ones(Int64, _ndims)
    final_dims[keep] .= _size[keep]
    
    # Restore shape
    R = reshape(R, Tuple(final_dims))

end




"""
    CRFix(nz)

SIRS cosmic ray fixer.

    Parameters: nz::Int
                  Number of up-the-ramp frames in integrations.
"""
mutable struct CRFix
    
    # Declarations
    nx    # Number of image columns
    ny    # Number of image rows
    nz    # Number of up-the-ramp frames
    𝒦     # Fourier transform of finder kernel
    A     # Apodizer function
    w     # Finder kernel width in frames
    j_min # Minimum size for jumpt to be considered a cosmic ray
    
    function CRFix(nz)
        
        # Defaults
        nx = 2048
        ny = 2048
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
        new(nx, ny, nz, 𝒦, A, w, j_min)
        
    end

end



"""
    crfix(F::CRFix, D::Float32; init=true)

SIRS cosmic ray fixing function. This uses convolution up-the-ramp
to find cosmic ray hits. Once found, it subtracts the hit amplitude from all
subsequent samples.

It may be necessary to call this more than once for pixels that have been
hit more than once.

    Parameters: F
                  A SIRS CRFix struct.
                D
                  Data to be scrubbed for cosmic rays.
       Returns: D
                
                  
"""
function crfix(F::CRFix, D::Array{Float32,3})
    
    # GPU RAM is very tight. Only use the GPU when needed.
    w_stripe = 128 # Work in vertical stripes this wide.
    
    # Extend the ends of the datacube to allow apodization. Do this
    # using CPUs to minimize GPU RAM needs
    L = reshape(D[:,:,1], (F.nx,F.ny,1)) .* ones(Float32, (F.nx,F.ny,F.w))
    R = reshape(D[:,:,end], (F.nx,F.ny,1)) .* ones(Float32, (F.nx,F.ny,F.w))
    D = cat(L, D, R, dims=3)
    
    # ***** Work in 128 column stripes to conserve GPU RAM ****** #
    D2 = Array{Float32,3}(undef, (F.nx,F.ny,F.nz)) # The result goes here
    for x1 in collect(1:w_stripe:F.nx)
                
        # Upload stripe to GPU
        x2 = x1 + w_stripe - 1 # Last column, this stripe
        CuD = CuArray(D[x1:x2,:,:]) # Upload stripe to GPU
    
        # Apodize
        CuD .*= F.A
    
        # Convolve with the finder. This gives the jump
        # size for every frame transition and every pixel.
        J = irfft(rfft(CuD, 3) .* F.𝒦, size(CuD,3), 3)

        # Crop to just real frames
        J = J[:,:,F.w+1:end-F.w] # Jumps
        CuD = CuD[:,:,F.w+1:end-F.w] # And data
        
        # Compute minimum allowable jump size for this stripe
        n_samp = 10^4 # Select this many non-repeating samples to speed things up
        CUDA.@allowscalar _J = sample(J, n_samp, replace=false)
        μ = median(_J)
        σ = 1.4826 * median(abs.(_J.-μ))
        F.j_min = μ + 6σ

        # Cosmic rays are the maximum jump in each pixel. Find and fix.
        J_amp = dropdims(maximum(J, dims=3), dims=3)
        for _z in 1:F.nz
            M = (J[:,:,_z] .== J_amp) .& (J_amp .> F.j_min) # Mask of pixels to fix in next frame
            CuD[M,_z+1:F.nz] .= CuD[M,_z+1:F.nz] .- J_amp[M] # Fix
        end
        
        # Download fixed data from GPU
        D2[x1:x2,:,:] = Array(CuD)
        
    end
        
    # Done
    return(D2)
    
end



mutable struct NIRCam_SIRS
    
    # Definitions
    nx        # Number of image columns
    ny        # Number of image rows
    nz        # Number of up-the-ramp frames
    nf        # Number of Fourier frequencies to fit
    rfftfreq  # Frequencies in result
    rb        # Reference pixel border width
    tpix      # Pixel dwell time in seconds
    nout      # Number of outputs
    wout      # Width of each output in columns
    nloh      # Newline overhead in pixels
    nstp      # Number of ticks in clocking pattern
    σ_rej     # Sigma clipping rejection threshold for finding statistical outliers
    B_x_pinvB # Legendre modeling matrix. If B
              # is the basis matrix and pinvB
              # is its Moore-Penrose inverse, then
              # B_x_pinvB computes the linear model.
    𝓚        # Fourier transform of the gap filling kernel
    coadds    # A counter
    𝕃         # A SIRS sum, see p. 9 of Rauscher et al. 2022
    ℝ         # ...
    𝕏         # ...
    𝕐         # ...
    ℤ         # ...
              #
    ℕ         # Save the power spectrum of the normal pixels while we are here
    
    function NIRCam_SIRS(nz)
        
        # Definitions
        nx = 2048
        ny = 2048
        nf = 1025
        rb = 4
        tpix = 10.e-6
        nout = 4
        wout = 512
        nloh = 12
        nstp = (wout + nloh)*ny
        σ_rej = 2.3
        
        # Build the Legendre basis matrix
        r = reshape(CuArray(collect(2/nz-1:2/nz:1)), (:,1)) # Shaped to broadcast
        c = reshape(CuArray([0,1]), (1,:))                  # Shaped to broadcast
        B = legendre.(r,c)                                  # Basis matrix, size(B) = (nz,2)
        B_x_pinvB = Float32.(B * pinv(B))                   # size(B_x_pinvB) = (nz,nz)
        
        #= Build a Gaussian comb kernel. This is used for filling in statistical
        outliers and end of line gaps. =#
        x = collect(1:nstp) # Clock ticks
        σ = 12       # Normal distribution parameter, roughly mimics what IRS^2 samples.
        μ = x[nstp÷2+1] # Normal distribution parameter, mid point of distribution
        K = 2 * (1/σ/sqrt(2π)) * exp.(-0.5*((x .- μ)/σ).^2) # Build the basic kernel shape
        m = ones(Float64, (nstp÷2,2)) # Build a mask to keep only odd indices
        m[:,2] .= 0
        K .*= m'[:] # Apply it. The result is a Gaussian multiplied by a comb.
        𝓚 = CuArray(ComplexF32.(reshape(rfft(ifftshift(K)), (:,1)))) # Transform and shape to broadcast
        
        # Initialize the coadds counter
        coadds = 0
        
        # These are the Fourier frequencies
        rfftfreq = FFTW.rfftfreq(nstp, 1/tpix)[1:nf]
        
        # Initialize SIRS sums
        𝕃 = CuVector{Float32}(undef, nf)
        ℝ = similar(𝕃)
        𝕏 = CuMatrix{ComplexF32}(undef, (nf,nout))
        𝕐 = similar(𝕏)
        ℤ = CuVector{ComplexF32}(undef, nf)
        fill!(𝕃,0); fill!(ℝ,0); fill!(𝕏,0); fill!(𝕐,0); fill!(ℤ,0)
        
        # Also initialize the power spectrum of the normal pixels
        ℕ = CuMatrix{Float32}(undef, (nf,nout))
        fill!(ℕ,0)
        
        # Instantiation
        new(nx, ny, nz, nf, rfftfreq, rb, tpix, nout, wout, nloh, nstp,
            σ_rej, B_x_pinvB, 𝓚, coadds,𝕃, ℝ, 𝕏, 𝕐, ℤ, ℕ)
        
    end
    
end



function nircam_sirs_coadd!(N::NIRCam_SIRS, D::Array{Float32,3})

    # Upload to GPU
    D = CuArray(D)
    
    # Compute line fit residuals
    Δ = D - ein"ij,klj->kli"(N.B_x_pinvB, D)
    D = [] # We don't need D anymore. Free GPU RAM.
    
    # Crudely subtract 1/f by subtracting row medians per output. We will
    # restore this before computing weights.
    Δ = reshape(Δ, (N.wout,N.nout,N.ny,:))
    U = cumedian(Δ, dims=1) # size(U) = (1, 4, 2048, nz) U stands for ugly
    Δ .-= U # Brutally remove 1/f
    Δ = reshape(Δ, (N.nx,N.ny,:)) # Restore shape
    
    # Estimate std as part of preparing to find outliers
    n_samp = 10^4 # Sample this many
    CUDA.@allowscalar _Δ = sample(Δ, n_samp, replace=false) # Nix scalar indexing warmings.
    μ = median(_Δ)
    σ = 1.4826median(abs.(_Δ .- μ))
    
    # Mark bad pixels by setting them equal to NaN. In practice, we know that a few
    # percent are bad. Use a fairly low threshold.
    G = fill!(CuArray{Float16,3}(undef, (N.nx,N.ny,N.nz)), 1) # Good pixels masks
    G[(Δ .<= μ-N.σ_rej*σ) .| (Δ .>= μ+N.σ_rej*σ)] .= NaN    # Mark outliers
    G = G .*
            circshift(G, (-1,0,0)) .*
            circshift(G, (+1,0,0)) .*
            circshift(G, (0,-1,0)) .*
            circshift(G, (0,+1,0))                            # Mark nearest neighbors

    # Brutally restore removed 1/f
    Δ = reshape(Δ, (N.wout,N.nout,N.ny,:))
    Δ .+= U # Restore 1/f
    Δ = reshape(Δ, (N.nx,N.ny,:)) # Restore shape
    
    # It is better for subsequent steps not to have NaNs in. Mark outliers
    # with zeros. This costs us a negligible fraction of good pixels that
    # just happened to be zero.
    Δ[.! isfinite.(G)] .= 0.
    
    # Add newline overheads and reformat into time-ordered vectors.
    nloh = fill!(CuArray{Float32,3}(undef, (N.nloh,N.ny,N.nz)), 0.) # Newline overheads
    _Δ = CuArray{Float32,3}(undef, (N.nstp,N.nout,N.nz))
    for op in 1:N.nout
        _Δ[:,op,:] .= reshape(cat(Δ[xitr(op; flip=true),:,:], nloh, dims=1), (N.nstp,N.nz))
    end
    Δ = _Δ # size(Δ) = (1073152, 4, 150)
    
    # Backfill outliers. This borrows and extends an idea
    # from Rick Arendt in NIRSpec IRS^2. Work frame-by-frame
    # because of GPU RAM limitations.
    for z in 1:N.nz
        _Δ = Δ[:,:,z] # Get frame
        G = _Δ .!= 0 # Compute good pixel mask for this frame
        F = irfft(rfft(_Δ, 1) .* N.𝓚, size(_Δ, 1), 1) ./ irfft(rfft(G, 1) .* N.𝓚, size(_Δ, 1), 1) # Compute fill values
        Δ[.!G,z] .= F[.!G] # Fill gaps
    end

    #= Compute all Fourier tranforms. We only need to keep the first 1025
    frequencies when working with line averages. We cannot do much better
    given the available NIRCam and NIRISS references. This simplifies things
    considerably. =#
    𝓵 = rfft(dropdims(mean(reshape(Δ, (:,N.ny,N.nout,N.nz))[1:N.rb,:,1,:], dims=1), dims=1), 1) # Left ref. is only output #1
    𝓻 = rfft(dropdims(mean(reshape(Δ, (:,N.ny,N.nout,N.nz))[1:N.rb,:,4,:], dims=1), dims=1), 1) # Right ref. is only output #4
    𝓷 = rfft(Δ, 1)[1:N.ny÷2+1,:,:] # Normal pixels
    #
    # The forward transforms multiply by the number of elements in the input vectors. Renormalize
    # so that the data are a true projection into Fourier space.
    𝓵 /= N.ny
    𝓻 /= N.ny
    𝓷 /= N.nstp
    
    # SIRS Sums
    N.𝕃 .+= real.(ein"ij,ij->i"(conj.(𝓵),𝓵))
    N.ℝ .+= real.(ein"ij,ij->i"(conj.(𝓻),𝓻))
    N.𝕏 .+= ein"ij,ikj->ik"(conj.(𝓻), 𝓷)
    N.𝕐 .+= ein"ij,ikj->ik"(conj.(𝓵), 𝓷)
    N.ℤ .+= ein"ij,ij->i"(conj.(𝓵), 𝓻)
    
    # Normal pixels power spectrum sum
    N.ℕ .+= real.(ein"ijk,ijk->ij"(conj.(𝓷),𝓷))
    
    # Increment the counter
    N.coadds = N.coadds + N.nz
            
end



function nircam_sirs_solve(F::CRFix, nz::Int64, files::Vector{String})
    
    # Get some necessary information form the first file
    nints,ngroups,fastaxis,slowaxis = FITS(files[1], "r") do fid
       (read_key(fid[1], "NINTS")[1],
        read_key(fid[1], "NGROUPS")[1],
        read_key(fid[1], "FASTAXIS")[1],
        read_key(fid[1], "SLOWAXIS")[1])
    end
    
    # Instantiate a NIRCam SIRS struct. This initializes the SIRS sums for this SCS.
    N = NIRCam_SIRS(nz)
    
    for file in files
        for i in 1:nints
            
            # Status
            println("processing "*basename(file)*" integration ",i)
            
            # Reading FITS files is sometimes very slow on Prism.
            # Use /lscratch disk to speed it up.
            D = FITS(file, "r") do fid
                read(fid[2],:,:,:,i)
            end
            
            #= Translate to detector coordinates. In detector coordinates,
            the detector is read out in 4 thick vertical stripes. Fast scan is
            from left to right in output #1. The slow scan direction is from
            bottom to top as displayed in SAOImage ds9. =#
            if abs(fastaxis) != 1; D = permutedims(D, (2,1,3)); end # Make fast axis first
            if sign(fastaxis) < 0; D = D[end:-1:1,:,:]; end # Flip if necessary
            if sign(slowaxis) < 0; D = D[:,end:-1:1,:]; end # Flip if necessary
            
            # Fix cosmic rays for entire integration
            D = crfix(F, D)
            
            # Coadd SIRS sums in chunks of nz frames to stay within GPU RAM limits
            for z1 in 1:nz:ngroups
                z2 = z1 + nz -1
                nircam_sirs_coadd!(N, D[:,:,z1:z2]) # coadd this chunk
            end
              
        end
    end
    
    # Solve for frequency dependent weights
    γ = Array((N.ℝ.*N.𝕐 .- N.𝕏.*N.ℤ) ./ (N.ℝ.*N.𝕃 .- N.ℤ .* conj(N.ℤ)))
    ζ = Array((N.𝕏.*N.𝕃 .- N.𝕐.*conj(N.ℤ)) ./ (N.ℝ.*N.𝕃 .- N.ℤ .* conj(N.ℤ)))
    
    # Compute power spectra
    ps_𝓵 = Array(N.𝕃/N.coadds)
    ps_𝓻 = Array(N.ℝ/N.coadds)
    ps_𝓷 = Array(N.ℕ/N.coadds)
    
    # Done
    return(Array(N.rfftfreq), γ, ζ, ps_𝓵, ps_𝓻, ps_𝓷)
    
    # GPU memory is very tight. Explicitly free everything
    # CUDA.unsafe_free!(N)
    
end



"""
    apply_sirs(fits_filename; sirs_file=nothing, zrng=:, irng=:, rolloff=(4,8))

Apply SIRS to JWST NIRCam FITS file. 

    Parameters: fits_filename
                  The input filename. SIRS assumes that you have run the JWST pipeline
                  through the reference correction step. You should have the
                  use_side_reference_pixels kwarg turned OFF. 
                sirs_filesname
                  Set this to use a SIRS calibration file that is different from the default.
                zrng
                  Range of frames to read in and apply SIRS correction to. The default reads
                  all frames.
                irng
                  Range of integrations to read in and apply SIRS correction to. The default
                  reads all integrations.
                rolloff
                  Roll alpha and beta from 1x gain to 0x gain over this range of
                  frequencies. This function uses a raised cosine to do the rolloff.
       Returns: The SIRS corrected data.

    Notes:
    *) This is Julia. The algorithm is as follows.
        1) Read frequency dependent weights, gamma and zeta, in from reference file.
        2) Apply a low pass filter to gamma and zeta. Use a lifted cosine for the roll-off.
           The roll-off currently starts at 4 Hz and finishes at 8 Hz. More work is
           Needed to define the optimal roll-off.
        3) Read the data from the FITS file. SIRS only works on full-frame data. The
           data structure potentially has 4 dimensions,
           (image columns, image rows, resultants, integrations). It simplifies things to 
           always work with 4th rank data and trim singleton dimensions later.
        4a) Get the left reference columns, average them (in rows), find statistical
            outliers, and backfill with local averages.
        4b) Get the right reference columns, average them (in rows), find statistical
            outliers, and backfill with local averages.
        5) Build reference array (for everything in parallel) using Eq. 3 from the SIRS paper.
        6) Subtract reference array
        7) Restore shape
        8) Crop off singleton dimensions.
    *) This is GPU code. The GPU automatically parallelizes almost all of it. If you are working
       in python using CPUs, it should be possible to parallelize the linear algebra operations. 
       To do this, set the appropriate shell environment variable to allow BLAS multithreading.
       On Intel machines, this is usually MKL_NUM_THREADs. On other machines, it is often
       OPM_NUM_THREADS.
"""
function apply_sirs(fits_filename::String; sirs_file::String="", zrng=:, irng=:,
    rolloff::Tuple{<:Number, <:Number}=(0,0))
    
    # Definitions
    nx,ny = 2048,2048 # Size of H2RG detector in pixels
    nout = 4          # Number of H2RG outputs
    rb = 4            # Width of reference pixels border
    sigrej = 4        # Sigma clipping rejection threshold
        
    # Set defaults
    if sirs_file == ""
        sirs_file = "/explore/nobackup/people/brausche/data"*
                        "/JWST/Library/SIRS/20240418_sirs_nircam.h5"
    end
    
    # Use a Gauss weighted average of neighboring good pixels to fill outliers. Build
    # the Gaussian kernel here.
    μ = ny÷2+1 # Mean is center row of detector.
    σ = 1.5 # Use a narrow Gaussian, 1.5 rows. There are not many bad pixels in the reference columns.
    x = CuArray(collect(1:2048)) # x-values for computing Gaussian kernel.
    fK = reshape(rfft(ifftshift(Float32.((1/σ/sqrt(2π)) * exp.(-0.5*((x.-μ)/σ).^2)))), (:,1,1)) # Fourier
                             # transorm of Gaussian kernel shaped to broadcast.
        
    # Get some necessary information from FITS header
    detector, ngroups, nints, fastaxis, slowaxis = FITS(fits_filename, "r") do fid
        (
            lowercase(read_key(fid[1], "DETECTOR")[1]),
            read_key(fid[1], "NGROUPS")[1],
            read_key(fid[1], "NINTS")[1],
            read_key(fid[1], "FASTAXIS")[1],
            read_key(fid[1], "SLOWAXIS")[1]
        )
    end
        
    # Get SIRS frequencies, weights, and filter definition from calibration file
    rfftfreq,γ,ζ,filter = h5open(sirs_file, "r") do fid
        (Float32.(read_attribute(fid, "rfftfreq")),
        CuArray(read(fid[detector])["gamma"]),
        CuArray(read(fid[detector])["zeta"]),
        Float32.(read(fid[detector])["filter"]))
    end
    
    # If set, the rolloff kwarg overrides filter
    if rolloff != (0,0); filter=rolloff; end
    
    #= For NIRCam, the reference columns are very sparsely sampled compared to the normal pixels.
    After only a few Hertz, white read noise becomes more important than 1/f noise. To avoid adding
    a small read noise penalty, roll the SIRS gamma and zeta off starting at a few Hertz using a raised
    cosine function. =#
    f_c = (filter[1]+filter[2])/2 # Hz, 1/2 power frequency
    λ_ap = 2(filter[2]-filter[1]) # Hz, Cosine apodizer wavelength (baseline=8)
    k = 2π/λ_ap
    cosine_rolloff = 0.5*(1 .+ cos.(k*(rfftfreq .- 4)))
    cosine_rolloff[rfftfreq .< f_c-λ_ap/4] .= 1
    cosine_rolloff[rfftfreq .> f_c+λ_ap/4] .= 0
    cosine_rolloff = CuArray(Float32.(reshape(cosine_rolloff, (:,1)))) # Reshape to broadcast
    γ = cosine_rolloff .* γ # Roll-off gamma
    ζ = cosine_rolloff .* ζ # Roll-off zeta
    
    #= We will need to identify outliers in the reference pixels and
    backfill them with something sensible. To find outliers, it will be
    helpful to apply a 1/f suppressing filter before using a sigma-clip
    to identify outliers. Here is that filter. =#
    w = CuArray(reshape(sqrt.(rfftfreq ./ (5 .+ rfftfreq)), (:,1)))
    
    # Get data from FITS file
    D = FITS(fits_filename, "r") do fid
         CuArray(read(fid[2], :, :, zrng, irng))
    end
    
    #= JWST can have up to a 4th rank arrays. Always work to 4th rank
    and drop singleton dimensions later =#
    if ndims(D) != 4
       D = reshape(D, (nx,ny,:,1))
    end
    _nx,_ny,nz,ni = size(D) # Capture dimensions
    
    #= Translate to detector coordinates. =#
    if abs(fastaxis) != 1; D = permutedims(D, (2,1,3,4)); end # Make fast axis first
    if sign(fastaxis) < 0; D = D[end:-1:1,:,:,:]; end # Flip if necessary
    if sign(slowaxis) < 0; D = D[:,end:-1:1,:,:]; end # Flip if necessary

    #= ***** BEGIN - Backfill all outliers in left reference columns ***** =#
    L = irfft(w .* rfft(dropdims(mean(D[1:rb,:,:,:], dims=1), dims=1), 1), ny, 1) # Get ref. cols. and filter 1/f
    μ = median(L)
    σ = 1.4826median(abs.(L.-μ))
    GL = (μ-sigrej*σ.<=L) .& (L.<=μ+sigrej*σ)                                        # Good pixel mask
    L = reshape(dropdims(mean(D[1:rb,:,:,:], dims=1), dims=1), (ny,:,ni))          # Reload unfiltered L
    L[.! GL] .= 0                                                                  # Mark outliers by setting them =0.
    L_fill = irfft(rfft(L, 1) .* fK, 2048, 1) ./ irfft(rfft(GL, 1) .* fK, 2048, 1) # Compute fill values
    L[.! GL] .= L_fill[.! GL]                                                      # Fill outliers
    #= ***** END - Backfill all outliers in left reference columns ***** =#

    #= ***** BEGIN - Backfill all outliers in right reference columns ***** =#
    R = irfft(w .* rfft(dropdims(mean(D[end-rb+1:end,:,:,:], dims=1), dims=1), 1), ny, 1) # Get ref. cols. and filter 1/f
    μ = median(R)
    σ = 1.4826median(abs.(R.-μ))
    GR = (μ-sigrej*σ.<=R) .& (R.<=μ+sigrej*σ)                                        # Good pixel mask
    R = reshape(dropdims(mean(D[end-rb+1:end,:,:,:], dims=1), dims=1), (ny,:,ni))   # Reload unfiltered R
    R[.! GR] .= 0                                                                  # Mark outliers by setting them =0.
    R_fill = irfft(rfft(R, 1) .* fK, 2048, 1) ./ irfft(rfft(GR, 1) .* fK, 2048, 1) # Compute fill values
    R[.! GR] .= R_fill[.! GR]                                                      # Fill outliers
    #= ***** END - Backfill all outliers in left reference columns ***** =#

    # Build the reference array
    R = irfft(ein"ij,ikl->ijkl"(γ, rfft(L,1)) .+ ein"ij,ikl->ijkl"(ζ, rfft(R,1)), ny, 1)
    R = reshape(permutedims(R, (2,1,3,4)), (1,nout,ny,:,ni))
    
    # Reference correct
    D = reshape(D, (nx÷nout,nout,ny,:,ni)) .- R
    
    # Restore shape
    D = reshape(D, (nx,ny,nz,ni))
    
    # Return to pipeline coordinates
    if sign(slowaxis) < 0; D = D[:,end:-1:1,:,:]; end # Unflip if necessary
    if sign(fastaxis) < 0; D = D[end:-1:1,:,:,:]; end # Unflip if necessary
    if abs(fastaxis) != 1; D = permutedims(D, (2,1,3,4)); end # Put fast axis back where it needs to be
    
    # Remove singleton dimensions
    D = dropdims(D, dims = tuple(findall(size(D) .== 1)...))

    # Done
    D
    
end

function apply_sirs(detector::String, D::CuArray{Float32,4}; sirs_file::String="")
    
    # Definitions
    nx,ny = 2048,2048 # Size of H2RG detector in pixels
    nout = 4          # Number of H2RG outputs
    rb = 4            # Width of reference pixels border
    sigrej = 4        # Sigma clipping rejection threshold
        
    # Set defaults
    if sirs_file == ""
        sirs_file = "/explore/nobackup/people/brausche/data"*
                        "/JWST/Library/SIRS/20240418_sirs_nircam.h5"
    end
    
    # Use a Gauss weighted average of neighboring good pixels to fill outliers. Build
    # the Gaussian kernel here.
    μ = ny÷2+1 # Mean is center row of detector.
    σ = 1.5 # Use a narrow Gaussian, 1.5 rows. There are not many bad pixels in the reference columns.
    x = CuArray(collect(1:2048)) # x-values for computing Gaussian kernel.
    fK = reshape(rfft(ifftshift(Float32.((1/σ/sqrt(2π)) * exp.(-0.5*((x.-μ)/σ).^2)))), (:,1,1)) # Fourier
                             # transorm of Gaussian kernel shaped to broadcast.
        
    # Get SIRS frequencies and weights from calibration file
    rfftfreq,γ,ζ,filter = h5open(sirs_file, "r") do fid
        (Float32.(read_attribute(fid, "rfftfreq")),
        CuArray(read(fid[detector])["gamma"]),
        CuArray(read(fid[detector])["zeta"]),
        Float32.(read(fid[detector])["filter"]))
    end
    
    #= For NIRCam, the reference columns are very sparsely sampled compared to the normal pixels.
    After only a few Hertz, white read noise becomes more important than 1/f noise. To avoid adding
    a small read noise penalty, roll the SIRS gamma and zeta off starting at a few Hertz using a raised
    cosine function. =#
    cosine_rolloff = [] # Put results here
    for i in 1:2:16
        f_c = (filter[1]+filter[2])/2 # Hz, 1/2 power frequency
        λ_ap = 2(filter[2]-filter[1]) # Hz, Cosine apodizer wavelength (baseline=8)
        k = 2π/λ_ap
        _cosine_rolloff = 0.5*(1 .+ cos.(k*(rfftfreq .- 4)))
        _cosine_rolloff[rfftfreq .< f_c-λ_ap/4] .= 1
        _cosine_rolloff[rfftfreq .> f_c+λ_ap/4] .= 0
        push!(cosine_rolloff, _cosine_rolloff)
    end
    cosine_rolloff = reshape(CuArray(Float32.(hcat(cosine_rolloff...))), (:,2,4))
    γ = cosine_rolloff[:,1,:] .* γ # Roll-off gamma
    ζ = cosine_rolloff[:,2,:] .* ζ # Roll-off zeta
    
    #= We will need to identify outliers in the reference pixels and
    backfill them with something sensible. To find outliers, it will be
    helpful to apply a 1/f suppressing filter before using a sigma-clip
    to identify outliers. Here is that filter. =#
    w = CuArray(reshape(sqrt.(rfftfreq ./ (5 .+ rfftfreq)), (:,1)))
    
    #= JWST can have up to a 4th rank arrays. Always work to 4th rank
    and drop singleton dimensions later =#
    if ndims(D) != 4
       D = reshape(D, (nx,ny,:,1))
    end
    _nx,_ny,nz,ni = size(D) # Capture dimensions

    #= ***** BEGIN - Backfill all outliers in left reference columns ***** =#
    L = irfft(w .* rfft(dropdims(mean(D[1:rb,:,:,:], dims=1), dims=1), 1), ny, 1) # Get ref. cols. and filter 1/f
    μ = median(L)
    σ = 1.4826median(abs.(L.-μ))
    GL = (μ-sigrej*σ.<=L) .& (L.<=μ+sigrej*σ)                                        # Good pixel mask
    L = reshape(dropdims(mean(D[1:rb,:,:,:], dims=1), dims=1), (ny,:,ni))          # Reload unfiltered L
    L[.! GL] .= 0                                                                  # Mark outliers by setting them =0.
    L_fill = irfft(rfft(L, 1) .* fK, 2048, 1) ./ irfft(rfft(GL, 1) .* fK, 2048, 1) # Compute fill values
    L[.! GL] .= L_fill[.! GL]                                                      # Fill outliers
    #= ***** END - Backfill all outliers in left reference columns ***** =#

    #= ***** BEGIN - Backfill all outliers in right reference columns ***** =#
    R = irfft(w .* rfft(dropdims(mean(D[end-rb+1:end,:,:,:], dims=1), dims=1), 1), ny, 1) # Get ref. cols. and filter 1/f
    μ = median(R)
    σ = 1.4826median(abs.(R.-μ))
    GR = (μ-sigrej*σ.<=R) .& (R.<=μ+sigrej*σ)                                        # Good pixel mask
    R = reshape(dropdims(mean(D[end-rb+1:end,:,:,:], dims=1), dims=1), (ny,:,ni))   # Reload unfiltered R
    R[.! GR] .= 0                                                                  # Mark outliers by setting them =0.
    R_fill = irfft(rfft(R, 1) .* fK, 2048, 1) ./ irfft(rfft(GR, 1) .* fK, 2048, 1) # Compute fill values
    R[.! GR] .= R_fill[.! GR]                                                      # Fill outliers
    #= ***** END - Backfill all outliers in left reference columns ***** =#

    # Build the reference array
    R = irfft(ein"ij,ikl->ijkl"(γ, rfft(L,1)) .+ ein"ij,ikl->ijkl"(ζ, rfft(R,1)), ny, 1)
    R = reshape(permutedims(R, (2,1,3,4)), (1,nout,ny,:,ni))
    
    # Reference correct
    D = reshape(D, (nx÷nout,nout,ny,:,ni)) .- R
    
    # Restore shape
    D = reshape(D, (nx,ny,nz,ni))
    
    # Remove singleton dimensions
    D = dropdims(D, dims = tuple(findall(size(D) .== 1)...))

    # Done
    D
    
end
