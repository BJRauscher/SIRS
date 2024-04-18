#= The following snippet was cut 4/5/2024 from cosmic_rays.jl. =#

# ***** This section should be revisited some day. Julia's CUDA package *****
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
# ***** END CPU CODE *****
    
