
function near_psd(a; epsilon=0.0)
    n = size(a,1)

    invSD = nothing
    out = copy(a)

    #calculate the correlation matrix if we got a covariance
    if count(x->x ≈ 1.0,diag(out)) != n
        invSD = diagm(1 ./ sqrt.(diag(out)))
        out = invSD * out * invSD
    end

    #SVD, update the eigen value and scale
    vals, vecs = eigen(out)
    vals = max.(vals,epsilon)
    T = 1 ./ (vecs .* vecs * vals)
    T = diagm(sqrt.(T))
    l = diagm(sqrt.(vals))
    B = T*vecs*l
    out = B*B'

    #Add back the variance
    if invSD !== nothing 
        invSD = diagm(1 ./ diag(invSD))
        out = invSD * out * invSD
    end
    return out
end
#Cholesky that assumes PSD
function chol_psd!(root,a)
    n = size(a,1)
    #Initialize the root matrix with 0 values
    root .= 0.0

    #loop over columns
    for j in 1:n
        s = 0.0
        #if we are not on the first column, calculate the dot product of the preceeding row values.
        if j>1
            s =  root[j,1:(j-1)]'* root[j,1:(j-1)]
        end
  
        #Diagonal Element
        temp = a[j,j] .- s
        if 0 >= temp >= -1e-8
            temp = 0.0
        end
        root[j,j] =  sqrt(temp);

        #Check for the 0 eigan value.  Just set the column to 0 if we have one
        if 0.0 == root[j,j]
            root[j,(j+1):n] .= 0.0
        else
            #update off diagonal rows of the column
            ir = 1.0/root[j,j]
            for i in (j+1):n
                s = root[i,1:(j-1)]' * root[j,1:(j-1)]
                root[i,j] = (a[i,j] - s) * ir 
            end
        end
    end
end

#Helper functions from Notes
function _getAplus(A)
    vals, vecs =eigen(A)
    vals = diagm(max.(vals,0))
    return vecs * vals*vecs'
end

function _getPS(A,W)
    W05 = sqrt.(W)
    iW = inv(W05)
    return (iW * _getAplus(W05*A*W05) * iW)
end

function _getPu(A,W)
    Aret = copy(A)
    for i in 1:size(Aret,1)
        Aret[i,i] = 1.0
    end
    return Aret
end

function wgtNorm(A,W)
    W05 = sqrt.(W)
    W05 = W05 * A * W05
    return sum(W05 .* W05)
end

function higham_nearestPSD(pc,W=nothing, epsilon=1e-9,maxIter=100,tol=1e-9)

    n = size(pc,1)
    if W === nothing
        W = diagm(fill(1.0,n))
    end

    deltaS = 0

    invSD = nothing
    
    Yk = copy(pc)

    #calculate the correlation matrix if we got a covariance
    if count(x->x ≈ 1.0,diag(Yk)) != n
        invSD = diagm(1 ./ sqrt.(diag(Yk)))
        Yk = invSD * Yk * invSD
    end

    Yo = copy(Yk)
    
    norml = typemax(Float64)
    i=1

    while i <= maxIter
        # println("$i - $norml")
        Rk = Yk .- deltaS
        #Ps Update
        Xk = _getPS(Rk,W)
        deltaS = Xk - Rk
        #Pu Update
        Yk = _getPu(Xk,W)
        #Get Norm
        norm = wgtNorm(Yk-Yo,W)
        #Smallest Eigenvalue
        minEigVal = min(real.(eigvals(Yk))...)

        # print("Yk: "); display(Yk)
        # print("Xk: "); display(Xk)
        # print("deltaS: "); display(deltaS)

        if norm - norml < tol && minEigVal > -epsilon
            # Norm converged and matrix is at least PSD
            break
        end
        # println("$norml -> $norm")
        norml = norm
        i += 1
    end
    if i < maxIter 
        println("Converged in $i iterations.")
    else
        println("Convergence failed after $(i-1) iterations")
    end

    #Add back the variance
    if invSD !== nothing 
        invSD = diagm(1 ./ diag(invSD))
        Yk = invSD * Yk * invSD
    end
    return Yk
end


#Normal Simulation Function:
function simulateNormal(N::Int64, cov::Array{Float64,2}; mean=[],seed=1234, fixMethod=near_psd)

    #Error Checking
    n, m = size(cov)
    if n != m
        throw(error("Covariance Matrix is not square ($n,$m)"))
    end


    out = Array{Float64,2}(undef,(n,N))

    #If the mean is missing then set to 0, otherwise use provided mean
    _mean = fill(0.0,n)
    m = size(mean,1)
    if !isempty(mean)
        if n!=m
            throw(error("Mean ($m) is not the size of cov ($n,$n"))
        end
        copy!(_mean,mean)
    end


    # Take the root
    l = Array{Float64,2}(undef,n,n)

    try
        l = Matrix(cholesky(cov).L)
    catch e
        if isa(e, LinearAlgebra.PosDefException)
            # println("Matrix is not PD, assuming PSD and continuing.")
            try
                chol_psd!(l,cov)
            catch e2
                chol_psd!(l,fixMethod(cov))
            end
        else
            throw(e)
        end
    end
    

    #Generate needed random standard normals
    Random.seed!(seed)
    d = Normal(0.0,1.0)

    rand!(d,out)

    #apply the standard normals to the cholesky root
    out = (l*out)'

    #Loop over itereations and add the mean
    for i in 1:n
        out[:,i] = out[:,i] .+ _mean[i]
    end
    out
end

#PCA
function simulate_pca(a, nsim; pctExp=1, mean=[],seed=1234)
    n = size(a,1)

    #If the mean is missing then set to 0, otherwise use provided mean
    _mean = fill(0.0,n)
    m = size(mean,1)
    if !isempty(mean)
        copy!(_mean,mean)
    end

    #Eigenvalue decomposition
    vals, vecs = eigen(a)
    vals = real.(vals)
    vecs = real.(vecs)
    #julia returns values lowest to highest, flip them and the vectors
    flip = [i for i in size(vals,1):-1:1]
    vals = vals[flip]
    vecs = vecs[:,flip]
    
    tv = sum(vals)

    posv = findall(x->x>=1e-8,vals)
    if pctExp < 1
        nval = 0
        pct = 0.0
        #figure out how many factors we need for the requested percent explained
        for i in 1:size(posv,1)
            pct += vals[i]/tv
            nval += 1
            if pct >= pctExp 
                break
            end
        end
        if nval < size(posv,1)
            posv = posv[1:nval]
        end
    end
    vals = vals[posv]

    vecs = vecs[:,posv]

    # println("Simulating with $(size(posv,1)) PC Factors: $(sum(vals)/tv*100)% total variance explained")
    B = vecs*diagm(sqrt.(vals))

    Random.seed!(seed)
    m = size(vals,1)
    r = randn(m,nsim)

    out = (B*r)'
    #Loop over itereations and add the mean
    for i in 1:n
        out[:,i] = out[:,i] .+ _mean[i]
    end
    return out
end