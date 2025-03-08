
function VaR(a; alpha=0.05)
    x = sort(a)
    nup = convert(Int64,ceil(size(a,1)*alpha))
    ndn = convert(Int64,floor(size(a,1)*alpha))
    v = 0.5*(x[nup]+x[ndn])

    return -v
end

function VaR(d::UnivariateDistribution; alpha=0.05)
    -quantile(d,alpha)
end

function ES(a; alpha=0.05)
    x = sort(a)
    nup = convert(Int64,ceil(size(a,1)*alpha))
    ndn = convert(Int64,floor(size(a,1)*alpha))
    v = 0.5*(x[nup]+x[ndn])
    
    es = mean(x[x.<=v])
    return -es
end

function ES(d::UnivariateDistribution; alpha=0.05)
    v = VaR(d;alpha=alpha)
    f(x) = x*pdf(d,x)
    st = quantile(d,1e-12)
    return -quadgk(f,st,-v)[1]/alpha
end
