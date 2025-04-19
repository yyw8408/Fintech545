using BenchmarkTools
using Distributions
using Random
using StatsBase
using DataFrames
using Plots
using StatsPlots
using LinearAlgebra
using JuMP
using Ipopt
using Dates
using ForwardDiff
using FiniteDiff
using CSV
using LoopVectorization
using Roots
using QuadGK
using StateSpaceModels
using Printf
using PyCall


include("../../library/return_calculate.jl")

#Read in the Data
portfolio = CSV.read("Final Project/initial_portfolio.csv", DataFrame)
rf = CSV.read("Final Project/rf.csv", DataFrame)
prices = CSV.read("Final Project/DailyPrices.csv", DataFrame)

#calculate the returns
allReturns = return_calculate(prices,method="DISCRETE", dateColumn="Date")

#add the risk free rate to the returns
allReturns = leftjoin(allReturns, rf, on=:Date)

#Get the stock symbols
stocks = portfolio.Symbol
nStocks = length(stocks)

#Fit OLS
function OLS(X,Y)
    n = size(X,1)
    X = hcat(ones(n),X)
    b = (X'X)\(X'Y)
    return b
end

# Create view of allReturns for the date range before 2024-01-01
toFit = @view allReturns[allReturns.Date .< Date(2023, 12, 31),:]

# Initialize arrays to store betas and alphas for each stock
Betas = zeros(nStocks)
Alphas = zeros(nStocks)
# Initialize a matrix to store the idiosyncratic returns for each stock
ido = Array{Float64,2}(undef, size(toFit[!,stocks]))

#Fit the CAPM Model
i=1
for s in stocks
    fit = OLS(toFit.SPY - toFit.rf, toFit[:,s]-toFit.rf)
    Alphas[i] = fit[1]
    Betas[i] = fit[2]
    ido[:,i] = toFit[:,s] .- toFit.rf .- Alphas[i] .- Betas[i]*(toFit.SPY .- toFit.rf)
    i += 1
end

#idiosyncratic covariance matrix
covar = cov(ido)

#start values for the weights as the inverse of the standard deviation of the idiosyncratic returns
start = 1 ./ sqrt.(diag(covar))
start = start ./ sum(start)

# Optimize the Portfolio
printLevel = 5
n = size(covar,1)
m = n

# Set up the JuMP model
model = Model(Ipopt.Optimizer)
set_optimizer_attribute(model, "print_level", printLevel)

# Risk budget is 1 for each stock
riskBudget = fill(1.0,n)
mult = riskBudget.^(-1)

# Set up the variables
# Weights with boundary at 0, starting values are from above
@variable(model, w[i=1:n] >= 0, start=start[i])

# Function for the portfolio volatility
function pvol(w...)
    x = collect(w)
    # println(x)
    return(sqrt(x'*covar*x))
end

# Function for Component Standard Deviation
function pCSD(w...)
    x = collect(w)
    pVol = pvol(w...)
    csd = x.*(covar*x)./pVol
    return (csd)
end

# Sum Square Error of Component Standard Deviation -- This is our objective function
function sseCSD(w...)
    csd = mult.*pCSD(w...)
    mCSD = sum(csd)/n
    dCsd = csd .- mCSD
    se = dCsd .*dCsd
    return(1.0e1*sum(se))
end

# Register the function with the model
register(model,:distCSD,n,sseCSD,autodiff=true)

# Set the objective function to minimize the SSE of the component standard deviation
@NLobjective(model,Min,distCSD(w...))

# Set the constraints for the weights
@constraint(model, (sum(w[i] for i in 1:n))==1)
# set the constrants for the total portfolio Beta
@constraint(model, (sum(w[i]*Betas[i] for i in 1:n))==1)

#optimize the model
optimize!(model)
# make sure the weights sum to 1, this removes small numerical errors
x = value.(w)/sum(value.(w))

status = raw_status(model)

# Print some statistics about the optimal solution
if printLevel > 0
    println("Solve Status - ", status)
    pVol = pvol(x...)
    println("Found Vol: ", pVol)
    csd = pCSD(x...)
    pct_RB = riskBudget ./ sum(riskBudget)
    pct_csd = csd ./ sum(csd)
    mDiff = max( abs.(pct_RB - pct_csd)...)
    println("Max Abs Pct Component Risk Diff From Tgt ", mDiff)
    println("SSE Component Risk: ", sseCSD(x...))

end

# Write results to CSV
CSV.write("Final Project/optimized_portfolio_extra_credit.csv", 
    DataFrame(:Symbol=>stocks, :Weight => x, 
    :Beta => Betas, :CSD=>pCSD(x...)))

Covar = DataFrame(covar, stocks)
CSV.write("Final Project/covariance_matrix.csv", Covar)