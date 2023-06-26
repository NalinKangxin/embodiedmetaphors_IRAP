using CSV
using DataFrames
using StatsPlots
using StatsBase
using FreqTables
using GLM

#specify current path
# cd("C:\\Users\\popo-eveny\\.julia\\byNalin\\notebooks\\classFiles\\UsedData")

#downloading .csv files
for file in ["230624_lkx_nalin_IRAP_2023-06-24_16h13.25.381.csv","230624_lax_nalin_IRAP_2023-06-24_16h22.26.389.csv", "230624_lqy_nalin_IRAP_2023-06-24_16h51.43.963.csv"]
    URL = string("https://raw.githubusercontent.com/NalinKangxin/embodiedmetaphors_IRAP/main/data/", file)
    path = joinpath(pwd(), file)
    download(URL, path)
end

#pre-processing .csv files
CSV_list = filter(endswith(".csv"), readdir())

df_IS = mapreduce(vcat, CSV_list) do file   #Inter-Subject
    dt = CSV.read(file, DataFrame)
    select!(dt, ["textStim", "key_resp.corr", "key_resp.rt", "participant", "age"])     #!!!!!! ["stimtype", "condition", "tasktype"]
    rename!(dt, "key_resp.rt" => "rt")
    rename!(dt, "key_resp.corr" => "corr") 
    rename!(dt, "textStim" => "stim")
    dropmissing!(dt, :rt)     #drop learning-phrase
    subset!(dt, :corr => ByRow(corr -> corr == 1) )     #drop wrong-response
    transform!(dt, :participant => ByRow(p -> last(split(p, "_"))) => :participant)  #rename participants'names
    # transform!(dt, :age => ByRow(age -> string(age)) => :age)
    return dt
end
typeof(df_IS.age)
describe(df_IS)

#cleaning data
#outlier analysis - box plot /IQR
StatsPlots.boxplot(df_IS.participant, df_IS.rt, title = "Box Plot - Response Time", ylabel = "Response Time(s)", legend = false)

Q1 = StatsBase.percentile(df_IS.rt, 25)
Q3 = StatsBase.percentile(df_IS.rt, 75)
# IQR = Q3 - Q1
iqr = StatsBase.iqr(df_IS.rt)
outlier_step = 1.5 * iqr
df_IS1 = subset(df_IS, :rt => ByRow(rt -> (rt > Q1 - outlier_step) & (rt < Q3 + outlier_step)))     #drop outliers
select!(df_IS1, Not("corr"))
# isequal(df_IS, df_IS1)  =>false
describe(df_IS1)

#analyzing data 
#frequenct table 
freqtable(df_IS1, :stim, :participant)
#!!!!!! freqtable(df_IS1, :tasktype, :participant)
#!!!!!! freqtable(df_IS1, :condition, :participant)
#!!!!!! tasktype = ["compitable", "incompitable"]
#!!!!!! condition = ["pro_metaphor", "anti_metaphor"]

#get mean and sd
df_MoA = combine(groupby(df_IS1, :age), :rt => (rt -> (M =mean(rt), SD = std(rt))) => AsTable) 
sort!(df_MoA, :M)      #sort from min to max
df_MoS = combine(groupby(df_IS1, :stim), :rt => (rt -> (M =mean(rt), SD = std(rt))) => AsTable)
sort!(df_MoS, :M)
#!!!!!! df_MoS = combine(groupby(df_IS1, :stimtype), :rt => (rt -> (M =mean(rt), SD = std(rt))) => AsTable)
#!!!!!! sort!(df_MoS, :M)
#!!!!!! df_MoT = combine(groupby(df_IS1, :tasktype), :rt => (rt -> (M =mean(rt), SD = std(rt))) => AsTable)
#!!!!!! sort!(df_MoT, :M)
#!!!!!! df_MoC = combine(groupby(df_IS1, :condition), :rt => (rt -> (M =mean(rt), SD = std(rt))) => AsTable)
#!!!!!! sort!(df_MoC, :M)

unique(df_MoS.stim)
negative_stim = subset(df_MoS, :stim => ByRow(stim -> (stim=="伤心") | (stim=="失落") | (stim=="悲伤") | (stim=="忧郁") | (stim=="愁苦") | (stim=="难过")))
positive_stim = subset(df_MoS, :stim => ByRow(stim -> (stim=="高兴") | (stim=="愉快") | (stim=="快乐") | (stim=="欣喜") | (stim=="喜悦") | (stim=="愉悦")))
MoN = mean(negative_stim.M)
MoP = mean(positive_stim.M)

#density plot

#scatter plot
StatsBase.cor(df_IS1.age, df_IS1.rt)
# => -0.10352771887197859
scatter(df_IS1.age, df_IS1.rt, title = "Scatter Plot Response Time vs Age", ylabel = "Response Time", xlabel = "Age",legend = false)
#!!!!!! scatter(df_IS1.score, df_IS1.rt, title = "Scatter Plot Response Time vs Age", ylabel = "Response Time", xlabel = "Age",legend = false)  #language score

#creating model, Ref: https://blog.csdn.net/weixin_41715077/article/details/107061800
fm = @formula(rt ~ age)
linearRegressor = lm(fm, df_IS1)
# julia> show(linearRegressor)
# StatsModels.TableRegressionModel{LinearModel{GLM.LmResp{Vector{Float64}}, GLM.DensePredChol{Float64, LinearAlgebra.CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}}}, Matrix{Float64}}
# rt ~ 1 + age
# Coefficients:
# ─────────────────────────────────────────────────────────────────────────────
#                    Coef.  Std. Error      t  Pr(>|t|)  Lower 95%    Upper 95%
# ─────────────────────────────────────────────────────────────────────────────
# (Intercept)   1.84165     0.075043    24.54    <1e-93   1.69429    1.98902
# age          -0.00661589  0.00252433  -2.62    0.0090  -0.011573  -0.00165882
# ─────────────────────────────────────────────────────────────────────────────
r2(linearRegressor)
# => 0.010717988574835746

#README
#dependent factor: 
#response time
#independent factors:
#age 
#tasktype: (L1)compitable (L2)incompitable
#condition: (L1)pro_metaphor (L2)anti_metaphor
#?stimtype: (L1)positive_emotional_adj (L2)negative_emotional_adj
#!!!!!! must modify the psychopy program to collect the data in need