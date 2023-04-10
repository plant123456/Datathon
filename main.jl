using Printf, PyPlot
using LinearAlgebra, Random, Statistics
using DataFrames, CSV, HDF5
using Distances, Zygote, QuadGK, GLM
using DelimitedFiles
#using SymPy

path = "D:\\Citedal European Datathon\\Crashes"#"D:\\Citedal European Datathon\\Traffic, Investigations _ Other"
csv = CSV.read("$path\\crash_info_general.csv",DataFrame)




x = csv[:,2]
y = csv[:,1]
data = DataFrame(x=x, y=y)
model = lm(@formula(y ~ x), data)
coef(model)




# make predictions
new_x = [6, 7, 8]
predicted_y = predict(model, DataFrame(x=new_x))
println("Predicted y values: ", predicted_y)


#### WorkSpace



# Total number of car crash per year
year = zeros(12) # how many car crash per year?
for i in eachindex(year)
    year[i] = sum(csv[:,6] .== (2010+i-1))
end
fig = figure("plot",figsize=(8,5), dpi=100)
bar(2010:1:2021,year,color="#0f87bf",width=0.6, align="center",alpha=0.5) # another color is 073763
ax = gca()
labelfont = Dict("family"=>"serif", "color"=>"black", "weight"=>"normal", "size"=>12)
ax.set_xlabel("Year", fontdict=labelfont)
ax.set_ylabel("Total amount of car crash per year", fontdict=labelfont)
ax.set_xticks(2010:1:2021,labels=2010:1:2021, fontsize=10)
#ax.set_xlim([2009,2022])
ax.set_ylim([0,14000])
savefig("D:\\Citedal European Datathon\\Crash_Year.svg")
close()

# Month of the Week when crash occurred
month = zeros(12,12) # plot line is along column (along years), for one column, different lines mean different years.
for Y = 1:12
    for M = 1:12
        month[M,Y] = sum(csv[csv[:,6] .== (2009+Y),7] .== M)
end;end
ioff()
plot(month) # 横轴是年份，不同的线代表不同月（环比）
#plot(month') # 横轴是月份，不同线代表不同年（同比）
ax = gca()
ax.set_xticks(0:1:11,labels=1:1:12,fontsize=9)
#ax.set_yticks(0:1:11,labels=1:1:12,fontsize=9)
ax.set_xlabel("Month",fontsize=11)
ax.set_ylabel("Car Crush",fontsize=11)

# Day of the Week when crash occurred
day = zeros(7)
for i=1:7
    day[i] = sum(csv[:,8] .== i)
end
fig = figure("plot",figsize=(8,5), dpi=100)
bar(["Monday","Tuseday","Wednesday","Thursday","Friday","Saturday","Sunday"],day,color="#0f87bf",width=0.6, align="center",alpha=0.5) # another color is 073763
ax = gca()
labelfont = Dict("family"=>"serif", "color"=>"black", "weight"=>"normal", "size"=>12)
ax.set_xlabel("Day of a week", fontdict=labelfont)
ax.set_ylabel("Total amount of car crash per day", fontdict=labelfont)
#ax.set_xticks(2010:1:2021,labels=2010:1:2021, fontsize=10)
#ax.set_xlim([2009,2022])
ax.set_ylim([17000,22000])
savefig("D:\\Citedal European Datathon\\Crash_Day.svg")
close()





# Car accident fatalities

fata_month = zeros(12,12) # different rows mean months, differnet columns mean years
for i=1:133013
    Y = csv[i,6]-2009
    M = csv[i,7]
    fata_month[M,Y] = fata_month[M,Y] + csv[i,42]
end
fata_year = zeros(12)
for i=1:12
fata_year[i] = sum(fata_month[:,i])
end
fig = figure("plot",figsize=(8,5), dpi=100)
bar(2010:1:2021,fata_year,color="#0f87bf",width=0.6, align="center",alpha=0.5) # another color is 073763
ax = gca()
labelfont = Dict("family"=>"serif", "color"=>"black", "weight"=>"normal", "size"=>12)
ax.set_xlabel("Year", fontdict=labelfont)
ax.set_ylabel("Total amount of fatalities involved", fontdict=labelfont)
ax.set_xticks(2010:1:2021,labels=2010:1:2021, fontsize=10)
savefig("D:\\Citedal European Datathon\\Fatality_Year.svg")
close()

fata_day = zeros(7)
for i=1:7
    fata_day[i] = sum(csv[csv[:,8] .== i,42])
end
fig = figure("plot",figsize=(8,5), dpi=100)
bar(["Monday","Tuseday","Wednesday","Thursday","Friday","Saturday","Sunday"],fata_day,color="#0f87bf",width=0.6, align="center",alpha=0.5) # another color is 073763
ax = gca()
labelfont = Dict("family"=>"serif", "color"=>"black", "weight"=>"normal", "size"=>12)
ax.set_xlabel("Day of a week", fontdict=labelfont)
ax.set_ylabel("Total amount of fatalities per day", fontdict=labelfont)
#ax.set_xticks(2010:1:2021,labels=2010:1:2021, fontsize=10)
#ax.set_xlim([2009,2022])
ax.set_ylim([0,280])
savefig("D:\\Citedal European Datathon\\Fatality_Day.svg")
close()











# traffic control
sum(csv[:,64] .== "0")
sum(csv[:,64] .== "3")
control = zeros(12,6)
for M in ["0","1","2","3","4","9"]

for Y=1:12
    control[Y,6] = sum(csv[csv[:,6] .== (2009+Y), 64] .== "9")
end
file = h5open("test.hdf5", "w")
file["control"] = control
close(file)


####
Predict the car crash fatality according to objective conditions of municipal roads

using Printf, PyPlot
using LinearAlgebra, Random, Statistics
using DataFrames, CSV, HDF5
using Distances, Zygote, QuadGK, GLM
using DelimitedFiles
using Flux, CUDA
using Flux: onehotbatch, onehot, onecold, flatten, crossentropy, throttle, params
using Flux.Losses: logitcrossentropy
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser, WeightDecay
using Logging: with_logger
using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!
using ProgressMeter: @showprogress
import BSON

month = csv[:,7]; #1
day = csv[:,8]; #2
time = csv[:,9]; #3 remove 99 & NA
latitude = csv[:,11]; #4 remove & NA
longitude = csv[:,12]; #5 remove & NA
weather = csv[:,90]; #6
y16 = csv[:,46]; #7
y17 = csv[:,47]; #8
y18 = csv[:,48]; #9
y19 = csv[:,49]; #10
y20 = csv[:,50]; #11
y50_64 = csv[:,51]; #12
y65_74 = csv[:,52]; #13
y75_ = csv[:,53] #14
illumination = csv[:,55]; #15
intersection = csv[:,56]; #16
road_relation = csv[:,63]; #17
road_surface = csv[:,64]; #18
traffic_control_state = csv[:,65]; #19
traffic_control_type = csv[:,66]; #20


fatality = csv[:,42]; #1
injury = csv[:,43]; #2


input_x = cat(month, day, time, latitude, longitude, weather, y16, y17, y18, y19, y20, y50_64, y65_74, y75_, illumination, intersection, road_relation, road_surface, traffic_control_state, traffic_control_type,dims=2)
output_y = cat(fatality,injury, dims=2);
# wash
newx = input_x[((input_x[:,3] .!= "NA") .& (input_x[:,3] .!= "99") .& (input_x[:,4] .!= "NA") .& (input_x[:,5] .!= "NA")),:]
newy = output_y[((input_x[:,3] .!= "NA") .& (input_x[:,3] .!= "99") .& (input_x[:,4] .!= "NA") .& (input_x[:,5] .!= "NA")),:]
for i = 3:5
    newx[:,i] = parse.(Float64,newx[:,i])
end
input_x = copy(newx')
output_y = copy(newy')

testpick = sample(1:127844,30000,replace=false);

x_train = 1:20; y_train = 1:2;
x_test = 1:20; y_test = 1:2;
for i = 1:127844
    if i in testpick
        x_test = hcat(x_test, newx[i,:])
        y_test = hcat(y_test, newy[i,:])
    else
        x_train = hcat(x_train, newx[i,:])
        y_train = hcat(y_train, newy[i,:])
    end
end

x_train = convert(Array{Float64}, x_train)
file = h5open("D:\\Citedal European Datathon\\training_data_x.hdf5", "w")
file["x_train"] = x_train[:,2:end]
close(file)
y_train = convert(Array{Float64}, y_train)
file = h5open("D:\\Citedal European Datathon\\training_data_y.hdf5", "w")
file["y_train"] = y_train[:,2:end]
close(file)
x_test = convert(Array{Float64}, x_test)
file = h5open("D:\\Citedal European Datathon\\testing_data_x.hdf5", "w")
file["x_test"] = x_test[:,2:end]
close(file)
y_test = convert(Array{Float64}, y_test)
file = h5open("D:\\Citedal European Datathon\\testing_data_y.hdf5", "w")
file["y_test"] = y_test[:,2:end]
close(file)
####

path="D:\\Citedal European Datathon\\training_data_x.hdf5"
file = h5open(path, "r")
x_train = read(file["//x_train"])
close(file)
path="D:\\Citedal European Datathon\\training_data_y.hdf5"
file = h5open(path, "r")
y_train = read(file["//y_train"])
close(file)
path="D:\\Citedal European Datathon\\testing_data_x.hdf5"
file = h5open(path, "r")
x_test = read(file["//x_test"])
close(file)
path="D:\\Citedal European Datathon\\testing_data_y.hdf5"
file = h5open(path, "r")
y_test = read(file["//y_test"])
close(file)







# Define the architecture of the neural network
DNN = Chain(
  Dense(20, 128, relu),
  Dense(128, 256, relu),
  Dense(256, 256, relu),
  Dense(256, 128, relu),
  Dense(128, 2)
)

Base.@kwdef mutable struct Args
    η = 3e-4             ## learning rate
    λ = 0                ## L2 regularizer param, implemented as weight decay
    batchsize = 1000      ## batch size
    epochs = 100         ## number of epochs
    seed = 0             ## set seed > 0 for reproducibility
    use_cuda = true      ## if true use cuda (if available)
    infotime = 1 	     ## report every `infotime` epochs
    checktime = 50        ## Save the model every `checktime` epochs. Set to 0 for no checkpoints.
    tblogger = true      ## log training with tensorboard
    savepath = "D:\\Citedal European Datathon\\model\\"   ## results path
end

function DNN()
    return Chain(
      Dense(20, 128, relu),
      Dense(128, 256, relu),
      Dense(256, 256, relu),
      Dense(256, 128, relu),
      Dense(128, 2)
    )
end

# Define the loss function, number of all params, and optimization algorithm
loss(ŷ, y) = logitcrossentropy(ŷ, y)
#loss(x, y) = Flux.mse(DNN(x), y)
num_params(model) = sum(length, Flux.params(model))
round4(x) = round(x, digits=4)
lossplot=[]; accplot=[]

function eval_loss_accuracy(loader, model, device)
    l = 0f0
    acc = 0
    ntot = 0
    for (x, y) in loader
        x, y = x |> device, y |> device
        ŷ = model(x)
        l += loss(ŷ, y) * size(x)[end]
        acc += sum(onecold(ŷ |> cpu) .== onecold(y |> cpu))
        ntot += size(x)[end]
    end
    return (loss = l/ntot |> round4, acc = acc/ntot*100 |> round4)
end

function train(; kws...)
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)
    use_cuda = args.use_cuda && CUDA.functional()
    if use_cuda
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    path="D:\\Citedal European Datathon\\training_data_x.hdf5"
    file = h5open(path, "r")
    x_train = read(file["//x_train"])
    close(file)
    path="D:\\Citedal European Datathon\\training_data_y.hdf5"
    file = h5open(path, "r")
    y_train = read(file["//y_train"])
    close(file)
    path="D:\\Citedal European Datathon\\testing_data_x.hdf5"
    file = h5open(path, "r")
    x_test = read(file["//x_test"])
    close(file)
    path="D:\\Citedal European Datathon\\testing_data_y.hdf5"
    file = h5open(path, "r")
    y_test = read(file["//y_test"])
    close(file)
    train_loader = DataLoader((x_train, y_train), batchsize=args.batchsize, shuffle=true) #all 51*21=1071 images
    test_loader = DataLoader((x_test, y_test),  batchsize=args.batchsize)
    @info "Dataset $(train_loader.nobs) train and $(test_loader.nobs) test examples"

    model = DNN() |> device
    @info "DNN model: $(num_params(model)) trainable params"

    ps = Flux.params(model)
    optimizer = ADAM(args.η)

    if args.tblogger
        tblogger = TBLogger(args.savepath, tb_overwrite)
        set_step_increment!(tblogger, 0) ## 0 auto increment since we manually set_step!
        @info "TensorBoard logging at \"$(args.savepath)\""
    end
    function report(epoch)
        train = eval_loss_accuracy(train_loader, model, device)
        test = eval_loss_accuracy(test_loader, model, device)
        println("Epoch: $epoch   Train: $(train)   Test: $(test)")
        if args.tblogger
            append!(lossplot, test.loss)
            append!(accplot, test.acc)
            set_step!(tblogger, epoch)
            with_logger(tblogger) do
                @info "train" loss=train.loss  acc=train.acc
                @info "test"  loss=test.loss   acc=test.acc
            end
        end
    end

    @info "Start Training"
    report(0)
    for epoch in 1:args.epochs
        @showprogress for (x, y) in train_loader
            x, y = x |> device, y |> device
            gs = Flux.gradient(ps) do
                    ŷ = model(x)
                    loss(ŷ, y)
                end

            Flux.Optimise.update!(optimizer, ps, gs)
        end

        ## Printing and logging
        epoch % args.infotime == 0 && report(epoch)
        if args.checktime > 0 && epoch % args.checktime == 0
            !ispath(args.savepath) && mkpath(args.savepath)
            modelpath = joinpath(args.savepath, "DNNmodel.bson")
            let model = cpu(model) ## return model to cpu before serialization
                BSON.@save modelpath model epoch
            end
            @info "Model saved in \"$(modelpath)\""
        end
    end
end

train()






for i = 1:100
  Flux.train!(loss, params(DNN), zip(eachcol(x_train), eachcol(y_train)), ADAM())
  y_pred = DNN(x_test)
  test_loss = Flux.mse(y_pred, y_test)
  append!(lossplot, test_loss)
  println("$i round, test loss: ", test_loss)
end

savepath = "D:\\Citedal European Datathon\\DNNmodel.bson"
let DNN = cpu(model) ## return model to cpu before serialization
    BSON.@save savepath DNN
end

using BSON: @load
@load "D:\\Citedal European Datathon\\DNNmodel.bson" DNN



prediction = zeros(size(y_test,2))
for i=1:size(y_test,2)
    prediction[i] = DNN(x_test[:,i])[1]
end
plot([prediction y_test[1,:]])


Flux.mse(prediction, y_test[1,:])  # sum((prediction - y_test[1,:]) .^ 2)/size(prediction)
