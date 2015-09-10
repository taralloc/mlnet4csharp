using System;
using MLNetWrapper;

namespace Test
{
    class Program
    {
        static void Main(string[] args)
        {
            // BASIC TUTORIAL

            //We'll create a neural network with 2 input units and 5 hidden units
            const int inputLayerSize = 2;
            const int hiddenLayerSize = 5;
            //Our training set, with 3 examples
            double[,] input = new double[3, 2] { { 0.1, 0.2 }, { 0.6, 0.7 }, { 0.9, 1 } };
            double[,] output = new double[3, 1] { { 0 }, { 0.8 }, { 1 } };
            //Let's start a MATLAB session
            Wrapper wrapper = new Wrapper();
            //We're going to initialize our network
            Net net = new Net(wrapper, inputLayerSize, hiddenLayerSize);
            //We can even simulate it before training: zero will be returned
            Console.WriteLine(net.Execute(new double[] { 0.9, 1 }));
            //Now let's train the network on our training set
            net.Train(input, output);
            //And simulate it again
            Console.WriteLine(net.Execute(new double[] { 0.9, 1 }));

            // MORE OPTIONS

            /* Execute() uses the calculated weigths and biases to get the predicted output.
            However you may want to simulate the network through MATLAB using the sim function of the Neural Network ToolBox.
            If so, you can set the optional parameter "usematlab" to true. Warning: it is way slower! */
            Console.WriteLine(net.Execute(new double[] { 0.9, 1 }, true));
            //Before training, you can change the Training Function
            net.TrainFunction = TrainFunctions.BR; //This uses Bayesian Regulation backpropagation
            //You can add/remove a pre/post-processing function
            net.ProcessFunctions.Remove(ProcessingFunctions.MAPMINMAX);
            //You can set different ratios for the training/validation/test sets
            net.TrainRatio = 0.9; net.ValRatio = 0.05; net.TestRatio = 0.05;
            //You can hide MATLAB's training window
            net.ShowWindow = false;
            //And set the maximum number of epochs of training
            net.Epochs = 500;
            //You can enable/disable parallelization of training. It is off by default.
            net.UseParallel = false;
            //You can read default values in the constructor of the Net class. They are the same as MATLAB's.

            //Now let's train again with these new settings
            net.Train(input, output);
            //And print the predicted output
            Console.WriteLine(net.Execute(new double[] { 0.9, 1 }));
            //When you're finished using the network, you can free its memory from MATLAB's workspace
            net.Remove();
            Console.Read();
        }
    }
   
}
