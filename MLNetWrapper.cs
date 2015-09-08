using System;
using System.Collections.Generic;
using MLApp;
using System.IO;
using System.Collections.ObjectModel;

namespace MLNetWrapper
{
    /// <summary>
    /// Specifies the training function to use. For more information see MATLAB's documentation.
    /// </summary>
    enum TrainFunctions { 
        /// <summary>Levenberg-Marquardt backpropagation</summary>
        LM,
        /// <summary>Bayesian Regulation backpropagation.</summary>
        BR,
        /// <summary>Scaled conjugate gradient backpropagation.</summary>
        SCG
    };

    /// <summary>
    /// Lists the input and output pre/post-Processing functions.
    /// </summary>
    enum ProcessingFunctions
    {
        /// <summary>Normalize inputs/targets to fall in the range [−1, 1]</summary>
        MAPMINMAX,
        /// <summary>Normalize inputs/targets to have zero mean and unity variance</summary>
        MAPSTD,
        /// <summary>Extract principal components from the input vector</summary>
        PROCESSPCA,
        /// <summary>Process unknown inputs</summary>
        FIXUNKNOWNS,
        /// <summary>Remove inputs/targets that are constant</summary>
        REMOVECONSTANTROWS
    }

    /// <summary>
    /// Creates a MATLAB instance.
    /// </summary>
    class Wrapper
    {
        public readonly MLApp.MLApp matlab = new MLApp.MLApp();
        public int index = 0;
    }

    /// <summary>
    /// Represents a full Neural Network that you can configure, train, and execute. 
    /// </summary>
    class Net
    {
        /// <summary>
        /// Gets the number of inputs of the network.
        /// </summary>
        /// <value>This property defines the number of inputs a network receives. It is a positive integer.</value>
        public int inputLayerSize { get; }
        public int hidden { get; }

        private TrainFunctions trainFcn;
        /// <summary>
        /// Gets or sets the training algorithm.
        /// </summary>
        public TrainFunctions TrainFunction
        {
            get { return trainFcn; }
            set
            {
                trainFcn = value;
                wrapper.matlab.Execute(name + ".trainFcn = " + TrainFunctionToString(TrainFunction));
            }
        }

        /// <summary>
        /// Gets pre/post-processing functions to use on inputs and outputs. You can add/remove items.
        /// </summary>
        public ObservableCollection<ProcessingFunctions> ProcessFunctions { get; } = new ObservableCollection<ProcessingFunctions>();

        private double trainRatio;
        /// <summary>
        /// Gets or sets the relative number of training samples to be selected from all samples,
        /// compared to the relative numbers of validation samples valRatio, and test samples testRatio.
        /// </summary>
        /// <value>It is a positive double in the interval (0, 1).</value>
        public double TrainRatio
        {
            get
            {
                return trainRatio;
            }
            set
            {
                if (value > 0 && value < 1)
                {
                    trainRatio = value;
                    wrapper.matlab.Execute(name + ".divideParam.trainRatio = " + trainRatio.ToString());
                }
            }
        }

        private double valRatio;
        /// <summary>
        /// Gets or sets the relative number of validation samples to be selected from all samples,
        /// compared to the relative numbers of training samples valRatio, and test samples testRatio.
        /// </summary>
        /// <value>It is a positive double in the interval (0, 1).</value>
        public double ValRatio
        {
            get
            {
                return valRatio;
            }
            set
            {
                if (value > 0 && value < 1)
                {
                    valRatio = value;
                    wrapper.matlab.Execute(name + ".divideParam.valRatio = " + valRatio.ToString());
                }
            }
        }

        private double testRatio;
        /// <summary>
        /// Gets or sets the relative number of test samples to be selected from all samples,
        /// compared to the relative numbers of validation samples valRatio, and training samples testRatio.
        /// </summary>
        /// <value>It is a positive double in the interval (0, 1).</value>
        public double TestRatio
        {
            get
            {
                return testRatio;
            }
            set
            {
                if (value > 0 && value < 1)
                {
                    testRatio = value;
                    wrapper.matlab.Execute(name + ".divideParam.testRatio = " + testRatio.ToString());
                }
            }
        }

        private bool showWindow;
        /// <summary>
        /// Whether to show the training MATLAB's window or not;
        /// </summary>
        public bool ShowWindow
        {
            get { return showWindow; }
            set
            {
                showWindow = value;
                wrapper.matlab.Execute(name + ".trainParam.showWindow = " + Convert.ToInt32(showWindow).ToString());
            }
        }

        private int epochs;
        /// <summary>
        /// Gets or sets the maximum number of training iterations before training is stopped.
        /// </summary>
        /// <value>It is a positive scalar.</value>
        public int Epochs
        {
            get { return epochs; }
            set
            {
                epochs = value;
                wrapper.matlab.Execute(name + ".trainParam.epochs = " + Epochs.ToString());
            }
        }

        /// <summary>
        /// Whether the network was trained at least once.
        /// If true, when executing 0 is returned.
        /// </summary>
        private bool untrained = true;

        /// <summary>
        /// Name of this net in the current MATLAB workspace.
        /// </summary>
        private string name;

        /// <summary>
        /// <c>Wrapper</c> used to communicate with MATLAB command line.
        /// </summary>
        private readonly Wrapper wrapper;

        /// <summary>
        /// Inizializes a new instance of the <c>Net</c> class, which represents a full network. 
        /// </summary>
        /// <param name="wrapper">The MATLAB instance to be used</param>
        /// <param name="input">Size of the input layer</param>
        /// <param name="hidden">Size of the hidden layer</param>
        public Net(Wrapper wrapper, int input, int hidden)
        {
            this.wrapper = wrapper;
            this.inputLayerSize = input;
            this.hidden = hidden;
            //Create network in MATLAB
            name = "net" + wrapper.index.ToString(); wrapper.index++; //Set the workspace name and Increase the index as we're creating a new network
            wrapper.matlab.Execute(name + " = 0"); //Initialize the variable in the workspace
            wrapper.matlab.Execute(@"trainFcn = 'trainlm'"); //Set the default training function, just because fitnet requires it, in any case we'll set it later
            wrapper.matlab.Execute(name + " = fitnet(" + this.hidden + ", trainFcn)"); //Creates the function fitting neural network

            //Set default values
            this.TrainRatio = 0.75;
            this.ValRatio = 0.15;
            this.TestRatio = 0.15;
            this.ShowWindow = true;
            this.Epochs = 1000;
            this.TrainFunction = TrainFunctions.LM;
            this.ProcessFunctions.CollectionChanged += new System.Collections.Specialized.NotifyCollectionChangedEventHandler(
                delegate (object sender, System.Collections.Specialized.NotifyCollectionChangedEventArgs e)
                {
                    //When a new item is added/removed, set the new processing functions for both input and output
                    if (e.Action == System.Collections.Specialized.NotifyCollectionChangedAction.Add || e.Action == System.Collections.Specialized.NotifyCollectionChangedAction.Remove)
                    {
                        string toexecute = "{";
                        foreach (ProcessingFunctions f in this.ProcessFunctions)
                            toexecute += "'" + ProcessingFunctionToString(f) + "',";
                        toexecute = toexecute.Remove(toexecute.Length - 1) + "}";
                        wrapper.matlab.Execute(name + ".input.processFcns = " + toexecute);
                        wrapper.matlab.Execute(name + ".output.processFcns = " + toexecute);
                    }
                }
            );
            this.ProcessFunctions.Add(ProcessingFunctions.REMOVECONSTANTROWS);
            this.ProcessFunctions.Add(ProcessingFunctions.MAPMINMAX);
        }


        /// <summary>
        /// Trains the neural network. Examples are rows.
        /// </summary>
        /// <param name="inputs">Network inputs</param>
        /// <param name="outputs">Network targets</param>
        public void Train(double[,] inputs, double[,] outputs)
        {
            //Checks
            if (inputs.GetLength(1) != inputLayerSize)
                throw new Exception("Input dimensions don't match");
            if (inputs.GetLength(0) != outputs.GetLength(0))
                throw new Exception("Number of inputs doesn't match number of outputs");
            //Execute training
            untrained = false;
            wrapper.matlab.PutWorkspaceData("x", "base", inputs); //Save inputs to workspace
            wrapper.matlab.PutWorkspaceData("y", "base", outputs); //Save outputs to workspace
            wrapper.matlab.Execute(name + " = train(" + name + ", x', y')"); //Call MATLAB's train function
        }

        /// <summary>
        /// Simulates the neural network for a single example.
        /// </summary>
        /// <param name="input">Network input</param>
        /// <returns>Predicted output</returns>
        public double Execute(double[] input)
        {
            //Checks
            if (input.Length != inputLayerSize)
                throw new Exception("Input dimensions don't match");
            if (untrained)
                return 0;
            //Simulate
            string value;
            wrapper.matlab.PutWorkspaceData("x", "base", input); //Save input to workspace
            wrapper.matlab.Execute("x = x'"); //Transpose, because MATLAB wants examples as columns
            value = wrapper.matlab.Execute(name + "(x)").Replace("ans", "").Replace("=", "").Replace("\n", ""); //Get and parse the answer
            return double.Parse(value);
        }

        //Some useful methods
        private string TrainFunctionToString(TrainFunctions f)
        {
            switch(f)
            {
                case TrainFunctions.BR:
                    return "trainbr";
                case TrainFunctions.LM:
                    return "trainlm";
                case TrainFunctions.SCG:
                    return "trainscg";
                default:
                    throw new Exception("Train Function not defined");
            }
        }

        private string ProcessingFunctionToString(ProcessingFunctions f)
        {
            switch(f)
            {
                case ProcessingFunctions.FIXUNKNOWNS:
                    return "fixunknowns";
                case ProcessingFunctions.MAPMINMAX:
                    return "mapminmax";
                case ProcessingFunctions.MAPSTD:
                    return "mapstd";
                case ProcessingFunctions.PROCESSPCA:
                    return "processpca";
                case ProcessingFunctions.REMOVECONSTANTROWS:
                    return "removeconstantrows";
                default:
                    throw new Exception("Processing Function not defined");
            }
        }


    }
}
