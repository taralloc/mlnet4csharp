using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MLApp;

namespace MLNetWrapper
{
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
            wrapper.matlab.Execute(@"cd c:\users\franc\desktop\"); //Change directory to our folder where we have some functions
            wrapper.matlab.Execute(name + "= CreateNetwork(" + name + "," + hidden.ToString() + ")");
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
            //Simulate
            string value;
            wrapper.matlab.PutWorkspaceData("x", "base", input); //Save input to workspace
            wrapper.matlab.Execute("x = x'"); //Transpose, because MATLAB wants examples as columns
            value = wrapper.matlab.Execute(name + "(x)").Replace("ans", "").Replace("=", "").Replace("\n", ""); //Get and parse the answer
            return double.Parse(value);
        }


    }
}
