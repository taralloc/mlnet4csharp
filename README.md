# mlnet4csharp
This is C# wrapper that allows you to leverage MATLAB's Neural Network Toolbox, creating, configuring, training and simulating two-layer neural networks. Requires MATLAB.

### Example Code
```c
//Initializes the wrapper starting a MATLAB session
Wrapper wrapper = new Wrapper();
//This is our training set of 3 examples
double[,] input = new double[3, 2] { { 0.1, 0.2 }, { 0.3, 0.4 }, { 0.5, 0.6 } }; 
double[,] output = new double[3, 1] { { 0 }, { 0.8 }, { 1 } };
//Creates a network with 2 input units and 5 hidden units
Net net = new Net(wrapper, 2, 5);
//Trains using MATLAB
net.Train(input, output); 
//After training, tests the network on a new input
Console.WriteLine(net.Execute(new double[] { 0.1, 0.2 })); 
```

### Getting Started
- Add a reference in your project to *MLApp.dll*, needed to communicate with MATLAB.
- Add the *MLNetWrapper.cs* file to your project and build.
