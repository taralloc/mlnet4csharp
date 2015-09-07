using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MLApp;
using System.IO;
using MLNetWrapper;

namespace ProvaMatlab
{
    class Program
    {
        static void Main(string[] args)
        {
            double[,] a = new double[3, 2] { { 0.1, 0.2 }, { 0.6, 0.7 }, { 0.9, 1 } };
            double[,] b = new double[3,1] { { 0 }, { 0.8 }, { 1 } };
            Wrapper wrapper = new Wrapper();
            Net net = new Net(wrapper, 2, 5);
            net.Train(a, b);
            Console.WriteLine(net.Execute(new double[] { 0.9, 1 }));

            Net net2 = new Net(wrapper, 1, 5);
            net2.Train(new double[3,1] { { 0.1 }, { 0.2 }, { 0.3 } }, new double[3,1] { { 0.4 }, { 0.5 }, { 0.6 } });
            Console.WriteLine(net2.Execute(new double[] { 0.1 }));
            Console.WriteLine(net.Execute(new double[] { 0.9, 1 }));
            Console.Read();
        }
    }
   
}
