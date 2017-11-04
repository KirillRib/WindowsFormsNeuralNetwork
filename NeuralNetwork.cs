using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;


namespace NeuralNetworkLibrary
{
    public class NeuralNetwork
    {
        public const double Version = 1.00;     //Текущая версия
        public float SpeedOfTraining = 10.85f;    //Скорость обучения
        public float Erorr = -1f;
        public Layer[] LayerArray;              //Массив слоев

        //Иницилизирует новую сеть
        public NeuralNetwork(Layer[] LayerArray)
        {
            this.LayerArray = LayerArray;
        }
        public NeuralNetwork(byte[] ByteArray)
        {
            DataArray DA = new DataArray(ByteArray);
            LayerArray = new Layer[DA.ReadInt()];
            for (int i = 0; i < LayerArray.Length; i++)
            {
                int LayerType = DA.ReadInt();
                if (LayerType == 0)
                    LayerArray[i] = new NeuralLayer(DA);
                if (LayerType == 1)
                    LayerArray[i] = new ConvolutionaryLayer(DA);
                if (LayerType == 2)
                    LayerArray[i] = new CompressionLayer();
            }
            ByteArray = Addition(BitConverter.GetBytes(ByteArray.Length), ByteArray);
        }


        //Вычисляет значение по данным InputArray
        public float[] Calculate(float[] InputArray)
        {
            if (LayerArray.Length == 0 || LayerArray[0]._LayerType != Layer.LayerType.NeuralLayer)
                Console.WriteLine("Error!!! Неверный тип входных данных для данного типа сети!");

            int index = 0;
            LayerArray[index].SetInputArray(InputArray);
            for (; index < LayerArray.Length; index++)
            {
                if (LayerArray[index]._LayerType == Layer.LayerType.NeuralLayer)
                {
                    LayerArray[index].Calculate();
                    if (index != LayerArray.Length - 1)
                        LayerArray[index + 1].SetInputArray(LayerArray[index].OutputArray);
                }
                if (LayerArray[index]._LayerType == Layer.LayerType.ConvolutionaryLayer)
                {
                    LayerArray[index].Calculate();
                    if (index != LayerArray.Length - 1)
                        LayerArray[index + 1].SetInputArray(LayerArray[index].OutputMatrix);
                }
            }
            LayerArray[--index].Calculate();//!!!
            return LayerArray[index].OutputArray;
        }
        public float[] Calculate(Matrix InputMatrix)
        {
            if (LayerArray.Length == 0 || LayerArray[0]._LayerType == Layer.LayerType.NeuralLayer)
                Console.WriteLine("Error!!! Неверный тип входных данных для данного типа сети!");

            int index = 0;
            LayerArray[index].SetInputArray(InputMatrix);
            for (; index < LayerArray.Length; index++)
            {
                if (LayerArray[index]._LayerType == Layer.LayerType.NeuralLayer)
                {
                    LayerArray[index].Calculate();
                    if (index != LayerArray.Length - 1)
                        LayerArray[index + 1].SetInputArray(LayerArray[index].OutputArray);
                }
                if (LayerArray[index]._LayerType == Layer.LayerType.ConvolutionaryLayer)
                {
                    LayerArray[index].Calculate();
                    if (index != LayerArray.Length - 1)
                        LayerArray[index + 1].SetInputArray(LayerArray[index].OutputMatrix);
                }
                if (LayerArray[index]._LayerType == Layer.LayerType.CompressionLayer)
                {
                    LayerArray[index].Calculate();
                    if (index != LayerArray.Length - 1)
                        LayerArray[index + 1].SetInputArray(LayerArray[index].OutputMatrix);
                }
            }
            --index;
            return LayerArray[index].OutputArray;
        }
        //Тестирует сеть
        public float TestNeuralNetwork(TrainingSet[] TrainingSetArray)
        {
            if (LayerArray.Length == 0)
                Console.WriteLine("Error!!! Неверный тип входных данных для данного типа сети!");

            float erorr = 0;
            for (int i = 0; i < TrainingSetArray.Length; i++)
            {
                float[] a;
                if (LayerArray[0]._LayerType == Layer.LayerType.NeuralLayer)
                    a = Calculate(TrainingSetArray[i].InputArray);
                else
                    a = Calculate(TrainingSetArray[i].InputMatrix);
                if (a.Length != TrainingSetArray[i].OutputArray.Length)
                    Console.WriteLine("Error!!! Неверный тип входных данных для данного типа сети! #2");
                
                for (int j = 0; j < a.Length; j++)
                {
                    erorr += Math.Abs((a[j] - TrainingSetArray[i].OutputArray[j]) / a.Length / TrainingSetArray.Length);
                }
            }
            Erorr = erorr;
            return erorr;
        }
        //Тестирует сеть
        public float TestNeuralNetworkWorking(TrainingSet[] TrainingSetArray)
        {
            if (LayerArray.Length == 0)
                Console.WriteLine("Error!!! Неверный тип входных данных для данного типа сети!");

            float erorr = 0;
            for (int i = 0; i < TrainingSetArray.Length; i++)
            {
                float[] a;
                if (LayerArray[0]._LayerType == Layer.LayerType.NeuralLayer)
                    a = Calculate(TrainingSetArray[i].InputArray);
                else
                    a = Calculate(TrainingSetArray[i].InputMatrix);
                if (a.Length != TrainingSetArray[i].OutputArray.Length)
                    Console.WriteLine("Error!!! Неверный тип входных данных для данного типа сети! #2");

                if (a[0] > 0.5f)
                    a[0] = 1;
                else
                    a[0] = 0;

                for (int j = 0; j < a.Length; j++)
                {
                    erorr += Math.Abs((a[j] - TrainingSetArray[i].OutputArray[j]) / a.Length / TrainingSetArray.Length);
                }
            }
            Erorr = erorr;
            return erorr;
        }
        //Случаным образом изменяет сеть
        public void ToRandomlyChange(float DegreeOfChange, Random rnd)
        {
            int NChange = (int)(rnd.NextDouble() * LayerArray.Length * 1000);
            for (int i = 0; i < NChange; i++)
            {
                LayerArray[(int)(rnd.NextDouble() * LayerArray.Length)].ToRandomlyChange(DegreeOfChange, rnd);
            }
        }
        //
        public void ToCorrectLayers(float[] Answer, float[] CorrectAnswer)
        {
            for (int i = 0; i < Answer.Length; i++)
            {
                Answer[i] = (CorrectAnswer[i] - Answer[i]) * LayerArray[LayerArray.Length - 1].OutputArray[i] *
                    (1 - LayerArray[LayerArray.Length - 1].OutputArray[i]);
            }
                
            LayerArray[LayerArray.Length - 1].ToCorrectLayer(Answer, SpeedOfTraining);
            for (int i = LayerArray.Length - 2; i >= 0; i--)
            {
                if (LayerArray[i]._LayerType == Layer.LayerType.NeuralLayer)
                    LayerArray[i].ToCorrectLayer(LayerArray[i + 1].ErrorArray, SpeedOfTraining);

                if (LayerArray[i]._LayerType == Layer.LayerType.ConvolutionaryLayer && LayerArray[i + 1]._LayerType == Layer.LayerType.NeuralLayer)
                    LayerArray[i + 1].FillErrorMatrix(LayerArray[i].OutputMatrix.W, LayerArray[i].OutputMatrix.H, LayerArray[i].OutputMatrix.D);
                if (LayerArray[i]._LayerType == Layer.LayerType.ConvolutionaryLayer || LayerArray[i]._LayerType == Layer.LayerType.CompressionLayer)
                    LayerArray[i].ToCorrectLayer(LayerArray[i + 1].ErrorMatrix, SpeedOfTraining);
            }
        }
        public void EducatNetwork(TrainingSet[] TrainingSetArray)
        {
            for (int i = 0; i < TrainingSetArray.Length; i++)
            {
                float[] a;
                if (LayerArray[0]._LayerType == Layer.LayerType.NeuralLayer)
                    a = Calculate(TrainingSetArray[i].InputArray);
                else
                    a = Calculate(TrainingSetArray[i].InputMatrix);
                if (a.Length != TrainingSetArray[i].OutputArray.Length)
                    Console.WriteLine("Error!!! Неверный тип входных данных для данного типа сети! #1");
                ToCorrectLayers(a, TrainingSetArray[i].OutputArray);
            }
        }


        //Клонирует объект
        public NeuralNetwork Clone()
        {
            Layer[] newLayerArray = new Layer[LayerArray.Length];
            for (int i = 0; i < LayerArray.Length; i++)
                newLayerArray[i] = LayerArray[i].Clone();
            return new NeuralNetwork(newLayerArray);
        }

        //Возвращает параметры слоя для сохранения
        public byte[] ToByteArray()
        {
            byte[] ByteArray = new byte[0];
            ByteArray = Addition(BitConverter.GetBytes(LayerArray.Length), ByteArray);
            for (int i = 0; i < LayerArray.Length; i++)
            {
                ByteArray = Addition(ByteArray, LayerArray[i].ToByteArray());
            }
            ByteArray = Addition(BitConverter.GetBytes(ByteArray.Length), ByteArray);
            return ByteArray;
        }
        //Объединяет битные массивы
        public static byte[] Addition(byte[] a, byte[] b)
        {
            byte[] s = new byte[a.Length + b.Length];
            int index;
            for (index = 0; index < a.Length; index++)
                s[index] = a[index];
            for (int i = 0; i < b.Length; i++)
                s[i + index] = b[i];
            return s;
        }
        //Выводит в консоль
        public void WriteInConsole()
        {
            for (int i = 0; i < LayerArray.Length; i++)
            {
                Console.WriteLine("Layer {0}:", i);
                LayerArray[i].WriteInConsole();
            }
        }
        //Сохраняет сеть в фаил
        public void SaveInFile(string fileName)
        {
            using (BinaryWriter writer = new BinaryWriter(File.Open(fileName, FileMode.OpenOrCreate)))
            {
                writer.Write(ToByteArray());
            }
        }
        //Загружает сеть из файла
        public void LoadFromFile(string fileName)
        {
            using (BinaryReader reader = new BinaryReader(File.Open(fileName, FileMode.Open)))
            {
                int a = reader.ReadInt32();
                NeuralNetwork newNeuralNetwork = new NeuralNetwork(reader.ReadBytes(a));
                LayerArray = newNeuralNetwork.LayerArray;
            }
        }
    }
    public class TrainingSet
    {
        public float[] InputArray;  //Входящие значения
        public Matrix InputMatrix;  //Входящие значения
        public float[] OutputArray; //Исходящие значения
        //Инициализирует класс
        public TrainingSet(float[] InputArray, float[] OutputArray)
        {
            this.InputArray = InputArray;
            this.OutputArray = OutputArray;
        }
        public TrainingSet(Matrix InputMatrix, float[] OutputArray)
        {
            this.InputMatrix = InputMatrix;
            this.OutputArray = OutputArray;
            InputArray = null;
        }
    }

    public class Layer
    {
        public enum LayerType { NeuralLayer, ConvolutionaryLayer, CompressionLayer };
        public LayerType _LayerType;

        public virtual void SetInputArray(Matrix InputMatrix)
        {

        }
        public virtual void SetInputArray(float[] InputMatrix)
        {

        }
        public virtual void Calculate()
        {
            return;
        }
        public virtual void ToRandomlyChange(float DegreeOfChange, Random rnd)
        {

        }

        //Расчитывает дельту и ошибку
        public virtual void ToCorrectLayer(float[] NextErrorArray, float SpeedOfTraining)
        {

        }
        public virtual void ToCorrectLayer(Matrix NextErrorArray, float SpeedOfTraining)
        {

        }

        public virtual void FillErrorMatrix(int W, int H, int D)
        {

        }

        public Matrix ErrorMatrix, OutputMatrix, InputMatrix;
        public float[] OutputArray, InputArray, ErrorArray;


        public virtual byte[] ToByteArray()
        {
            byte[] ByteArray = BitConverter.GetBytes(_LayerType.GetHashCode());
            return ByteArray;
        }
        //Клонирует объект
        public virtual Layer Clone()
        {
            return null;
        }
        public static byte[] Addition(byte[] a, byte[] b)
        {
            byte[] s = new byte[a.Length + b.Length];
            int index;
            for (index = 0; index < a.Length; index++)
                s[index] = a[index];
            for (int i = 0; i < b.Length; i++)
                s[i + index] = b[i];
            return s;
        }

        //Функция активации
        public float ActivationFunction(float Input)
        {
            return 1 / (float)(1 + Math.Exp(-Input));
        }
        public float InverseActivationFunction(float Input)
        {
            return ActivationFunction(Input) * (1 - ActivationFunction(Input));
        }

        //Выводит в консоль
        public virtual void WriteInConsole()
        {

        }
    }

    public class NeuralLayer : Layer
    {
        public Neuron[] NeuronArray;    //Массив нейронов

        //Инициализирует класс NeuralLayer, генерирует случайные веса связей
        public NeuralLayer(int NumberOfNeuronsInLayer, int NumberOfNeuronsInNextLayer, Random rnd)
        {
            this._LayerType = LayerType.NeuralLayer;
            NeuronArray = new Neuron[NumberOfNeuronsInLayer];
            for (int i = 0; i < NumberOfNeuronsInLayer; i++)
            {
                NeuronArray[i] = new Neuron(new float[NumberOfNeuronsInNextLayer]);
                NeuronArray[i].SetRandomSynapseArray(rnd);
            }
        }
        //Инициализирует класс  NeuralLayer, заполняет веса значениями из DataArray
        public NeuralLayer(DataArray DA)
        {
            this._LayerType = LayerType.NeuralLayer;

            NeuronArray = new Neuron[DA.ReadInt()];
            for (int i = 0; i < NeuronArray.Length; i++)
                NeuronArray[i] = new Neuron(DA);
        }
        //Инициализирует класс  NeuralLayer, заполняет веса значениями из DataArray
        public NeuralLayer(Neuron[] NeuronArray)
        {
            this._LayerType = LayerType.NeuralLayer;
            this.NeuronArray = NeuronArray;
        }

        //Вычисляет значения массива OutputArray из массива InputArray
        public override void Calculate()
        {
            if (InputArray == null && InputArray.Length == NeuronArray.Length)
                Console.WriteLine("Error! Ошибка входящего массива");
            for (int i = 0; i < InputArray.Length; i++)
                NeuronArray[i].Value = InputArray[i];
            OutputArray = new float[NeuronArray[0].SynapseArray.Length];
            for (int i = 0; i < OutputArray.Length; i++)
                OutputArray[i] = 0;

            for (int i = 0; i < InputArray.Length; i++)
                for (int j = 0; j < NeuronArray[0].SynapseArray.Length; j++)
                    OutputArray[j] += InputArray[i] * NeuronArray[i].SynapseArray[j] / NeuronArray.Length;

            for (int i = 0; i < OutputArray.Length; i++)
                OutputArray[i] = ActivationFunction(OutputArray[i]);
        }

        //Заполняет массив InputArray
        public override void SetInputArray(Matrix InputMatrix)
        {
            InputArray = InputMatrix.ToFloatArray();
        }
        //Заполняет массив InputArray
        public override void SetInputArray(float[] inputArray)
        {
            InputArray = inputArray;
        }


        //Расчитывает дельту и ошибку
        public override void ToCorrectLayer(float[] NextErrorArray, float SpeedOfTraining)
        {
            ErrorArray = new float[NeuronArray.Length];
            for (int i = 0; i < NeuronArray.Length; i++)
            {
                ErrorArray[i] = 0;
                for (int j = 0; j < NextErrorArray.Length; j++)
                    ErrorArray[i] += NeuronArray[i].SynapseArray[j] * NextErrorArray[j] / NeuronArray.Length;
                ErrorArray[i] *= InputArray[i] * (1 - InputArray[i]);
            }
            for (int i = 0; i < NeuronArray.Length; i++)
                for (int j = 0; j < NeuronArray[i].SynapseArray.Length; j++)
                    NeuronArray[i].SynapseArray[j] += SpeedOfTraining * NextErrorArray[j] * InputArray[i];
        }

        //-------!!! Не проверенная функция !!!-------
        public override void FillErrorMatrix(int W, int H, int D)
        {
            ErrorMatrix = new Matrix(W, H, D);
            for (int i = 0; i < W; i++)
                for (int j = 0; j < H; j++)
                    for (int k = 0; k < D; k++)
                        ErrorMatrix.matrix[i][j][k] = ErrorArray[i + j * W + k * W * H];
        }

        //Вносит случайнык изменения
        public override void ToRandomlyChange(float DegreeOfChange, Random rnd)
        {
            int index = (int)(rnd.NextDouble() * NeuronArray.Length);
            if (NeuronArray[index].SynapseArray.Length != 0)
                NeuronArray[index].SynapseArray[(int)(rnd.NextDouble() * NeuronArray[index].SynapseArray.Length)] *= (float)(rnd.NextDouble() * DegreeOfChange);//(float)(rnd.NextDouble() * 2 - 1) * DegreeOfChange + 1 + (float)rnd.NextDouble();
        }

        //Возвращает параметры слоя для сохранения
        public override byte[] ToByteArray()
        {
            byte[] ByteArray = BitConverter.GetBytes(_LayerType.GetHashCode());
            ByteArray = Addition(ByteArray, BitConverter.GetBytes(NeuronArray.Length));
            for (int i = 0; i < NeuronArray.Length; i++)
            {
                ByteArray = Addition(ByteArray, NeuronArray[i].ToByteArray());
            }
            return ByteArray;
        }
        //Клонирует объект
        public override Layer Clone()
        {
            Neuron[] newNeuronArray = new Neuron[NeuronArray.Length];
            for (int i = 0; i < NeuronArray.Length; i++)
                newNeuronArray[i] = NeuronArray[i].Clone();
            return new NeuralLayer(newNeuronArray);
        }
        //Выводит в консоль
        public override void WriteInConsole()
        {
            for (int i = 0; i < NeuronArray.Length; i++)
            {
                Console.WriteLine("Neuron {0}:", i);
                NeuronArray[i].WriteInConsole();
            }
        }
    }
    public class ConvolutionaryLayer : Layer
    {
        private Matrix[] ArrayFilters;
        public int WFilters, HFilters;
        private Matrix BiasMatrix;


        //Инициализирует класс ConvolutionaryLayer
        public ConvolutionaryLayer(int WFilters, int HFilters, int DepthOfInputMatrix, int NFilters, Random rnd)
        {
            this._LayerType = LayerType.ConvolutionaryLayer;

            ArrayFilters = new Matrix[NFilters];
            for (int i = 0; i < NFilters; i++)
            {
                ArrayFilters[i] = new Matrix(WFilters, HFilters, DepthOfInputMatrix, rnd);
            }
            this.WFilters = WFilters;
            this.HFilters = HFilters;
        }
        public ConvolutionaryLayer(Matrix[] newArrayFilters, int WFilters, int HFilters)
        {
            this._LayerType = LayerType.ConvolutionaryLayer;

            ArrayFilters = newArrayFilters;
            this.WFilters = WFilters;
            this.HFilters = HFilters;
        }
        public ConvolutionaryLayer(DataArray DA)
        {
            this._LayerType = LayerType.ConvolutionaryLayer;

            ArrayFilters = new Matrix[DA.ReadInt()];
            this.WFilters = DA.ReadInt();
            this.HFilters = DA.ReadInt();
            for (int i = 0; i < ArrayFilters.Length; i++)
            {
                ArrayFilters[i] = new Matrix(DA);
            }
        }

        //Вычисляет значения массива OutputArray из массива InputArray
        public override void Calculate()
        {
            if (InputMatrix == null)
                Console.WriteLine("Error! Нет входящей матрицы");
            OutputMatrix = new Matrix(InputMatrix.W - WFilters + 1, InputMatrix.H - HFilters + 1, ArrayFilters.Length);
            for (int i = 0; i < OutputMatrix.W; i++)
                for (int j = 0; j < OutputMatrix.H; j++)
                    for (int k = 0; k < OutputMatrix.D; k++)
                        OutputMatrix.matrix[i][j][k] = ActivationFunction(InputMatrix.MultiplyPartOfTheMatrix(i, j, ArrayFilters[k]) / WFilters / HFilters - 17);
        }

        //Заполняет массив InputArray
        public override void SetInputArray(Matrix InputMatrix)
        {
            this.InputMatrix = InputMatrix;
        }

        //Случаным образом изменяет сеть
        public override void ToRandomlyChange(float DegreeOfChange, Random rnd)
        {
            ArrayFilters[(int)(rnd.NextDouble() * ArrayFilters.Length)].ToRandomlyChange(DegreeOfChange, rnd);
        }


        //-------!!! Не проверенная функция !!!-------
        public override void ToCorrectLayer(Matrix NextErrorMatrix, float SpeedOfTraining)
        {
            ErrorMatrix = new Matrix(InputMatrix.W, InputMatrix.H, InputMatrix.D);
            for (int i = 0; i < NextErrorMatrix.W; i++)
                for (int j = 0; j < NextErrorMatrix.H; j++)
                    for (int f = 0; f < NextErrorMatrix.D; f++)
                        ErrorMatrix.AddPartOfMatrix(i, j, NextErrorMatrix.matrix[i][j][f] * ArrayFilters[f]);

            for (int i = 0; i < ErrorMatrix.W; i++)
                for (int j = 0; j < ErrorMatrix.H; j++)
                    for (int f = 0; f < ErrorMatrix.D; f++)
                        ErrorMatrix.matrix[i][j][f] *= InputMatrix.matrix[i][j][f] * (1 - InputMatrix.matrix[i][j][f]) / WFilters / HFilters;// Добавленное  / WFilters / HFilters возможно не нужно

            for (int i = 0; i < NextErrorMatrix.W; i++)
                for (int j = 0; j < NextErrorMatrix.H; j++)
                    for (int f = 0; f < NextErrorMatrix.D; f++)
                        //
                        for (int Xf = 0; Xf < ArrayFilters[f].W; Xf++)
                            for (int Yf = 0; Yf < ArrayFilters[f].H; Yf++)
                                for (int Zf = 0; Zf < ArrayFilters[f].D; Zf++)
                                    ArrayFilters[f].matrix[Xf][Yf][Zf] += SpeedOfTraining * NextErrorMatrix.matrix[i][j][f] * InputMatrix.matrix[i + Xf][j + Yf][Zf];
        }


        //Возвращает параметры слоя для сохранения
        public override byte[] ToByteArray()
        {
            byte[] ByteArray = BitConverter.GetBytes(_LayerType.GetHashCode());
            ByteArray = Addition(ByteArray, BitConverter.GetBytes(ArrayFilters.Length));
            ByteArray = Addition(ByteArray, BitConverter.GetBytes(WFilters));
            ByteArray = Addition(ByteArray, BitConverter.GetBytes(HFilters));
            for (int i = 0; i < ArrayFilters.Length; i++)
            {
                ByteArray = Addition(ByteArray, ArrayFilters[i].ToByteArray());
            }
            return ByteArray;
        }
        //Клонирует объект
        public override Layer Clone()
        {
            Matrix[] newArrayFilters = new Matrix[ArrayFilters.Length];
            for (int i = 0; i < newArrayFilters.Length; i++)
                newArrayFilters[i] = ArrayFilters[i].Clone();
            return new ConvolutionaryLayer(newArrayFilters, WFilters, HFilters);
        }
        //Выводит в консоль
        public override void WriteInConsole()
        {
            for (int i = 0; i < ArrayFilters.Length; i++)
            {
                Console.WriteLine("Filter {0}:", i);
                ArrayFilters[i].WriteInConsole();
            }
        }
        public Matrix[] GetArrayFilters()
        {
            return ArrayFilters;
        }

        //Сохраняет слой в фаил
        public void SaveInFile(string fileName)
        {
            using (BinaryWriter writer = new BinaryWriter(File.Open(fileName, FileMode.OpenOrCreate)))
            {
                byte[] ByteArray = ToByteArray();
                ByteArray = Addition(BitConverter.GetBytes(ByteArray.Length), ByteArray);
                writer.Write(ByteArray);
            }
        }
        //Загружает слой из файла
        public void LoadFromFile(string fileName)
        {
            using (BinaryReader reader = new BinaryReader(File.Open(fileName, FileMode.Open)))
            {
                int a = reader.ReadInt32();
                int t = reader.ReadInt32();
                DataArray DA = new DataArray(reader.ReadBytes(a));
                ConvolutionaryLayer newConvolutionaryLayer = new ConvolutionaryLayer(DA);
                this.ArrayFilters = newConvolutionaryLayer.ArrayFilters;
            }
        }
    }
        public class CompressionLayer : Layer
    {
        public CompressionLayer()
        {
            this._LayerType = LayerType.CompressionLayer;
        }
        public override void Calculate()
        {
            OutputMatrix = new Matrix((int)(InputMatrix.W / 2), (int)(InputMatrix.H / 2), InputMatrix.D);
            for (int w = 0; w < OutputMatrix.W; w++)
                for (int h = 0; h < OutputMatrix.H; h++)
                    for (int d = 0; d < OutputMatrix.D; d++)
                        OutputMatrix.matrix[w][h][d] = Math.Max(Math.Max(InputMatrix.matrix[w * 2][h * 2][d], InputMatrix.matrix[w * 2 + 1][h * 2][d]),
                            Math.Max(InputMatrix.matrix[w * 2][h * 2 + 1][d], InputMatrix.matrix[w * 2 + 1][h * 2 + 1][d]));
        }

        //Заполняет массив InputArray
        public override void SetInputArray(Matrix InputMatrix)
        {
            this.InputMatrix = InputMatrix;
        }

        public override void ToCorrectLayer(Matrix NextErrorArray, float SpeedOfTraining)
        {
            ErrorMatrix = new Matrix(InputMatrix.W, InputMatrix.H, InputMatrix.D);
            for (int i = 0; i < ErrorMatrix.W; i++)
                for (int j = 0; j < ErrorMatrix.H; j++)
                    for (int k = 0; k < ErrorMatrix.D; k++)
                        ErrorMatrix.matrix[i][j][k] = NextErrorArray.matrix[i / 2][j / 2][k];
        }
        public override byte[] ToByteArray()
        {
            byte[] ByteArray = BitConverter.GetBytes(_LayerType.GetHashCode());
            return ByteArray;
        }
        //Клонирует объект
        public override Layer Clone()
        {
            return new CompressionLayer();
        }
    }



    public class Neuron
    {
        public float Value;
        public float[] SynapseArray;

        public Neuron(float[] SynapseArray)
        {
            Value = 0;
            this.SynapseArray = SynapseArray;
        }
        public Neuron(DataArray DA)
        {
            Value = 0;
            SynapseArray = new float[DA.ReadInt()];
            for (int i = 0; i < SynapseArray.Length; i++)
                SynapseArray[i] = DA.ReadFloat();
        }
        public void SetRandomSynapseArray(Random rnd)
        {
            for (int i = 0; i < SynapseArray.Length; i++)
                SynapseArray[i] = (float)(rnd.NextDouble()) * 10;// - 0.5f
        }

        public byte[] ToByteArray()
        {
            byte[] ByteArray = new byte[0];
            float[] F = SynapseArray;
            ByteArray = Addition(ByteArray, BitConverter.GetBytes(SynapseArray.Length));
            for (int i = 0; i < F.Length; i++)
            {
                ByteArray = Addition(ByteArray, BitConverter.GetBytes(F[i]));
            }
            return ByteArray;
        }
        public static byte[] Addition(byte[] a, byte[] b)
        {
            byte[] s = new byte[a.Length + b.Length];
            int index;
            for (index = 0; index < a.Length; index++)
                s[index] = a[index];
            for (int i = 0; i < b.Length; i++)
                s[i + index] = b[i];
            return s;
        }
        //Клонирует объект
        public Neuron Clone()
        {
            float[] newSynapseArray = new float[SynapseArray.Length];
            for (int i = 0; i < SynapseArray.Length; i++)
                newSynapseArray[i] = SynapseArray[i];
            return new Neuron(newSynapseArray);
        }
        public void WriteInConsole()
        {
            for (int i = 0; i < SynapseArray.Length; i++)
            {
                
                Console.Write("{0, 5:f3}\t", SynapseArray[i]);
            }
            Console.Write('\n');
        }
    }
    public class Matrix
    {
        public float[][][] matrix;
        public int W, H, D;
        public Matrix(int W, int H, int D)
        {
            matrix = new float[W][][];
            for (int i = 0; i < W; i++)
            {
                matrix[i] = new float[H][];
                for (int j = 0; j < H; j++)
                {
                    matrix[i][j] = new float[D];
                    for (int k = 0; k < D; k++)
                    {
                        matrix[i][j][k] = 0;
                    }
                }
            }
            this.W = W;
            this.H = H;
            this.D = D;
        }
        public Matrix(int W, int H, int D, float v)
        {
            matrix = new float[W][][];
            for (int i = 0; i < W; i++)
            {
                matrix[i] = new float[H][];
                for (int j = 0; j < H; j++)
                {
                    matrix[i][j] = new float[D];
                    for (int k = 0; k < D; k++)
                    {
                        matrix[i][j][k] = v;
                    }
                }
            }
            this.W = W;
            this.H = H;
            this.D = D;
        }
        public Matrix(int W, int H, int D, float[][][] newMatrix)
        {
            matrix = newMatrix;
            this.W = W;
            this.H = H;
            this.D = D;
        }
        public Matrix(int W, int H, int D, Random rnd)
        {
            matrix = new float[W][][];
            for (int i = 0; i < W; i++)
            {
                matrix[i] = new float[H][];
                for (int j = 0; j < H; j++)
                {
                    matrix[i][j] = new float[D];
                    for (int k = 0; k < D; k++)
                    {
                        matrix[i][j][k] = (float)rnd.NextDouble();
                    }
                }
            }
            this.W = W;
            this.H = H;
            this.D = D;
        }
        public Matrix(DataArray DA)
        {
            W = DA.ReadInt();
            H = DA.ReadInt();
            D = DA.ReadInt();
            matrix = new float[W][][];
            for (int i = 0; i < W; i++)
            {
                matrix[i] = new float[H][];
                for (int j = 0; j < H; j++)
                {
                    matrix[i][j] = new float[D];
                    for (int k = 0; k < D; k++)
                    {
                        matrix[i][j][k] = DA.ReadFloat();
                    }
                }
            }
        }
        public Matrix(Bitmap _Image)
        {
            matrix = new float[_Image.Width][][];
            for (int i = 0; i < _Image.Width; i++)
            {
                matrix[i] = new float[_Image.Height][];
                for (int j = 0; j < _Image.Height; j++)
                {
                    Color C = _Image.GetPixel(i, j);
                    matrix[i][j] = new float[3];
                    matrix[i][j][0] = 1 - (float)C.R / 255;
                    matrix[i][j][1] = 1 - (float)C.G / 255;
                    matrix[i][j][2] = 1 - (float)C.B / 255;
                }
            }
            W = _Image.Width;
            H = _Image.Height;
            D = 3;
        }


        public void AddPartOfMatrix(int X, int Y, Matrix P)
        {
            if (X < 0 || Y < 0 || X + P.W > W || Y + P.H > H || P.D != D)
                Console.WriteLine("Error! Ошибка при добавлении матрицы!");
            for (int i = 0; i < P.W; i++)
                for (int j = 0; j < P.H; j++)
                    for (int k = 0; k < P.D; k++)
                        matrix[i + X][j + Y][k] += P.matrix[i][j][k];
        }
        public float MultiplyPartOfTheMatrix(int x, int y, Matrix M)
        {
            if (x < 0 || y < 0 || x + M.W > W || y + M.H > H || M.D != D)
                Console.WriteLine("Error! Ошибка при умножении матрицы!");
            float s = 0;
            for (int i = 0; i < M.W; i++)
                for (int j = 0; j < M.H; j++)
                    for (int k = 0; k < M.D; k++)
                    {
                        s += M.matrix[i][j][k] * matrix[i + x][j + y][k];
                    }
            return s;
        }
        public float[] ToFloatArray()
        {
            float[] a = new float[W * H * D];
            for (int i = 0; i < W; i++)
                for (int j = 0; j < H; j++)
                    for (int k = 0; k < D; k++)
                        a[i + j * W + k * W * H] = matrix[i][j][k];
            return a;
        }

        public static float operator *(Matrix a, Matrix b)
        {
            if (a.W != b.W || a.H != b.H || a.D != b.D)
                Console.WriteLine("Erorr! Невозможно сложить матрицы!");
            float s = 0;
            for (int i = 0; i < a.W; i++)
                for (int j = 0; j < a.H; j++)
                    for (int k = 0; k < a.D; k++)
                        s += a.matrix[i][j][k] * b.matrix[i][j][k];
            return s;
        }
        public static Matrix operator *(float a, Matrix b)
        {
            Matrix s = new Matrix(b.W, b.H, b.D);
            for (int i = 0; i < b.W; i++)
                for (int j = 0; j < b.H; j++)
                    for (int k = 0; k < b.D; k++)
                        s.matrix[i][j][k] = b.matrix[i][j][k] * a;
            return s;
        }
        public static Matrix operator +(Matrix a, Matrix b)
        {
            if (a.W != b.W || a.H != b.H || a.D != b.D)
                Console.WriteLine("Erorr! Невозможно сложить матрицы!");
            Matrix s = new Matrix(a.W, a.H, a.D);
            for (int i = 0; i < a.W; i++)
                for (int j = 0; j < a.H; j++)
                    for (int k = 0; k < a.D; k++)
                        s.matrix[i][j][k] += a.matrix[i][j][k] + b.matrix[i][j][k];
            return s;
        }
        public static Matrix operator -(Matrix a, Matrix b)
        {
            if (a.W != b.W || a.H != b.H || a.D != b.D)
                Console.WriteLine("Erorr! Невозможно сложить матрицы!");
            Matrix s = new Matrix(a.W, a.H, a.D);
            for (int i = 0; i < a.W; i++)
                for (int j = 0; j < a.H; j++)
                    for (int k = 0; k < a.D; k++)
                        s.matrix[i][j][k] += a.matrix[i][j][k] - b.matrix[i][j][k];
            return s;
        }
        public byte[] ToByteArray()
        {
            byte[] ByteArray = new byte[0];
            ByteArray = Addition(ByteArray, BitConverter.GetBytes(W));
            ByteArray = Addition(ByteArray, BitConverter.GetBytes(H));
            ByteArray = Addition(ByteArray, BitConverter.GetBytes(D));
            for (int i = 0; i < W; i++)
                for (int j = 0; j < H; j++)
                    for (int k = 0; k < D; k++)
                        ByteArray = Addition(ByteArray, BitConverter.GetBytes(matrix[i][j][k]));
            return ByteArray;
        }
        public float GetMax()
        {
            float max = float.MinValue;
            for (int i = 0; i < W; i++)
                for (int j = 0; j < H; j++)
                    for (int k = 0; k < D; k++)
                        if (max < matrix[i][j][k])
                            max = matrix[i][j][k];
            return max;
        }
        public float[] GetMaxIndex()
        {
            float[] max = new float[4] { float.MinValue, -1, -1, -1 };
            string s = "";
            for (int i = 0; i < W; i++)
                for (int j = 0; j < H; j++)
                    for (int k = 0; k < D; k++)
                        if (max[0] < matrix[i][j][k])
                        {
                            max[0] = matrix[i][j][k];
                            max[1] = i;
                            max[2] = j;
                            max[3] = k;
                        }
                            
            return max;
        }
        public void ToNormalizeVectors(float m)
        {
            float _m = 0;
            for (int i = 0; i < W; i++)
                for (int j = 0; j < H; j++)
                {
                    float l = 0;
                    for (int k = 0; k < D; k++)
                        l += matrix[i][j][k] * matrix[i][j][k];
                    _m += (float)Math.Sqrt(l) / W / H;
                }
            _m = m / _m;
            for (int i = 0; i < W; i++)
                for (int j = 0; j < H; j++)
                {
                    for (int k = 0; k < D; k++)
                        matrix[i][j][k] *= _m;
                }
        }

        //Случаным образом изменяет сеть
        public void ToRandomlyChange(float DegreeOfChange, Random rnd)
        {
            matrix[(int)(rnd.NextDouble() * W)][(int)(rnd.NextDouble() * H)][(int)(rnd.NextDouble() * D)] *= (float)(rnd.NextDouble() * DegreeOfChange);
        }

        public static byte[] Addition(byte[] a, byte[] b)
        {
            byte[] s = new byte[a.Length + b.Length];
            int index;
            for (index = 0; index < a.Length; index++)
                s[index] = a[index];
            for (int i = 0; i < b.Length; i++)
                s[i + index] = b[i];
            return s;
        }
        //Клонирует объект
        public Matrix Clone()
        {
            float[][][] newMatrix = new float[W][][];
            for (int i = 0; i < W; i++)
            {
                newMatrix[i] = new float[H][];
                for (int j = 0; j < H; j++)
                {
                    newMatrix[i][j] = new float[D];
                    for (int k = 0; k < D; k++)
                    {
                        newMatrix[i][j][k] = this.matrix[i][j][k];
                    }
                }
            }
            return new Matrix(W, H, D, newMatrix);
        }
        public void WriteInConsole()
        {
            for (int h = 0; h < H; h++)
            {
                for (int d = 0; d < D; d++)
                {
                    Console.Write('\t');
                    for (int w = 0; w < W; w++)
                    {
                        Console.Write("{0, 5:f3}\t", matrix[w][h][d]);
                    }
                }
                Console.Write('\n');
            }
        }
        public float GetAverage()
        {
            float s = 0;
            for (int i = 0; i < W; i++)
            {
                for (int j = 0; j < H; j++)
                {
                    for (int k = 0; k < D; k++)
                    {
                        s += this.matrix[i][j][k];
                    }
                }
            }
            return s / W / H / D;
        }
    }
    public class DataArray
    {
        byte[] Array;
        public int index;
        public DataArray(byte[] Array)
        {
            this.Array = Array;
        }
        public int ReadInt()
        {
            index += 4;
            return BitConverter.ToInt32(Array, index - 4);
        }
        public float ReadFloat()
        {
            index += 4;
            return BitConverter.ToSingle(Array, index - 4);
        }
        public double ReadDouble()
        {
            index += 8;
            return BitConverter.ToDouble(Array, index - 8);
        }
    }
}
