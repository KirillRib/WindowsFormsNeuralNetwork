using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Windows.Forms;
using AForge.Video;
using AForge.Video.DirectShow;
using AForge.Vision;
using System.Threading;
using System.Drawing.Imaging;
using NeuralNetworkLibrary;
using System.Runtime.InteropServices;

namespace WindowsFormsNeuralNetwork
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
            this.KeyDown += new System.Windows.Forms.KeyEventHandler(this.Form1_KeyDown);
            this.KeyPreview = true;
        }
        private void Form1_Load(object sender, EventArgs e)
        {
            pictureBox1.SizeMode = PictureBoxSizeMode.Zoom;
            videodevices = new FilterInfoCollection(FilterCategory.VideoInputDevice);
            videoSource = new VideoCaptureDevice(videodevices[1].MonikerString);
            videoSourcePlayer1.VideoSource = videoSource;

            System.DateTime DT = DateTime.Now;
            Bitmap b = new Bitmap(new Bitmap(@"C:\Users\Kirill\Pictures\000.jpg"), 200, 150);

            for (int i = 0;i < 10; i++)
                new HOG(b);

            Console.WriteLine("Time 1: {0}s", (DateTime.Now - DT).TotalSeconds);

            Random rnd = new Random();

            ConvolutionaryLayerArray = new ConvolutionaryLayer[3];
            ConvolutionaryLayerArrayFace = new ConvolutionaryLayer[3];
            for (int j = 0; j < ConvolutionaryLayerArray.Length; j++)
            {
                int w = (3 * (j + 3)) / 3;
                int h = (3 * (j + 3)) / 3;

                ConvolutionaryLayerArray[j] = new ConvolutionaryLayer(new Matrix[] { 5 * new Matrix(w, h, 8, rnd), 5 * new Matrix(w, h, 8, rnd) , 5 * new Matrix(w, h, 8, rnd)}, w, h);

                ConvolutionaryLayerArrayFace[j] = new ConvolutionaryLayer( new Matrix[] { 5 * new Matrix(w * 3, h * 3, 3, rnd), 5 * new Matrix(w * 3, h * 3, 3, rnd), 5 * new Matrix(w * 3, h * 3, 3, rnd) }, w * 3, h * 3);



                ConvolutionaryLayerArrayFace[j].LoadFromFile("F" + j.ToString());
                ConvolutionaryLayerArray[j].LoadFromFile("EM" + j.ToString());
            }


            pictureBox8.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBox9.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBox10.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBox2.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBox3.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBox4.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBox5.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBox6.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBox7.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBox13.SizeMode = PictureBoxSizeMode.Zoom;

            Thread myThread = new Thread(TreatmentImage);//Training TreatmentImage
            myThread.Start();
            //Training();


        }


        NeuralNetwork mNeuralNetwork;
        ConvolutionaryLayer[] ConvolutionaryLayerArray, ConvolutionaryLayerArrayFace;
        
        FilterInfoCollection videodevices;
        VideoCaptureDevice videoSource;
        private void videoSourcePlayer1_NewFrame(object sender, ref Bitmap image)
        {
            //BM = new Bitmap(image, 400, 300);
            BM = new Bitmap(image, 300, 220);
            //BM = new Bitmap(image, 480, 320);
        }

        ImageFormat IF = System.Drawing.Imaging.ImageFormat.Png;
        Bitmap BM;
        public float Chance = 0;
        int ind = 0;

        public bool ShowTestResults = true;

        private void SaveImage()
        {
            videoSource.Start();
            string name = @"C:/Users/Kirill/Pictures/Image/";
            string[] dirs = Directory.GetFiles(name, " *.jpg");
            ind = dirs.Length;
            for (int i = 0; i < 100; i++)
            {
                if (BM != null)
                {
                    BM.Save(name + ind + ".jpg", IF);
                    ind++;
                }
                    
                
                Thread.Sleep(150);
            }
            Console.WriteLine("Stop");
        }


        private void TreatmentImage()
        {

            videoSource.Start();
            pictureBox2.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBox3.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBox4.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBox5.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBox6.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBox7.SizeMode = PictureBoxSizeMode.Zoom;
            Matrix pHOG = new Matrix(30, 22, 8);

            while (true)
            {
                if (BM != null)
                {
                    System.DateTime DT = DateTime.Now;
                    HOG hog = new HOG(BM);
                    Console.WriteLine("TimeHOG: {0}s", (DateTime.Now - DT).TotalSeconds);
                    float[] a;
                    int index = 0;
                    float max = -100;

                    for (int k = 0; k < ConvolutionaryLayerArray.Length; k++)
                    {
                        ConvolutionaryLayerArray[k].InputMatrix = hog.HOGMatrix;
                        ConvolutionaryLayerArray[k].Calculate();
                        

                        ConvolutionaryLayerArrayFace[k].InputMatrix = ConvolutionaryLayerArray[k].OutputMatrix;
                        ConvolutionaryLayerArrayFace[k].Calculate();
                        

                        if (ConvolutionaryLayerArrayFace[k].OutputMatrix.GetMax() > max)
                        {
                            max = ConvolutionaryLayerArrayFace[k].OutputMatrix.GetMax();
                            index = k;
                        }
                        //Console.WriteLine("Time1: {0}s", (DateTime.Now - DT).TotalSeconds);
                    }

                    a = ConvolutionaryLayerArrayFace[index].OutputMatrix.GetMaxIndex();


                    Console.WriteLine(a[0]);
                    Console.WriteLine("Full Time:           {0}s", (DateTime.Now - DT).TotalSeconds);

                    pHOG = pHOG - hog.HOGMatrixWithoutNormalization;
                    Matrix m = pHOG.GetModuleMatrix();
                    //pictureBox13.Image = MatrixToBitmap(pHOG.GetModuleMatrix().GetMatrixSum(2, 2));
                    pHOG = hog.HOGMatrixWithoutNormalization;

                    if (max > 0.25f)
                        BM = AddFrameOnImage(BM, (int)a[1] * 10, (int)a[2] * 10, ConvolutionaryLayerArrayFace[index].GetArrayFilters()[(int)a[3]].W * 10, ConvolutionaryLayerArrayFace[index].GetArrayFilters()[(int)a[3]].H * 10);



                    //compressionLayer.InputMatrix = answer;
                    //compressionLayer.Calculate();
                    //answer = compressionLayer.OutputMatrix;

                    //pictureBox2.Image = MatrixToBitmap(ConvolutionaryLayerArrayFace[0].OutputMatrix, 0);
                    //pictureBox3.Image = MatrixToBitmap(ConvolutionaryLayerArrayFace[0].OutputMatrix, 1);
                    //pictureBox4.Image = MatrixToBitmap(ConvolutionaryLayerArrayFace[0].OutputMatrix, 2);
                    //pictureBox5.Image = MatrixToBitmap(ConvolutionaryLayerArrayFace[1].OutputMatrix, 0);
                    //pictureBox6.Image = MatrixToBitmap(ConvolutionaryLayerArrayFace[1].OutputMatrix, 1);
                    //pictureBox7.Image = MatrixToBitmap(ConvolutionaryLayerArrayFace[1].OutputMatrix, 2);
                    //pictureBox8.Image = MatrixToBitmap(ConvolutionaryLayerArrayFace[2].OutputMatrix, 0);
                    //pictureBox9.Image = MatrixToBitmap(ConvolutionaryLayerArrayFace[2].OutputMatrix, 1);
                    //pictureBox10.Image = MatrixToBitmap(ConvolutionaryLayerArrayFace[2].OutputMatrix, 2);


                    //pictureBox5.Image = MatrixToBitmap(1 / ConvolutionaryLayerArray[2].OutputMatrix.GetMax() * ConvolutionaryLayerArray[2].OutputMatrix);
                    //pictureBox6.Image = MatrixToBitmap(1 / ConvolutionaryLayerArray[3].OutputMatrix.GetMax() * ConvolutionaryLayerArray[3].OutputMatrix);
                    //pictureBox7.Image = MatrixToBitmap(1 / ConvolutionaryLayerArray[4].OutputMatrix.GetMax() * ConvolutionaryLayerArray[4].OutputMatrix);

                    //pictureBox2.Image = MatrixToBitmap(1 / ConvolutionaryLayerArray[0].OutputMatrix.GetMax(0) * ConvolutionaryLayerArray[0].OutputMatrix, 0);
                    //pictureBox3.Image = MatrixToBitmap(1 / ConvolutionaryLayerArray[1].OutputMatrix.GetMax(0) * ConvolutionaryLayerArray[1].OutputMatrix, 0);
                    //pictureBox4.Image = MatrixToBitmap(1 / ConvolutionaryLayerArray[2].OutputMatrix.GetMax(0) * ConvolutionaryLayerArray[2].OutputMatrix, 0);


                    //pictureBox5.Image = MatrixToBitmap(1 / ConvolutionaryLayerArray[0].OutputMatrix.GetMax(1) * ConvolutionaryLayerArray[0].OutputMatrix, 1);
                    //pictureBox6.Image = MatrixToBitmap(1 / ConvolutionaryLayerArray[1].OutputMatrix.GetMax(1) * ConvolutionaryLayerArray[1].OutputMatrix, 1);
                    //pictureBox7.Image = MatrixToBitmap(1 / ConvolutionaryLayerArray[2].OutputMatrix.GetMax(1) * ConvolutionaryLayerArray[2].OutputMatrix, 1);


                    //pictureBox8.Image = MatrixToBitmap(1 / ConvolutionaryLayerArray[0].OutputMatrix.GetMax(2) * ConvolutionaryLayerArray[0].OutputMatrix, 2);
                    //pictureBox9.Image = MatrixToBitmap(1 / ConvolutionaryLayerArray[1].OutputMatrix.GetMax(2) * ConvolutionaryLayerArray[1].OutputMatrix, 2);
                    //pictureBox10.Image = MatrixToBitmap(1 / ConvolutionaryLayerArray[2].OutputMatrix.GetMax(2) * ConvolutionaryLayerArray[2].OutputMatrix, 2);

                    //pictureBox11.Image = MatrixToBitmap(1 / max * ConvolutionaryLayerArray[3].OutputMatrix, 1);
                    //pictureBox12.Image = MatrixToBitmap(1 / max * ConvolutionaryLayerArray[4].OutputMatrix, 1);

                    

                    pictureBox1.Image = BM;

                    //pictureBox13.Image = hog.ToBitmap(20);

                }
                Thread.Sleep(20);
            }
        }

        private Bitmap AddFrameOnImage(Bitmap InputBitmap, int x, int y, int w, int h)
        {
            Graphics g = Graphics.FromImage(InputBitmap);

            g.DrawLine(new Pen(Color.Red), x + 0, y + 0, x + w, y + 0);
            g.DrawLine(new Pen(Color.Red), x + w, y + 0, x + w, y + h);
            g.DrawLine(new Pen(Color.Red), x + w, y + h, x + 0, y + h);
            g.DrawLine(new Pen(Color.Red), x + 0, y + h, x + 0, y + 0);

            g.Dispose();
            return InputBitmap;
        }


        private void Training()
        {
            int p = 200;
            float Erorr = 0;
            TrainingSet[] TrainingSetArray = LoadImages();
            for (int i = 0; i < 10000; i++)
            {
                Erorr = 0;
                for (int j = 0; j < TrainingSetArray.Length; j++)
                {
                    if (TrainingSetArray[j].OutputArray[0] != 0)
                    {
                        int index = (int)TrainingSetArray[j].OutputArray[1];
                        ConvolutionaryLayerArray[index].InputMatrix = TrainingSetArray[j].InputMatrix;
                        ConvolutionaryLayerArray[index].Calculate();

                        ConvolutionaryLayerArrayFace[index].InputMatrix = ConvolutionaryLayerArray[index].OutputMatrix;
                        ConvolutionaryLayerArrayFace[index].Calculate();


                        float[] a = ConvolutionaryLayerArrayFace[index].OutputMatrix.GetMaxIndex();
                        float max = ConvolutionaryLayerArrayFace[index].OutputMatrix.GetMax();

                        
                        Matrix E = new Matrix(ConvolutionaryLayerArrayFace[index].OutputMatrix.W, ConvolutionaryLayerArrayFace[index].OutputMatrix.H, ConvolutionaryLayerArrayFace[index].OutputMatrix.D);
                        E.matrix[(int)a[1]][(int)a[2]][(int)a[3]] = TrainingSetArray[j].OutputArray[0] - a[0];

                        ConvolutionaryLayerArrayFace[index].ToCorrectLayer(E, 2);
                        ConvolutionaryLayerArray[index].ToCorrectLayer(ConvolutionaryLayerArrayFace[index].ErrorMatrix, 2);


                        //BM = new Bitmap(new Bitmap(@"C:\Users\Kirill\Pictures\Image\9\" + j / 6 + ".png"), 40 * (index + 3), 40 * (index + 3));
                       // pictureBox1.Image = AddFrameOnImage(BM, (int)a[1] * 10, (int)a[2] * 10, ConvolutionaryLayerArrayFace[index].WFilters * 10, ConvolutionaryLayerArrayFace[index].HFilters * 10);
                        //Thread.Sleep(p);

                        Erorr += Math.Abs(max - TrainingSetArray[j].OutputArray[0]) / TrainingSetArray.Length;
                        
                    }
                    else
                    {
                        int index = (int)TrainingSetArray[j].OutputArray[1];
                        ConvolutionaryLayerArray[index].InputMatrix = TrainingSetArray[j].InputMatrix;
                        ConvolutionaryLayerArray[index].Calculate();

                        ConvolutionaryLayerArrayFace[index].InputMatrix = ConvolutionaryLayerArray[index].OutputMatrix;
                        ConvolutionaryLayerArrayFace[index].Calculate();


                        float[] a = ConvolutionaryLayerArrayFace[index].OutputMatrix.GetMaxIndex();
                        float max = ConvolutionaryLayerArrayFace[index].OutputMatrix.GetMax();


                        Matrix E = new Matrix(ConvolutionaryLayerArrayFace[index].OutputMatrix.W, ConvolutionaryLayerArrayFace[index].OutputMatrix.H, ConvolutionaryLayerArrayFace[index].OutputMatrix.D);
                        E.matrix[(int)a[1]][(int)a[2]][(int)a[3]] = TrainingSetArray[j].OutputArray[0] - a[0];

                        ConvolutionaryLayerArrayFace[index].ToCorrectLayer(E, 5);
                        ConvolutionaryLayerArray[index].ToCorrectLayer(ConvolutionaryLayerArrayFace[index].ErrorMatrix, 5);

                        //BM = new Bitmap(new Bitmap(@"C:\Users\Kirill\Pictures\Image\0\" + j / 6 + ".jpg"), 100 * (index + 3), 100 * (index + 3));
                        //pictureBox1.Image = AddFrameOnImage(BM, (int)a[1] * 10, (int)a[2] * 10, ConvolutionaryLayerArrayFace[index].WFilters * 10, ConvolutionaryLayerArrayFace[index].HFilters * 10);
                        //Thread.Sleep(p);

                        Erorr += Math.Abs(max - TrainingSetArray[j].OutputArray[0]) / TrainingSetArray.Length;
                    }
                }
                Console.WriteLine(Erorr);
                for (int j = 0; j < ConvolutionaryLayerArray.Length; j++)
                {
                    ConvolutionaryLayerArrayFace[j].SaveInFile("F" + j.ToString());
                    ConvolutionaryLayerArray[j].SaveInFile("EM" + j.ToString());
                }
            }
            for (int j = 0; j < ConvolutionaryLayerArray.Length; j++)
            {
                //ConvolutionaryLayerArray[j].SaveInFile("M" + j.ToString());
            }
        }

        public TrainingSet[] LoadImages()
        {
            Random rnd = new Random();
            TrainingSet[] TrainingSetArray = new TrainingSet[2800 * 3];
            List<TrainingSet> TrainingSetLis = new List<TrainingSet>();
            //StreamReader sr = new StreamReader(@"C:\Users\Kirill\Pictures\Image\1\Face.txt");
            for (int i = 0; i < 1400; i++)
            {
                for (int k = 0; k < 3; k++)
                {
                    TrainingSetLis.Add(new TrainingSet(new HOG(new Bitmap(new Bitmap(@"C:\Users\Kirill\Pictures\Image\0\" + (i) + ".jpg"), 70 * (k + 3), 70 * (k + 3))).HOGMatrix, new float[] { 0, k }));
                    TrainingSetLis.Add(new TrainingSet(new HOG(new Bitmap(new Bitmap(@"C:\Users\Kirill\Pictures\Image\9\" + i + ".png"), 40 * (k + 3), 40 * (k + 3))).HOGMatrix, new float[] { 1, k }));

                }
            }
            for (int i = 0; i < 2800 * 3; i++)
            {
                //int ind = (int)(rnd.NextDouble() * TrainingSetLis.Count);
                TrainingSetArray[i] = TrainingSetLis[i];
                //TrainingSetLis.RemoveAt(i);
            }
            return TrainingSetArray;
        }
        private delegate void TB(string Msg);


        Keys LastClickedKey;
        private void Form1_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.A || e.KeyCode == Keys.D)
            {
                LastClickedKey = e.KeyCode;
            }
        }
        private Bitmap MatrixToBitmap(Matrix matrix)
        {
            Bitmap bmp = new Bitmap(matrix.W, matrix.H);

            // Задаём формат Пикселя.
            PixelFormat pxf = PixelFormat.Format24bppRgb;

            // Получаем данные картинки.
            Rectangle rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
            //Блокируем набор данных изображения в памяти
            BitmapData bmpData = bmp.LockBits(rect, ImageLockMode.ReadWrite, pxf);

            // Получаем адрес первой линии.
            IntPtr ptr = bmpData.Scan0;

            // Задаём массив из Byte и помещаем в него надор данных.
            // int numBytes = bmp.Width * bmp.Height * 3; 
            //На 3 умножаем - поскольку RGB цвет кодируется 3-мя байтами
            //Либо используем вместо Width - Stride
            int numBytes = bmpData.Stride * bmp.Height;
            int widthBytes = bmpData.Stride;
            byte[] rgbValues = new byte[numBytes];

            // Копируем значения в массив.
            Marshal.Copy(ptr, rgbValues, 0, numBytes);


            int ind = 0;
            // Перебираем пикселы по 3 байта на каждый и меняем значения
            for (int counter = 0; counter < rgbValues.Length - 2; counter += 3)
            {
                float l = 0;
                for (int i = 0; i < matrix.D; i++)
                    l += matrix.matrix[ind % matrix.W][ind / matrix.W][i] * matrix.matrix[ind % matrix.W][ind / matrix.W][i];
                l = (float)Math.Sqrt(l);
                if (l > 1)
                    l = 1;
                if (l < 0)
                    l = 0;
                rgbValues[counter] = (byte)(l * 255);
                rgbValues[counter + 1] = (byte)(l * 255);
                rgbValues[counter + 2] = (byte)(l * 255);
                if (ind % matrix.W == matrix.W - 1)
                {
                    counter = bmpData.Stride * ind / matrix.W + 1;
                }
                ind++;

            }
            // Копируем набор данных обратно в изображение
            Marshal.Copy(rgbValues, 0, ptr, numBytes);

            // Разблокируем набор данных изображения в памяти.
            bmp.UnlockBits(bmpData);

            return bmp;
        }
        private Bitmap MatrixToBitmap(Matrix matrix, int d)
        {
            Bitmap bmp = new Bitmap(matrix.W, matrix.H);

            // Задаём формат Пикселя.
            PixelFormat pxf = PixelFormat.Format24bppRgb;

            // Получаем данные картинки.
            Rectangle rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
            //Блокируем набор данных изображения в памяти
            BitmapData bmpData = bmp.LockBits(rect, ImageLockMode.ReadWrite, pxf);

            // Получаем адрес первой линии.
            IntPtr ptr = bmpData.Scan0;

            // Задаём массив из Byte и помещаем в него надор данных.
            // int numBytes = bmp.Width * bmp.Height * 3; 
            //На 3 умножаем - поскольку RGB цвет кодируется 3-мя байтами
            //Либо используем вместо Width - Stride
            int numBytes = bmpData.Stride * bmp.Height;
            int widthBytes = bmpData.Stride;
            byte[] rgbValues = new byte[numBytes];

            // Копируем значения в массив.
            Marshal.Copy(ptr, rgbValues, 0, numBytes);


            int ind = 0;
            // Перебираем пикселы по 3 байта на каждый и меняем значения
            for (int counter = 0; counter < rgbValues.Length - 2; counter += 3)
            {
                float l = 0;
                l = matrix.matrix[ind % matrix.W][ind / matrix.W][d];
                if (l > 1)
                    l = 1;
                if (l < 0)
                    l = 0;
                rgbValues[counter] = (byte)(l * 255);
                rgbValues[counter + 1] = (byte)(l * 255);
                rgbValues[counter + 2] = (byte)(l * 255);
                if (ind % matrix.W == matrix.W - 1)
                {
                    counter = bmpData.Stride * ind / matrix.W + 1;
                }
                ind++;

            }
            // Копируем набор данных обратно в изображение
            Marshal.Copy(rgbValues, 0, ptr, numBytes);

            // Разблокируем набор данных изображения в памяти.
            bmp.UnlockBits(bmpData);

            return bmp;
        }
        private Bitmap MatrixToBitmap(Matrix matrix, float k)
        {
            Bitmap bmp = new Bitmap(matrix.W, matrix.H);

            // Задаём формат Пикселя.
            PixelFormat pxf = PixelFormat.Format24bppRgb;

            // Получаем данные картинки.
            Rectangle rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
            //Блокируем набор данных изображения в памяти
            BitmapData bmpData = bmp.LockBits(rect, ImageLockMode.ReadWrite, pxf);

            // Получаем адрес первой линии.
            IntPtr ptr = bmpData.Scan0;

            // Задаём массив из Byte и помещаем в него надор данных.
            // int numBytes = bmp.Width * bmp.Height * 3; 
            //На 3 умножаем - поскольку RGB цвет кодируется 3-мя байтами
            //Либо используем вместо Width - Stride
            int numBytes = bmpData.Stride * bmp.Height;
            int widthBytes = bmpData.Stride;
            byte[] rgbValues = new byte[numBytes];

            // Копируем значения в массив.
            Marshal.Copy(ptr, rgbValues, 0, numBytes);


            int ind = 0;
            // Перебираем пикселы по 3 байта на каждый и меняем значения
            for (int counter = 0; counter < rgbValues.Length - 2; counter += 3)
            {
                float l = 0;
                for (int i = 0; i < matrix.D; i++)
                    l += matrix.matrix[ind % matrix.W][ind / matrix.W][i] * matrix.matrix[ind % matrix.W][ind / matrix.W][i];
                l = (float)Math.Sqrt(l);
                l *= k;
                if (l > 1)
                    l = 1;
                if (l < 0)
                    l = 0;
                rgbValues[counter] = (byte)(l * 255);
                rgbValues[counter + 1] = (byte)(l * 255);
                rgbValues[counter + 2] = (byte)(l * 255);
                if (ind % matrix.W == matrix.W - 1)
                {
                    counter = bmpData.Stride * ind / matrix.W + 1;
                }
                ind++;

            }
            // Копируем набор данных обратно в изображение
            Marshal.Copy(rgbValues, 0, ptr, numBytes);

            // Разблокируем набор данных изображения в памяти.
            bmp.UnlockBits(bmpData);

            return bmp;
        }
        private Matrix CompressMatrix(Matrix M)
        {
            Matrix N = new Matrix(M.W / 2 + M.W % 2, M.H / 2 + M.H % 2, M.D);
            for (int i = 0; i < M.W; i++)
                for (int j = 0; j < M.H; j++)
                    for (int k = 0; k < M.D; k++)
                        N.matrix[i / 2][j / 2][k / 2] = Math.Max(N.matrix[i / 2][j / 2][k / 2], M.matrix[i][j][k]);
            return N;
        }



        public void CreatingLibrary()
        {
            Random rnd = new Random(); 
            string[] f = Directory.GetFiles(@"C:\Users\Kirill\Pictures\Noise");
            for (int j = 0; j < f.Length; j++)
            {
                Bitmap b = new Bitmap(f[j]);
                if (b.Width > 600)
                    b = new Bitmap(b, (int)(b.Width / 30) * 10, (int)(b.Height / 30) * 10);
                else
                    b = new Bitmap(b, (int)(b.Width / 10) * 10, (int)(b.Height / 10) * 10);
                b.Save(@"C:\Users\Kirill\Pictures\Image\0\" + j + ".jpg");
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {

        }

        private void button1_KeyPress(object sender, KeyPressEventArgs e)
        {
            ShowTestResults = !ShowTestResults;
        }
    }
}
