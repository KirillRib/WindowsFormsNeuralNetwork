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
            videoSource = new VideoCaptureDevice(videodevices[0].MonikerString);
            videoSourcePlayer1.VideoSource = videoSource;
            new HOG(new Bitmap(new Bitmap(@"C:\Users\Kirill\Pictures\game-of-thrones_9b94_2400x1350.jpg"), 1000, 1000));



            ConvolutionaryLayerArray = new ConvolutionaryLayer[5];
            for (int i = 0; i < ConvolutionaryLayerArray.Length; i++)
            {
                //ConvolutionaryLayerArray[i] = new ConvolutionaryLayer(new Matrix[] { 50 * new HOG(new Bitmap(new Bitmap(@"C:\Users\Kirill\Pictures\000.jpg"), 100 + 20 * i, 100 + 20 * i)).HOGMatrix}, 10 + 2 * i, 10 + 2 * i);
                ConvolutionaryLayerArray[i] = new ConvolutionaryLayer(new Matrix[] { 50 * new HOG(new Bitmap(new Bitmap(@"C:\Users\Kirill\Pictures\000.jpg"), 100 + 20 * i, 100 + 20 * i)).HOGMatrix }, 10 + 2 * i, 10 + 2 * i);

            }



            for (int j = 0; j < ConvolutionaryLayerArray.Length; j++)
            {
                ConvolutionaryLayerArray[j].LoadFromFile(j.ToString());
                //foreach (Matrix M in ConvolutionaryLayerArray[j].GetArrayFilters())
                //    M.ToNormalizeVectors(100);
            }


            Thread myThread = new Thread(TreatmentImage);
            myThread.Start();


        }


        NeuralNetwork mNeuralNetwork;
        ConvolutionaryLayer[] ConvolutionaryLayerArray;
        FilterInfoCollection videodevices;
        VideoCaptureDevice videoSource;
        private void videoSourcePlayer1_NewFrame(object sender, ref Bitmap image)
        {
            //System.DateTime DT = DateTime.Now;
            //pictureBox1.SizeMode = PictureBoxSizeMode.Zoom;
            BM = new Bitmap(image, 300, 220);
        }

        ImageFormat IF = System.Drawing.Imaging.ImageFormat.Png;
        Bitmap BM;
        public float Chance = 0;
        int ind = 0;

        public bool ShowTestResults = true;

        private void SaveImage()
        {
            while (ind < 400)
            {
                string name = "C:/Users/Kirill/Pictures/Image/0_" + ind + ".png";
                if (BM != null)
                {
                    BM.Save(name, IF);
                    ind++;
                }
                    
                
                Thread.Sleep(100);
            }
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
            while (true)
            {
                if (BM != null)
                {
                    System.DateTime DT = DateTime.Now;
                    HOG hog = new HOG(BM);
                    float[] a = new float[4];
                    int index = 0;
                    Console.WriteLine("Time0: {0}s", (DateTime.Now - DT).TotalSeconds);

                    Matrix answer = new Matrix(32, 24, 1);
                    for (int k = 0; k < ConvolutionaryLayerArray.Length; k++)
                    {
                        ConvolutionaryLayerArray[k].InputMatrix = hog.HOGMatrix;
                        ConvolutionaryLayerArray[k].Calculate();
                        answer.AddPartOfMatrix(0, 0, ConvolutionaryLayerArray[k].OutputMatrix);

                        //a = ConvolutionaryLayerArray[k].OutputMatrix.GetMaxIndex();
                        //BM = AddFrameOnImage(BM, (int)a[1] * 10, (int)a[2] * 10, ConvolutionaryLayerArray[k].GetArrayFilters()[(int)a[3]].W * 10, ConvolutionaryLayerArray[k].GetArrayFilters()[(int)a[3]].H * 10);
                    }
                    Console.WriteLine("Time1: {0}s", (DateTime.Now - DT).TotalSeconds);



                    a = answer.GetMaxIndex();

                    float max = -100;
                    for (int k = 0; k < ConvolutionaryLayerArray.Length; k++)
                        if (ConvolutionaryLayerArray[k].OutputMatrix.W > (int)a[1] && ConvolutionaryLayerArray[k].OutputMatrix.H > (int)a[2] 
                            && max < ConvolutionaryLayerArray[k].OutputMatrix.matrix[(int)a[1]][(int)a[2]][(int)a[3]])
                        {
                            max = ConvolutionaryLayerArray[k].OutputMatrix.matrix[(int)a[1]][(int)a[2]][(int)a[3]];
                            index = k;
                        }
                    //BM.Save(@"C:\Users\Kirill\Pictures\0.jpg");
                    //new HOG(BM).ToBitmap(20).Save(@"C:\Users\Kirill\Pictures\1.jpg");
                    Console.WriteLine(max);
                    if (max > 0.1f)
                        BM = AddFrameOnImage(BM, (int)a[1] * 10, (int)a[2] * 10, ConvolutionaryLayerArray[index].GetArrayFilters()[(int)a[3]].W * 10, ConvolutionaryLayerArray[index].GetArrayFilters()[(int)a[3]].H * 10);

                    pictureBox2.Image = MatrixToBitmap(1 / answer.GetMax() * answer);
                    //pictureBox3.Image = MatrixToBitmap(1 / ConvolutionaryLayerArray[0].OutputMatrix.GetMax() * ConvolutionaryLayerArray[0].OutputMatrix);
                    //pictureBox4.Image = MatrixToBitmap(1 / ConvolutionaryLayerArray[1].OutputMatrix.GetMax() * ConvolutionaryLayerArray[1].OutputMatrix);
                    //pictureBox5.Image = MatrixToBitmap(1 / ConvolutionaryLayerArray[2].OutputMatrix.GetMax() * ConvolutionaryLayerArray[2].OutputMatrix);
                    //pictureBox6.Image = MatrixToBitmap(1 / ConvolutionaryLayerArray[3].OutputMatrix.GetMax() * ConvolutionaryLayerArray[3].OutputMatrix);
                    //pictureBox7.Image = MatrixToBitmap(1 / ConvolutionaryLayerArray[4].OutputMatrix.GetMax() * ConvolutionaryLayerArray[4].OutputMatrix);

                    pictureBox3.Image = MatrixToBitmap(1 / max * ConvolutionaryLayerArray[0].OutputMatrix);
                    pictureBox4.Image = MatrixToBitmap(1 / max * ConvolutionaryLayerArray[1].OutputMatrix);
                    pictureBox5.Image = MatrixToBitmap(1 / max * ConvolutionaryLayerArray[2].OutputMatrix);
                    pictureBox6.Image = MatrixToBitmap(1 / max * ConvolutionaryLayerArray[3].OutputMatrix);
                    pictureBox7.Image = MatrixToBitmap(1 / max * ConvolutionaryLayerArray[4].OutputMatrix);
                    //pictureBox7.Image = MatrixToBitmap(10 * ConvolutionaryLayerArray[5].OutputMatrix);

                    pictureBox1.Image = BM;
                    //BeginInvoke(new TB(AppText), Math.Round(CL.OutputMatrix.GetMax(), 3).ToString());
                    Console.WriteLine("Time2: {0}s", (DateTime.Now - DT).TotalSeconds);
                }
                Thread.Sleep(60);
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

                        float max = ConvolutionaryLayerArray[index].OutputMatrix.GetMax();
                        float[] a = ConvolutionaryLayerArray[index].OutputMatrix.GetMaxIndex();
                        Erorr += Math.Abs(max - TrainingSetArray[j].OutputArray[0]) / TrainingSetArray.Length;
                        Matrix E = new Matrix(ConvolutionaryLayerArray[index].OutputMatrix.W, ConvolutionaryLayerArray[index].OutputMatrix.H, ConvolutionaryLayerArray[index].OutputMatrix.D);
                        E.matrix[(int)a[1]][(int)a[2]][(int)a[3]] = TrainingSetArray[j].OutputArray[0] - a[0];
                        ConvolutionaryLayerArray[index].ToCorrectLayer(E, 2);
                        //BM = new Bitmap(new Bitmap(@"C:\Users\Kirill\Pictures\Image\3\" + j / 10 + ".jpg"), 130 + 20 * index, 130 + 20 * index);
                        //pictureBox1.Image = AddFrameOnImage(BM, (int)a[1] * 10, (int)a[2] * 10, ConvolutionaryLayerArray[index].WFilters * 10, ConvolutionaryLayerArray[index].HFilters * 10);
                        //Thread.Sleep(p);
                    }
                    else
                    {
                        Matrix answer = new Matrix(TrainingSetArray[j].InputMatrix.W, TrainingSetArray[j].InputMatrix.H, 1);
                        for (int k = 0; k < ConvolutionaryLayerArray.Length; k++)
                        {
                            if (TrainingSetArray[j].InputMatrix.W > ConvolutionaryLayerArray[k].GetArrayFilters()[0].W && TrainingSetArray[j].InputMatrix.H > ConvolutionaryLayerArray[k].GetArrayFilters()[0].H)
                            {
                                ConvolutionaryLayerArray[k].InputMatrix = TrainingSetArray[j].InputMatrix;
                                ConvolutionaryLayerArray[k].Calculate();
                                answer.AddPartOfMatrix(0, 0, ConvolutionaryLayerArray[k].OutputMatrix);

                            }

                        }
                        float[] a = answer.GetMaxIndex();

                        float max = -100;
                        int index = 0;
                        for (int k = 0; k < ConvolutionaryLayerArray.Length; k++)
                            if (ConvolutionaryLayerArray[k].OutputMatrix.W > (int)a[1] && ConvolutionaryLayerArray[k].OutputMatrix.H > (int)a[2]
                                && max < ConvolutionaryLayerArray[k].OutputMatrix.matrix[(int)a[1]][(int)a[2]][(int)a[3]])
                            {
                                max = ConvolutionaryLayerArray[k].OutputMatrix.matrix[(int)a[1]][(int)a[2]][(int)a[3]];
                                index = k;
                            }




                        Erorr += Math.Abs(max - TrainingSetArray[j].OutputArray[0]) / TrainingSetArray.Length;

                        Matrix E = new Matrix(ConvolutionaryLayerArray[index].OutputMatrix.W, ConvolutionaryLayerArray[index].OutputMatrix.H, ConvolutionaryLayerArray[index].OutputMatrix.D);


                        E.matrix[(int)a[1]][(int)a[2]][(int)a[3]] = TrainingSetArray[j].OutputArray[0] - a[0];


                        ConvolutionaryLayerArray[index].ToCorrectLayer(E, 5);
                    }
                }
                Console.WriteLine(Erorr);
                for (int j = 0; j < ConvolutionaryLayerArray.Length; j++)
                {
                    ConvolutionaryLayerArray[j].SaveInFile(j.ToString());
                }
            }
            for (int j = 0; j < ConvolutionaryLayerArray.Length; j++)
            {
                ConvolutionaryLayerArray[j].SaveInFile(j.ToString());
            }
        }

        public TrainingSet[] LoadImages()
        {
            Random rnd = new Random();
            TrainingSet[] TrainingSetArray = new TrainingSet[600 * 5];
            List<TrainingSet> TrainingSetLis = new List<TrainingSet>();
            //StreamReader sr = new StreamReader(@"C:\Users\Kirill\Pictures\Image\1\Face.txt");
            for (int i = 0; i < 300; i++)
            {
                for (int k = 0; k < 5; k++)
                {
                    TrainingSetLis.Add(new TrainingSet(new HOG(new Bitmap(new Bitmap(@"C:\Users\Kirill\Pictures\Image\0\" + (i + 150 * k) + ".jpg"), 250, 200)).HOGMatrix, new float[] { 0 }));
                    TrainingSetLis.Add(new TrainingSet(new HOG(new Bitmap(new Bitmap(@"C:\Users\Kirill\Pictures\Image\3\" + i + ".jpg"), 120 + 20 * k, 120 + 20 * k)).HOGMatrix, new float[] { 1, k }));
                }

                //string[] s = sr.ReadLine().Split();
                //Bitmap b = new Bitmap(@"C:\Users\Kirill\Pictures\Image\1\" + i + ".jpg");
                //TrainingSetLis.Add(new TrainingSet(new HOG(b).HOGMatrix, new float[] { 1, float.Parse(s[0]) / 10, float.Parse(s[1]) / 10,
                //    float.Parse(s[2]) / 10, float.Parse(s[3]) / 10}));
            }
            for (int i = 0; i < 600 * 5; i++)
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
                rgbValues[counter] = (byte)(matrix.matrix[ind % matrix.W][ind / matrix.W][0] * 255);
                rgbValues[counter + 1] = (byte)(matrix.matrix[ind % matrix.W][ind / matrix.W][0] * 255);
                rgbValues[counter + 2] = (byte)(matrix.matrix[ind % matrix.W][ind / matrix.W][0] * 255);
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
