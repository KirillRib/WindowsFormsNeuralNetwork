using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace NeuralNetworkLibrary
{
    class HOG
    {
        public Matrix HOGMatrix;
        public Matrix HOGMatrixWithoutNormalization;
        public float[][] BWImage;
        public HOG(Bitmap InBitmap)
        {
            //System.DateTime DT = DateTime.Now;
            BWImage = MakeGray(InBitmap);
            ConvertBWImageToHOG(10, 10);
            //Console.WriteLine("Time: {0}s", (DateTime.Now - DT).TotalSeconds);
        }

        public void ConvertBWImageToHOG(int w, int h)
        {
            

            HOGMatrix = new Matrix(BWImage.Length / 10, BWImage[0].Length / 10, 8);


            for (int i = 1; i < BWImage.Length - 1; i++)
                for (int j = 1; j < BWImage[i].Length - 1; j++)
                {
                    float x = (BWImage[i + 1][j] - BWImage[i - 1][j]);
                    float y = (BWImage[i][j + 1] - BWImage[i][j - 1]);
                    float l = (float)Math.Sqrt(x * x + y * y);
                    int index = GetIndexOfBin(x, y, l);
                    HOGMatrix.matrix[i / 10][j / 10][index] += l;
                }

            HOGMatrixWithoutNormalization = HOGMatrix.Clone();
            

            for (int i = 0; i < HOGMatrix.W; i++)
                for (int j = 0; j < HOGMatrix.H; j++)
                {
                    float s = 0;
                    for (int k = 0; k < HOGMatrix.D; k++)
                        s += HOGMatrix.matrix[i][j][k] * HOGMatrix.matrix[i][j][k];
                    s = (float)Math.Sqrt(s);
                    if (s != 0)
                        for (int k = 0; k < HOGMatrix.D; k++)
                        {
                            HOGMatrix.matrix[i][j][k] = HOGMatrix.matrix[i][j][k] / s;
                            
                        }
                }
        }


        private int GetIndexOfBin(float x, float y, float l)
        {
            if (l == 0)
                return 0;
            x /= l;
            if (x > 0)
            {
                if (x >= 0.707106769f)
                {
                    if (y > 0)
                        return 0;
                    else
                        return 7;
                }
                else
                {
                    if (y > 0)
                        return 1;
                    else
                        return 6;
                }
            }
            else
            {
                if (x >= -0.707106769f)
                {
                    if (y > 0)
                        return 2;
                    else
                        return 5;
                }
                else
                {
                    if (y > 0)
                        return 3;
                    else
                        return 4;
                }
            }
        }



        private float[][] MakeGray(Bitmap bmp)
        {
            BWImage = new float[bmp.Width][];
            for (int i = 0; i < BWImage.Length; i++)
                BWImage[i] = new float[bmp.Height];

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
            for (int counter = 0; counter < rgbValues.Length; counter += 3)
            {
                
                BWImage[ind % BWImage.Length][ind / BWImage.Length] = (float)(rgbValues[counter] + rgbValues[counter + 1] + rgbValues[counter + 2]) / 765;
                if (ind % BWImage.Length == BWImage.Length - 1)
                {
                    counter += bmpData.Stride % 3;
                }
                ind++;

            }
            // Копируем набор данных обратно в изображение
            //Marshal.Copy(rgbValues, 0, ptr, numBytes);

            // Разблокируем набор данных изображения в памяти.
            bmp.UnlockBits(bmpData);

            return BWImage;
        }

        public Bitmap ToBitmap()
        {
            float l = 50f;
            Bitmap B = new Bitmap(30 * HOGMatrix.W + 100, 30 * HOGMatrix.H + 100);

            Graphics g = Graphics.FromImage(B);

            for (int i = 0; i < HOGMatrix.W; i++)
                for (int j = 0; j < HOGMatrix.H; j++)
                {
                    g.FillEllipse(new SolidBrush(Color.Black), i * 50, j * 100, 1, 1);
                    for (int k = 0; k < HOGMatrix.D; k++)
                    {
                        g.DrawLine(new Pen(Color.Black), i * 30 + 50, j * 30 + 50, (float)(i * 30 + l * Math.Cos((k + 1f / 16) * Math.PI / 4) * HOGMatrix.matrix[i][j][k]) + 50,
                            (float)(j * 30 + l * Math.Sin((k + 1f/16) * Math.PI / 4) * HOGMatrix.matrix[i][j][k]) + 50);
                    }
                }

            g.Dispose();

            return B;
        }
        public Bitmap ToBitmap(float l)
        {
            Bitmap B = new Bitmap(30 * HOGMatrix.W + 100, 30 * HOGMatrix.H + 100);

            Graphics g = Graphics.FromImage(B);

            for (int i = 0; i < HOGMatrix.W; i++)
                for (int j = 0; j < HOGMatrix.H; j++)
                {
                    g.FillEllipse(new SolidBrush(Color.Black), i * 50, j * 100, 1, 1);
                    for (int k = 0; k < HOGMatrix.D; k++)
                    {
                        g.DrawLine(new Pen(Color.Black), i * 30 + 50, j * 30 + 50, (float)(i * 30 + l * Math.Cos((k + 1f / 16) * Math.PI / 4) * HOGMatrix.matrix[i][j][k]) + 50,
                            (float)(j * 30 + l * Math.Sin((k + 1f / 16) * Math.PI / 4) * HOGMatrix.matrix[i][j][k]) + 50);
                    }
                }

            g.Dispose();

            return B;
        }
        public Bitmap ArrayToBitmap(float[][] matrix)
        {
            Bitmap bmp = new Bitmap(matrix.Length, matrix[0].Length);

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
                rgbValues[counter] = (byte)(matrix[ind % matrix.Length][ind / matrix.Length] * 255);
                rgbValues[counter + 1] = (byte)(matrix[ind % matrix.Length][ind / matrix.Length] * 255);
                rgbValues[counter + 2] = (byte)(matrix[ind % matrix.Length][ind / matrix.Length] * 255);
                if (ind % matrix.Length == matrix.Length - 1)
                {
                    counter = bmpData.Stride * ind / matrix.Length + 1;
                }
                ind++;

            }
            // Копируем набор данных обратно в изображение
            Marshal.Copy(rgbValues, 0, ptr, numBytes);

            // Разблокируем набор данных изображения в памяти.
            bmp.UnlockBits(bmpData);

            return bmp;
        }
    }
}
