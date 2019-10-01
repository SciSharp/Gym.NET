using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Point = SixLabors.Primitives.Point;
using Size = SixLabors.Primitives.Size;

namespace AtariDeepQLearner
{
    public class Imager
    {
        private Image<Rgba32>[] _images;
        private Image<Rgba32> _outputImage;
        private List<Action<IImageProcessingContext>> _actions;

        public virtual Imager Load(Image<Rgba32>[] images)
        {
            _images = images;
            _outputImage = null;
            _actions = new List<Action<IImageProcessingContext>>();
            return this;
        }

        public virtual Imager ComposeFrames(int imageWidth, int imageHeight)
        {
            var stageFrames = _images.Length;
            _outputImage = new Image<Rgba32>(imageWidth, imageHeight);
            var singleImageWidth = imageWidth / stageFrames;

            Parallel.For(0, _images.Length, index =>
            {
                var image = _images[index];
                image.Mutate(o => o.Resize(new Size(singleImageWidth, imageHeight)));
                _outputImage.Mutate(o => o.DrawImage(image, new Point(singleImageWidth * index, 0), 1f));
            });

            return this;
        }

        public virtual Imager Resize(int newWidth, int newHeight)
        {
            _actions.Add(x => x.Resize(newWidth, newHeight));
            return this;
        }

        public virtual IEnumerable<float> Rectify()
        {
            var rgbaArray = MemoryMarshal.AsBytes(_outputImage.GetPixelSpan()).ToArray();

            if (rgbaArray.Length % 4 != 0)
            {
                throw new Exception("Should never happen");
            }

            for (var index = 0; index < rgbaArray.Length; index += 4)
            {
                yield return (float)rgbaArray[index] / 255;
            }
        }

        public virtual Imager Grayscale()
        {
            _actions.Add(x => x.Grayscale());
            return this;
        }

        public virtual Imager InvertColors()
        {
            _actions.Add(x => x.Invert());
            return this;
        }

        public virtual Imager Compile()
        {
            var compiledAction = _actions[0];
            for (var index = 1; index < _actions.Count; index++)
            {
                compiledAction += _actions[index];
            }

            _outputImage.Mutate(compiledAction);
            return this;
        }

        public virtual Image<Rgba32> Result() =>
            Compile()._outputImage;
    }
}