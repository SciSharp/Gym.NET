using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using ReinforcementLearning.GameConfigurations;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.Primitives;
using Point = SixLabors.Primitives.Point;
using Size = SixLabors.Primitives.Size;

namespace ReinforcementLearning {
    public class Imager {
        private Image[] _images;
        private Image<Rgba32> _outputImage;
        private List<Action<IImageProcessingContext>> _actions;

        public virtual Imager Load(Image[] images) {
            _images = images.Select(x => x.Clone(c => {})).ToArray();
            _outputImage = null;
            _actions = new List<Action<IImageProcessingContext>>();
            return this;
        }

        public virtual Imager ComposeFrames(int imageWidth, int imageHeight, ImageStackLayout imageStackLayout) {
            var stageFrames = _images.Length;
            _outputImage = new Image<Rgba32>(imageWidth, imageHeight);
            var singleImageWidth = imageWidth;
            var singleImageHeight = imageHeight;
            Func<int, Point> pointBuilder;

            switch (imageStackLayout) {
                case ImageStackLayout.Horizontal:
                    singleImageWidth = imageWidth / stageFrames;
                    pointBuilder = index => new Point(singleImageWidth * index, 0);
                    break;
                case ImageStackLayout.Vertical:
                    singleImageHeight = imageHeight / stageFrames;
                    pointBuilder = index => new Point(0, singleImageHeight * index);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(imageStackLayout), imageStackLayout, null);
            }

            Parallel.For(0, _images.Length, index => {
                var image = _images[index];
                image.Mutate(o => o.Resize(new Size(singleImageWidth, singleImageHeight)));
                _outputImage.Mutate(o => o.DrawImage(image, pointBuilder.Invoke(index), 1f));
            });

            return this;
        }

        public virtual Imager Crop(FramePadding framePadding) {
            Parallel.For(0, _images.Length, index => {
                var image = _images[index];
                var widthToCrop = image.Width - framePadding.Left - framePadding.Right;
                var heightToCrop = image.Height - framePadding.Top - framePadding.Bottom;
                image.Mutate(o => o.Crop(new Rectangle(framePadding.Left, framePadding.Top, widthToCrop, heightToCrop)));
            });

            return this;
        }

        public virtual Imager Resize(int newWidth, int newHeight) {
            _actions.Add(x => x.Resize(newWidth, newHeight));
            return this;
        }

        public virtual IEnumerable<float> Rectify() {
            var rgbaArray = MemoryMarshal.AsBytes(_outputImage.GetPixelSpan()).ToArray();

            if (rgbaArray.Length % 4 != 0) {
                throw new Exception("Should never happen");
            }

            for (var index = 0; index < rgbaArray.Length; index += 4) {
                //yield return (float)rgbaArray[index] / 255;
                yield return (float) rgbaArray[index] > 0 ? 1 : 0; // hardcore, fix this
            }
        }

        public virtual Imager Greyscale() {
            _actions.Add(x => x.Grayscale());
            return this;
        }

        public virtual Imager InvertColors() {
            _actions.Add(x => x.Invert());
            return this;
        }

        public virtual Imager Compile() {
            var compiledAction = _actions[0];
            for (var index = 1; index < _actions.Count; index++) {
                compiledAction += _actions[index];
            }

            _outputImage.Mutate(compiledAction);
            return this;
        }

        public virtual Image Result() =>
            Compile()._outputImage;
    }
}