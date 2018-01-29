package hopfield

import (
	"fmt"
	"image"
	"image/draw"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Pattern is a data pattern
type Pattern struct {
	// v is a vector which stores binary data
	v *mat.VecDense
}

// String implements Stringer interface
func (p *Pattern) String() string {
	fa := mat.Formatted(p.v, mat.Prefix(""), mat.Squeeze())
	return fmt.Sprintf("%v", fa)
}

// Vec returns internal data vector
func (p *Pattern) Vec() *mat.VecDense {
	return p.v
}

// RawData returns pattern raw data
func (p *Pattern) RawData() []float64 {
	return p.v.RawVector().Data
}

// At returns valir of patttern on position i
func (p *Pattern) At(i int) float64 {
	return p.v.RawVector().Data[i]
}

// Set sets value on position i
func (p *Pattern) Set(i int, val float64) error {
	if i > p.v.Len() {
		return fmt.Errorf("invalid index: %d", i)
	}
	// we transform vals to +1/-1
	if val <= 0.0 {
		p.v.RawVector().Data[i] = -1.0
	} else {
		p.v.RawVector().Data[i] = 1.0
	}

	return nil
}

// Len returns the length of the pattern
func (p *Pattern) Len() int {
	if p.v == nil {
		return 0
	}
	return p.v.Len()
}

// Encode encodes data to a pattern of values: +1/-1. Non-positive data items are set to -1, positive ones are set to +1
// Encode modifies the data slice in place and returns pointer to Pattern.
func Encode(data []float64) *Pattern {
	for i := 0; i < len(data); i++ {
		if data[i] <= 0.0 {
			data[i] = -1.0
		} else {
			data[i] = 1.0
		}
	}
	v := mat.NewVecDense(len(data), data)

	return &Pattern{
		v: v,
	}
}

// AddNoise adds random noise to pattern p and returns it. Noise is added by flipping the sign of existing pattern value.
// It allows to specify the percentage of noise via pcnt parameter. AddNoise modifies the pattern p in place.
func AddNoise(p *Pattern, pcnt int) *Pattern {
	n, _ := p.v.Dims()
	for i := 0; i < n; i++ {
		if i > (pcnt*n)/100 {
			break
		}
		j := rand.Intn(n)
		p.v.SetVec(j, -p.v.At(j, 0))
	}

	return p
}

// Image2Pattern transforms img raw data into binary encoded pattern that can be used in Hopfield Network
// It first turns the image into a Grey scaled image and then encodes its pixels into binary values of -1/+1
func Image2Pattern(img image.Image) *Pattern {
	// convert image to Gray scaled image
	imGray := image.NewGray(image.Rect(0, 0, img.Bounds().Dx(), img.Bounds().Dy()))
	draw.Draw(imGray, imGray.Bounds(), img, img.Bounds().Min, draw.Src)
	// convert pixels into floats
	pattern := make([]float64, len(imGray.Pix))
	for i := range imGray.Pix {
		pattern[i] = float64(imGray.Pix[i])
	}

	return Encode(pattern)
}

// Pattern2Image turns pattern p to a *lossy* Gray scaled image.
// Data to pixel transformation is lossy: non-positive elements are transformed to 0, otherwise 255
func Pattern2Image(p *Pattern, r image.Rectangle) image.Image {
	// pix is a slice that contains pixels
	pix := make([]byte, len(p.v.RawVector().Data))
	for i := range pix {
		if p.v.At(i, 0) <= 0.0 {
			pix[i] = 0
		} else {
			pix[i] = 255
		}
	}
	// create new Gray image from data
	img := image.NewGray(r)
	copy(img.Pix, pix)

	return img
}
