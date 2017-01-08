package hopfield

import (
	"image"
	"image/draw"
	"math/rand"
)

// Pattern is a data pattern
type Pattern []float64

// Encode encodes data to a pattern of values: +1/-1 where non-positive items are set to -1 and returns it.
func Encode(data []float64) Pattern {
	p := make(Pattern, len(data))
	for i := 0; i < len(p); i++ {
		if data[i] <= 0.0 {
			p[i] = -1.0
		} else {
			p[i] = 1.0
		}
	}

	return p
}

// AddNoise adds random noise to pattern and returns it. Noise is added by flipping the sign of existing pattern value.
// It allows to specify the percentage of noise via pcnt parameter. AddNoise does not modify the pattern in place.
func AddNoise(p Pattern, pcnt int) Pattern {
	np := make(Pattern, len(p))
	copy(np, p)
	for i := 0; i < len(np); i++ {
		if i > (pcnt*len(np))/100 {
			break
		}
		j := rand.Intn(len(np))
		np[j] = -np[j]
	}

	return np
}

// Image2Pattern transforms img raw data into binary encoded pattern that can be used in Hopfield Network
// It first turns the image into a Grey scaled image and then encodes its pixels into binary values of -1/+1
func Image2Pattern(img image.Image) Pattern {
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

// Pattern2Image turns passed in Hopfield network pattern to a lossy Gray scaled image from pattern.
// Data to pixel transformation is lossy: non-positive elements are transformed to 0, otherwise 255
func Pattern2Image(p Pattern, r image.Rectangle) image.Image {
	// pix is a slice that contains pixels
	pix := make([]byte, len(p))
	for i := range p {
		if p[i] <= 0.0 {
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
