package hopfield

import (
	"image"
	"image/draw"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestEncode(t *testing.T) {
	assert := assert.New(t)

	testCases := []struct {
		p        Pattern
		expected Pattern
	}{
		{[]float64{1.0, -10.0, 0.0}, []float64{1.0, -1.0, -1.0}},
		{[]float64{1.0, 10.0}, []float64{1.0, 1.0}},
	}

	for _, tc := range testCases {
		res := Encode(tc.p)
		assert.EqualValues(tc.expected, res)
	}
}

func TestAddNoise(t *testing.T) {
	assert := assert.New(t)

	p := Pattern([]float64{1.0, 1.0, -1.0, 0.0})
	np := AddNoise(p, 50)

	assert.NotEqual(p, np)
}

func TestImage2Pattern(t *testing.T) {
	assert := assert.New(t)

	img := image.NewGray(image.Rect(0, 0, 2, 2))
	draw.Draw(img, img.Bounds(), image.White, image.ZP, draw.Src)
	p := Image2Pattern(img)

	assert.Equal(Pattern{1.0, 1.0, 1.0, 1.0}, p)
}

func TestPattern2Image(t *testing.T) {
	assert := assert.New(t)

	p := Pattern([]float64{1.0, 1.0, -1.0, 0.0})
	resImg := Pattern2Image(p, image.Rect(0, 0, 2, 2))

	expImage := image.NewGray(image.Rect(0, 0, 2, 2))
	expImage.Pix = []byte{255, 255, 0, 0}

	assert.Equal(expImage, resImg)
}
