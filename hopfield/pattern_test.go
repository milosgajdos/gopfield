package hopfield

import (
	"fmt"
	"image"
	"image/draw"
	"testing"

	"github.com/gonum/matrix/mat64"
	"github.com/stretchr/testify/assert"
)

func TestVec(t *testing.T) {
	assert := assert.New(t)

	data := []float64{1.0, 2.0}
	v := mat64.NewVector(len(data), data)
	p := &Pattern{v: v}
	assert.EqualValues(v, p.Vec())
}

func TestRawData(t *testing.T) {
	assert := assert.New(t)

	data := []float64{1.0, 2.0}
	v := mat64.NewVector(len(data), data)
	p := &Pattern{v: v}
	assert.EqualValues(p.RawData(), v.RawVector().Data)
}

func TestAt(t *testing.T) {
	assert := assert.New(t)

	data := []float64{1.0, 2.0}
	v := mat64.NewVector(len(data), data)
	p := &Pattern{v: v}

	assert.Equal(p.At(0), data[0])
}

func TestSet(t *testing.T) {
	assert := assert.New(t)

	data := []float64{1.0, 2.0}
	v := mat64.NewVector(len(data), data)
	p := &Pattern{v: v}

	val := -10.10
	err := p.Set(0, val)
	assert.InDelta(p.At(0), -1.0, 0.0001)

	val = 10.10
	err = p.Set(0, val)
	assert.InDelta(p.At(0), 1.0, 0.0001)

	i := 100
	errString := "invalid index: %d"
	err = p.Set(i, val)
	assert.EqualError(err, fmt.Sprintf(errString, i))
}

func TestLen(t *testing.T) {
	assert := assert.New(t)

	data := []float64{1.0, 2.0}
	p := Encode(data)
	assert.Equal(p.Len(), len(data))

	p = Encode([]float64{})
	assert.Equal(p.Len(), 0)

	p = &Pattern{}
	assert.Equal(p.Len(), 0)
}

func TestEncode(t *testing.T) {
	assert := assert.New(t)

	testCases := []struct {
		raw      []float64
		expected []float64
	}{
		{[]float64{1.0, -10.0, 0.0}, []float64{1.0, -1.0, -1.0}},
		{[]float64{1.0, 10.0}, []float64{1.0, 1.0}},
	}

	for _, tc := range testCases {
		res := Encode(tc.raw)
		assert.EqualValues(tc.expected, res.v.RawVector().Data)
	}
}

func TestAddNoise(t *testing.T) {
	assert := assert.New(t)

	data := []float64{1.0, 1.0, -1.0, 0.0}
	p := Encode(data)
	np := AddNoise(p, 50)

	assert.NotEqual(p.v.RawVector().Data, np)
}

func TestImage2Pattern(t *testing.T) {
	assert := assert.New(t)

	img := image.NewGray(image.Rect(0, 0, 2, 2))
	// draw a white image i.e. all pixels are set to 1
	draw.Draw(img, img.Bounds(), image.White, image.ZP, draw.Src)
	imgP := Image2Pattern(img)
	// 1-pixels are encoded to 1s
	p := Encode([]float64{1.0, 1.0, 1.0, 1.0})

	assert.Equal(imgP, p)
}

func TestPattern2Image(t *testing.T) {
	assert := assert.New(t)

	p := Encode([]float64{1.0, 1.0, -1.0, 0.0})
	resImg := Pattern2Image(p, image.Rect(0, 0, 2, 2))

	expImage := image.NewGray(image.Rect(0, 0, 2, 2))
	expImage.Pix = []byte{255, 255, 0, 0}

	assert.Equal(expImage, resImg)
}
