package hopfield

import (
	"testing"

	"github.com/gonum/matrix/mat64"
	"github.com/stretchr/testify/assert"
)

func TestNewnet(t *testing.T) {
	assert := assert.New(t)

	size := 5
	n, err := NewNet(size)
	assert.NotNil(n)
	assert.NoError(err)

	n, err = NewNet(-2)
	assert.Nil(n)
	assert.Error(err)
}

func TestWeights(t *testing.T) {
	assert := assert.New(t)

	size := 5
	n, err := NewNet(size)
	assert.NotNil(n)
	assert.NoError(err)

	w := n.Weights()
	rows, cols := w.Dims()
	assert.Equal(rows, size)
	assert.Equal(cols, size)
}

func TestStore(t *testing.T) {
	assert := assert.New(t)

	size := 4
	n, err := NewNet(size)
	assert.NotNil(n)
	assert.NoError(err)

	err = n.Store(nil)
	assert.Error(err)

	data := mat64.NewDense(1, 4, []float64{1.0, 0.0, 0.0, 1.0})
	err = n.Store(data)
	assert.NoError(err)
}

func TestEnergy(t *testing.T) {
	assert := assert.New(t)

	size := 4
	n, err := NewNet(size)
	assert.NotNil(n)
	assert.NoError(err)

	energy := n.Energy()
	assert.Equal(0.0, energy)
}
