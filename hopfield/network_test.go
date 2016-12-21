package hopfield

import (
	"fmt"
	"testing"

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

func TestNeurons(t *testing.T) {
	assert := assert.New(t)

	size := 5
	n, err := NewNet(size)
	assert.NotNil(n)
	assert.NoError(err)

	neurons := n.Neurons()
	assert.Equal(size, len(neurons))

}

func TestBiases(t *testing.T) {
	assert := assert.New(t)

	size := 5
	n, err := NewNet(size)
	assert.NotNil(n)
	assert.NoError(err)

	biases := n.Biases()
	rows, cols := biases.Dims()
	assert.Equal(size, rows)
	assert.Equal(1, cols)

}

func TestStore(t *testing.T) {
	assert := assert.New(t)

	size := 4
	n, err := NewNet(size)
	assert.NotNil(n)
	assert.NoError(err)

	pattern := []float64(nil)
	errString := "Invalid data supplied: %v\n"
	err = n.Store(pattern)
	assert.EqualError(err, fmt.Sprintf(errString, pattern))

	pattern = []float64{1.0, 0.0}
	errString = "Dimension mismatch: %v\n"
	err = n.Store(pattern)
	assert.EqualError(err, fmt.Sprintf(errString, pattern))

	pattern = []float64{1.0, 0.0, 0.0, 1.0}
	err = n.Store(pattern)
	assert.NoError(err)
}

func TestRestore(t *testing.T) {
	assert := assert.New(t)

	size := 4
	iters := 10
	n, err := NewNet(size)
	assert.NotNil(n)
	assert.NoError(err)

	pattern := []float64{1.0, 0.0, 0.0, 1.0}
	err = n.Store(pattern)
	assert.NoError(err)

	pattern = []float64(nil)
	errString := "Invalid data supplied: %v\n"
	res, err := n.Restore(pattern, iters)
	assert.Nil(res)
	assert.EqualError(err, fmt.Sprintf(errString, pattern))

	pattern = []float64{1.0, 0.0}
	errString = "Dimension mismatch: %v\n"
	res, err = n.Restore(pattern, iters)
	assert.Nil(res)
	assert.EqualError(err, fmt.Sprintf(errString, pattern))

	iters = -5
	pattern = []float64{1.0, 0.0, 0.0, 1.0}
	errString = "Invalid number of iterations: %d\n"
	res, err = n.Restore(pattern, iters)
	assert.Nil(res)
	assert.EqualError(err, fmt.Sprintf(errString, iters))

	iters = 10
	res, err = n.Restore(pattern, iters)
	assert.NotNil(res)
	assert.NoError(err)
}

func TestEnergy(t *testing.T) {
	assert := assert.New(t)

	size := 4
	n, err := NewNet(size)
	assert.NotNil(n)
	assert.NoError(err)

	pattern := []float64{1.0, 0.0, 0.0, 1.0}
	energy := n.Energy(pattern)
	assert.Equal(0.0, energy)
}
