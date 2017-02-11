package hopfield

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestChangeStateNeuron(t *testing.T) {
	assert := assert.New(t)

	n := &Neuron{state: 1.0}
	assert.True(!n.ChangeState(1.0))
	assert.True(n.ChangeState(-1.0))
}

func TestNewnet(t *testing.T) {
	assert := assert.New(t)

	method := "storkey"
	size := 5
	n, err := NewNet(size, method)
	assert.NotNil(n)
	assert.NoError(err)

	n, err = NewNet(-2, method)
	assert.Nil(n)
	assert.Error(err)

	n, err = NewNet(size, "foobar")
	assert.Nil(n)
	assert.Error(err)
}

func TestNeurons(t *testing.T) {
	assert := assert.New(t)

	method := "storkey"
	size := 5
	n, err := NewNet(size, method)
	assert.NotNil(n)
	assert.NoError(err)

	neurons := n.Neurons()
	assert.Equal(size, len(neurons))
}

func TestWeights(t *testing.T) {
	assert := assert.New(t)

	method := "storkey"
	size := 5
	n, err := NewNet(size, method)
	assert.NotNil(n)
	assert.NoError(err)

	w := n.Weights()
	rows, cols := w.Dims()
	assert.Equal(rows, size)
	assert.Equal(cols, size)
}

func TestBias(t *testing.T) {
	assert := assert.New(t)

	method := "storkey"
	size := 5
	n, err := NewNet(size, method)
	assert.NotNil(n)
	assert.NoError(err)

	bias := n.Bias()
	rows, cols := bias.Dims()
	assert.Equal(size, rows)
	assert.Equal(1, cols)
}

func TestStore(t *testing.T) {
	assert := assert.New(t)

	size := 4
	// Hebbian learning
	n, err := NewNet(size, "hebbian")
	assert.NotNil(n)
	assert.NoError(err)

	var patterns []Pattern
	errString := "invalid patterns supplied: %v"
	err = n.Store(patterns)
	assert.EqualError(err, fmt.Sprintf(errString, patterns))

	patterns = []Pattern{Pattern{1.0, -1.0}}
	errString = "pattern dimension mismatch: %v"
	err = n.Store(patterns)
	assert.EqualError(err, fmt.Sprintf(errString, patterns[0]))

	patterns = []Pattern{Pattern{1.0, -1.0, -1.0, 1.0}}
	err = n.Store(patterns)
	assert.NoError(err)
	assert.Equal(n.Weights().At(0, 3), n.Weights().At(3, 0))

	// Storkey learning
	n, err = NewNet(size, "storkey")
	assert.NotNil(n)
	assert.NoError(err)

	err = n.Store(patterns)
	assert.NoError(err)
	assert.Equal(n.Weights().At(0, 3), n.Weights().At(3, 0))
}

func TestRestore(t *testing.T) {
	assert := assert.New(t)

	size := 4
	maxiters := 10
	eqiters := 5
	method := "hebbian"
	n, err := NewNet(size, method)
	assert.NotNil(n)
	assert.NoError(err)

	patterns := []Pattern{Pattern{1.0, -1.0, -1.0, 1.0}}
	err = n.Store(patterns)
	assert.NoError(err)

	pattern := Pattern(nil)
	errString := "invalid pattern supplied: %v"
	res, err := n.Restore(pattern, maxiters, eqiters)
	assert.Nil(res)
	assert.EqualError(err, fmt.Sprintf(errString, pattern))

	pattern = Pattern{1.0, -1.0}
	errString = "dimension mismatch: %v"
	res, err = n.Restore(pattern, maxiters, eqiters)
	assert.Nil(res)
	assert.EqualError(err, fmt.Sprintf(errString, pattern))

	maxiters = -5
	pattern = Pattern{1.0, -1.0, -1.0, 1.0}
	errString = "invalid number of max iterations: %d"
	res, err = n.Restore(pattern, maxiters, eqiters)
	assert.Nil(res)
	assert.EqualError(err, fmt.Sprintf(errString, maxiters))
	maxiters = 10

	eqiters = -3
	errString = "invalid number of equilibrium iterations: %d"
	res, err = n.Restore(pattern, maxiters, eqiters)
	assert.Nil(res)
	assert.EqualError(err, fmt.Sprintf(errString, eqiters))
	eqiters = 5

	res, err = n.Restore(pattern, maxiters, eqiters)
	assert.NotNil(res)
	assert.NoError(err)

	pattern = Pattern{-1.0, -1.0, 1.0, 1.0}
	res, err = n.Restore(pattern, maxiters, eqiters)
	assert.NotNil(res)
	assert.NoError(err)
}

func TestEnergy(t *testing.T) {
	assert := assert.New(t)

	size := 4
	method := "hebbian"
	n, err := NewNet(size, method)
	assert.NotNil(n)
	assert.NoError(err)

	pattern := Pattern{1.0, -1.0, -1.0, 1.0}
	energy, err := n.Energy(pattern)
	assert.Equal(0.0, energy)
	assert.NoError(err)

	pattern = Pattern(nil)
	errString := "invalid pattern supplied: %v"
	energy, err = n.Energy(pattern)
	assert.Equal(0.0, energy)
	assert.EqualError(err, fmt.Sprintf(errString, pattern))

	pattern = Pattern{1.0, -1.0}
	errString = "dimension mismatch: %v"
	energy, err = n.Energy(pattern)
	assert.Equal(0.0, energy)
	assert.EqualError(err, fmt.Sprintf(errString, pattern))
}
