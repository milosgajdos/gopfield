package hopfield

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestNewNetwork(t *testing.T) {
	assert := assert.New(t)

	method := "storkey"
	size := 5
	n, err := NewNetwork(size, method)
	assert.NotNil(n)
	assert.NoError(err)

	errString := "invalid network size: %d"
	size = -2
	n, err = NewNetwork(size, method)
	assert.Nil(n)
	assert.EqualError(err, fmt.Sprintf(errString, size))

	errString = "unsupported training method: %s"
	size = 5
	method = "foobar"
	n, err = NewNetwork(size, method)
	assert.Nil(n)
	assert.EqualError(err, fmt.Sprintf(errString, method))
}

func TestWeights(t *testing.T) {
	assert := assert.New(t)

	method := "storkey"
	size := 5
	n, err := NewNetwork(size, method)
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
	n, err := NewNetwork(size, method)
	assert.NotNil(n)
	assert.NoError(err)

	bias := n.Bias()
	rows, cols := bias.Dims()
	assert.Equal(size, rows)
	assert.Equal(1, cols)
}

func TestCapacity(t *testing.T) {
	assert := assert.New(t)

	method := "storkey"
	size := 10
	n, err := NewNetwork(size, method)
	assert.NotNil(n)
	assert.NoError(err)
	capStorkey := n.Capacity()

	method = "hebbian"
	n2, err := NewNetwork(size, method)
	assert.NotNil(n2)
	assert.NoError(err)
	capHebbian := n2.Capacity()
	// in the same sized network storkey training provides higher capacity
	assert.True(capStorkey > capHebbian)
}

func TestMemorised(t *testing.T) {
	assert := assert.New(t)

	n, err := NewNetwork(5, "hebbian")
	assert.NotNil(n)
	assert.NoError(err)
	assert.True(n.Memorised() == 0)
}

func TestStore(t *testing.T) {
	assert := assert.New(t)

	size := 4
	// Hebbian learning
	n, err := NewNetwork(size, "hebbian")
	assert.NotNil(n)
	assert.NoError(err)

	var patterns []*Pattern
	errString := "invalid patterns supplied: %v"
	err = n.Store(patterns)
	assert.EqualError(err, fmt.Sprintf(errString, patterns))

	patterns = []*Pattern{nil}
	errString = "invalid pattern supplied: %v"
	err = n.Store(patterns)
	assert.EqualError(err, fmt.Sprintf(errString, patterns[0]))

	data := []float64{1.0, -1.0}
	v := mat.NewVecDense(len(data), data)
	patterns = []*Pattern{{v: v}}
	errString = "invalid pattern dimension: %d"
	err = n.Store(patterns)
	assert.EqualError(err, fmt.Sprintf(errString, patterns[0].Len()))

	data = []float64{1.0, -1.0, -1.0, 1.0}
	v = mat.NewVecDense(len(data), data)
	patterns = []*Pattern{{v: v}}
	err = n.Store(patterns)
	assert.NoError(err)
	assert.Equal(n.Weights().At(0, 3), n.Weights().At(3, 0))

	// Storkey learning
	n, err = NewNetwork(size, "storkey")
	assert.NotNil(n)
	assert.NoError(err)

	err = n.Store(patterns)
	assert.NoError(err)
	assert.Equal(n.Weights().At(0, 3), n.Weights().At(3, 0))
}

func TestRestore(t *testing.T) {
	assert := assert.New(t)

	size := 4
	iters := 10
	mode := "async"
	method := "hebbian"
	n, err := NewNetwork(size, method)
	assert.NotNil(n)
	assert.NoError(err)

	data := []float64{1.0, -1.0, -1.0, 1.0}
	v := mat.NewVecDense(len(data), data)
	patterns := []*Pattern{{v: v}}
	err = n.Store(patterns)
	assert.NoError(err)

	var pattern *Pattern
	errString := "invalid pattern supplied: %v"
	res, err := n.Restore(pattern, mode, iters)
	assert.Nil(res)
	assert.EqualError(err, fmt.Sprintf(errString, pattern))

	pattern = &Pattern{v: mat.NewVecDense(2, []float64{-1.0, 1.0})}
	errString = "invalid pattern dimension: %v"
	res, err = n.Restore(pattern, mode, iters)
	assert.Nil(res)
	assert.EqualError(err, fmt.Sprintf(errString, pattern.Len()))

	iters = -5
	data = []float64{1.0, -1.0, -1.0, 1.0}
	v = mat.NewVecDense(len(data), data)
	pattern = &Pattern{v: v}
	errString = "invalid number of iterations: %d"
	res, err = n.Restore(pattern, mode, iters)
	assert.Nil(res)
	assert.EqualError(err, fmt.Sprintf(errString, iters))

	iters = 1
	res, err = n.Restore(pattern, mode, iters)
	assert.NotNil(res)
	assert.NoError(err)

	mode = "sync"
	res, err = n.Restore(pattern, mode, iters)
	assert.NotNil(res)
	assert.NoError(err)

	mode = "foobar"
	errString = "unsupported mode: %s"
	res, err = n.Restore(pattern, mode, iters)
	assert.Nil(res)
	assert.EqualError(err, fmt.Sprintf(errString, mode))
}

func TestEnergy(t *testing.T) {
	assert := assert.New(t)

	size := 4
	method := "hebbian"
	n, err := NewNetwork(size, method)
	assert.NotNil(n)
	assert.NoError(err)

	var pattern *Pattern
	errString := "invalid pattern supplied: %v"
	energy, err := n.Energy(pattern)
	assert.Equal(0.0, energy)
	assert.EqualError(err, fmt.Sprintf(errString, pattern))

	pattern = &Pattern{v: mat.NewVecDense(2, []float64{1.0, -1.0})}
	errString = "invalid pattern dimension: %v"
	energy, err = n.Energy(pattern)
	assert.Equal(0.0, energy)
	assert.EqualError(err, fmt.Sprintf(errString, pattern.Len()))

	data := []float64{1.0, -1.0, -1.0, 1.0}
	v := mat.NewVecDense(len(data), data)
	pattern = &Pattern{v: v}
	energy, err = n.Energy(pattern)
	assert.Equal(0.0, energy)
	assert.NoError(err)
}
